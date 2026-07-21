import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from einops import rearrange

from model.modules.base_module import BaseNetwork
from model.modules.sparse_transformer import (
    TemporalSparseTransformer,
    SoftSplit,
    SoftComp,
)

from model.modules.flow_loss_utils import flow_warp
from model.modules.deformconv import ModulatedDeformConv2d
from model.sr_backbone_utils import make_layer, ResidualBlockNoBN
from model.upsample import PixelShufflePack

from .misc import constant_init


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels)
        )

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)


def length_sq(x):
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def fbConsistencyCheck(flow_fw, flow_bw, alpha1=0.01, alpha2=0.5):
    # Ensure no NaN or Inf values in the flow fields
    flow_fw = torch.nan_to_num(flow_fw)
    flow_bw = torch.nan_to_num(flow_bw)

    flow_fw = torch.clamp(flow_fw, -1000, 1000)
    flow_bw = torch.clamp(flow_bw, -1000, 1000)

    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1))  # wb(wf(x))
    flow_diff_fw = flow_fw + flow_bw_warped  # wf + wb(wf(x))

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)  # |wf| + |wb(wf(x))|
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

    fb_valid_fw = (length_sq(flow_diff_fw) < occ_thresh_fw).to(flow_fw)

    # Apply morphological operations to clean up the validity mask
    fb_valid_fw = F.max_pool2d(fb_valid_fw, kernel_size=3, stride=1, padding=1)

    return fb_valid_fw


class DeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""

    def __init__(self, *args, **kwargs):
        # self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 3)

        super(DeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2 + 1 + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, cond_feat, flow):
        out = self.conv_offset(cond_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel, learnable=True):
        super(BidirectionalPropagation, self).__init__()
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        self.prop_list = ["backward_1", "forward_1"]
        self.learnable = learnable

        if self.learnable:
            for i, module in enumerate(self.prop_list):
                self.deform_align[module] = DeformableAlignment(
                    channel, channel, 3, padding=1, deform_groups=16
                )

                self.backbone[module] = nn.Sequential(
                    nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(channel, channel, 3, 1, 1),
                )

            self.fuse = nn.Sequential(
                nn.Conv2d(2 * channel + 2, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

    def binary_mask(self, mask, th=0.1):
        mask[mask > th] = 1
        mask[mask <= th] = 0
        return mask.to(mask)

    def forward(self, x, flows_forward, flows_backward, mask, interpolation="bilinear"):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """

        # For backward warping
        # pred_flows_forward for backward feature propagation
        # pred_flows_backward for forward feature propagation
        b, t, c, h, w = x.shape
        feats, masks = {}, {}
        feats["input"] = [x[:, i, :, :, :] for i in range(0, t)]
        masks["input"] = [mask[:, i, :, :, :] for i in range(0, t)]

        prop_list = ["backward_1", "forward_1"]
        cache_list = ["input"] + prop_list

        for p_i, module_name in enumerate(prop_list):
            feats[module_name] = []
            masks[module_name] = []

            if "backward" in module_name:
                frame_idx = range(0, t)
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                frame_idx = range(0, t)
                flow_idx = range(-1, t - 1)
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_idx):
                feat_current = feats[cache_list[p_i]][idx]
                mask_current = masks[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop = flows_for_prop[:, flow_idx[i], :, :, :]
                    flow_check = flows_for_check[:, flow_idx[i], :, :, :]
                    flow_valid_mask = fbConsistencyCheck(flow_prop, flow_check)
                    feat_warped = flow_warp(
                        feat_prop, flow_prop.permute(0, 2, 3, 1), interpolation
                    )

                    if self.learnable:
                        cond = torch.cat(
                            [
                                feat_current,
                                feat_warped,
                                flow_prop,
                                flow_valid_mask,
                                mask_current,
                            ],
                            dim=1,
                        )
                        feat_prop = self.deform_align[module_name](
                            feat_prop, cond, flow_prop
                        )
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = flow_warp(
                            mask_prop, flow_prop.permute(0, 2, 3, 1)
                        )
                        mask_prop_valid = self.binary_mask(mask_prop_valid)
                        mask_prop_valid = F.max_pool2d(mask_prop_valid, kernel_size=3, stride=1, padding=1)

                        union_valid_mask = self.binary_mask(
                            mask_current * flow_valid_mask * (1 - mask_prop_valid)
                        )
                        feat_prop = (
                            union_valid_mask * feat_warped
                            + (1 - union_valid_mask) * feat_current
                        )
                        mask_prop = self.binary_mask(
                            mask_current
                            * (1 - (flow_valid_mask * (1 - mask_prop_valid)))
                        )

                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop, mask_current], dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)

                feats[module_name].append(feat_prop)
                masks[module_name].append(mask_prop)

            if "backward" in module_name:
                feats[module_name] = feats[module_name][::-1]
                masks[module_name] = masks[module_name][::-1]

        outputs_b = torch.stack(feats["backward_1"], dim=1).view(-1, c, h, w)
        outputs_f = torch.stack(feats["forward_1"], dim=1).view(-1, c, h, w)

        if self.learnable:
            mask_in = mask.view(-1, 2, h, w)
            masks_b, masks_f = None, None
            outputs = self.fuse(
                torch.cat([outputs_b, outputs_f, mask_in], dim=1)
            ) + x.reshape(-1, c, h, w)
        else:
            masks_b = torch.stack(masks["backward_1"], dim=1)
            masks_f = torch.stack(masks["forward_1"], dim=1)
            outputs = outputs_f

        return (
            outputs.view(b, -1, c, h, w),
            masks_f,
        )


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

    def forward(self, x):
        bt, c, _, _ = x.size()
        # h, w = h//4, w//4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        return out


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.conv(x)


class DaBiT(BaseNetwork):
    def __init__(self):
        super(DaBiT, self).__init__()
        channel = 128
        hidden = 512
        self.mid_channels = 64
        self.num_blocks = 7
        self.cpu_cache_length = 100
        self.cpu_cache = False

        # encoder
        self.encoder = Encoder()

        # decoder
        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

        # soft split and soft composition
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        t2t_params = {"kernel_size": kernel_size, "stride": stride, "padding": padding}
        self.ss = SoftSplit(channel, hidden, kernel_size, stride, padding)
        self.sc = SoftComp(channel, hidden, kernel_size, stride, padding)
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

        # feature propagation module
        self.img_prop_module = BidirectionalPropagation(3, learnable=False)
        self.feat_prop_module = BidirectionalPropagation(128, learnable=True)

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * self.mid_channels,
                self.mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=10,
            )
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * self.mid_channels, self.mid_channels, self.num_blocks
            )

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(128, self.mid_channels, 5)
        self.upsample1 = PixelShufflePack(
            self.mid_channels, self.mid_channels, 2, upsample_kernel=3
        )
        self.upsample2 = PixelShufflePack(self.mid_channels, 64, 2, upsample_kernel=3)
        self.upsample3 = PixelShufflePack(64, 64, 2, upsample_kernel=3)
        # self.upsample4 = PixelShufflePack(64, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # feature extraction
        self.feat_extract = ResidualBlocksWithInputConv(3, self.mid_channels, 5)

        depths = 8
        num_heads = 4
        window_size = (5, 9)
        pool_size = (4, 4)
        self.transformers = TemporalSparseTransformer(
            dim=hidden,
            n_head=num_heads,
            window_size=window_size,
            pool_size=pool_size,
            depths=depths,
            t2t_params=t2t_params,
        )

        # print network parameter number
        self.print_network()

    def img_propagation(
        self, blurry_frames, flows, masks, interpolation="nearest"
    ):

        prop_frames, updated_masks = self.img_prop_module(
            blurry_frames, flows[0], flows[1], masks, interpolation
        )

        return prop_frames, updated_masks

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 2h, 2w).

        """

        outputs = []

        for i in range(0, lqs.size(1)):
            hr = feats[:, i, :, :, :]  # [b, 128, 60, 108]
            # hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()
            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.upsample3(hr))
            # hr = self.lrelu(self.upsample4(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            hr += self.img_upsample(lqs[:, i, :, :, :])

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(
        self,
        blurry_frames,
        depths,
        flows,
        blur_maps,
        bin_masks,
        num_local_frames,
        interpolation="bilinear",
        t_dilation=1,
    ):
        """
        Args:
            blur_maps: blur masks
            bin_masks: updated binary masks after image propagation
        """

        l_t = num_local_frames
        b, t, _, h_in, w_in = blurry_frames.size()

        # extracting features
        enc_feat = self.encoder(
            torch.cat(
                [
                    blurry_frames.view(b * t, 3, h_in, w_in),
                    blur_maps.view(b * t, 1, h_in, w_in),
                    bin_masks.view(b * t, 1, h_in, w_in),
                    depths.view(b * t, 1, h_in, w_in),
                ],
                dim=1,
            )
        )
        _, c, h, w = enc_feat.size()
        fold_feat_size = (h, w)

        # flow and mask
        ds_flows_f = (
            F.interpolate(
                flows[0].view(-1, 2, h_in, w_in),
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
            ).view(b, -1, 2, h, w)
            / 4.0
        )
        ds_flows_b = (
            F.interpolate(
                flows[1].view(-1, 2, h_in, w_in),
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
            ).view(b, -1, 2, h, w)
            / 4.0
        )

        ds_blur_maps = F.interpolate(
            blur_maps.reshape(-1, 1, h_in, w_in), scale_factor=0.25, mode="nearest"
        ).view(b, t, 1, h, w)

        ds_bin_masks = F.interpolate(
            bin_masks.reshape(-1, 1, h_in, w_in),
            scale_factor=0.25,
            mode="nearest",
        ).view(b, -1, 1, h, w)

        mask_pool_l = self.max_pool(ds_blur_maps.view(-1, 1, h, w))
        mask_pool_l = mask_pool_l.view(
            b, t, 1, mask_pool_l.size(-2), mask_pool_l.size(-1)
        )

        prop_mask_in = torch.cat([ds_blur_maps, ds_bin_masks], dim=2)
        enc_feat, _ = self.feat_prop_module(
            enc_feat.view(b, t, c, h, w), ds_flows_f, ds_flows_b, prop_mask_in, interpolation
        )

        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_feat_size)
        mask_pool_l = mask_pool_l.permute(0,1,3,4,2).contiguous()

        # transformer
        trans_feat = self.transformers(
            trans_feat, fold_feat_size, mask_pool_l, t_dilation=t_dilation
        )
        trans_feat = self.sc(trans_feat, t, fold_feat_size)
        trans_feat = trans_feat.view(b, t, -1, h, w)

        # residual
        enc_feat = enc_feat.view(b, t, -1, h, w) + trans_feat

        output = self.decoder(enc_feat.view(-1, c, h, w))
        output = torch.tanh(output).view(b, t, 3, h_in, w_in)


        # ------------------------------------
        #       Super Resolution
        # ------------------------------------
        sr_output = self.upsample(output, enc_feat)

        return sr_output

class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )
