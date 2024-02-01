import math
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Union


class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.t2t = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

    def forward(self, x, b, output_size):
        f_h = int(
            (output_size[0] + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1)
            / self.stride[0]
            + 1
        )
        f_w = int(
            (output_size[1] + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1)
            / self.stride[1]
            + 1
        )

        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        # feat shape [b*t, num_vec, ks*ks*c]
        feat = self.embedding(feat)
        # feat shape after embedding [b, t*num_vec, hidden]
        feat = feat.view(b, -1, f_h, f_w, feat.size(2))
        return feat


class SoftComp(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        x = x.view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(
            feat,
            output_size=output_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        feat = self.bias_conv(feat)
        return feat


class FusionFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=1960, t2t_params=None):
        super(FusionFeedForward, self).__init__()
        # We set hidden_dim as a default to 1960
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, dim))
        assert t2t_params is not None
        self.t2t_params = t2t_params
        self.kernel_shape = reduce(
            (lambda x, y: x * y), t2t_params["kernel_size"]
        )  # 49

    def forward(self, x, output_size):
        n_vecs = 1
        for i, d in enumerate(self.t2t_params["kernel_size"]):
            n_vecs *= int(
                (output_size[i] + 2 * self.t2t_params["padding"][i] - (d - 1) - 1)
                / self.t2t_params["stride"][i]
                + 1
            )

        x = self.fc1(x)
        b, n, c = x.size()
        normalizer = (
            x.new_ones(b, n, self.kernel_shape)
            .view(-1, n_vecs, self.kernel_shape)
            .permute(0, 2, 1)
        )
        normalizer = F.fold(
            normalizer,
            output_size=output_size,
            kernel_size=self.t2t_params["kernel_size"],
            padding=self.t2t_params["padding"],
            stride=self.t2t_params["stride"],
        )

        x = F.fold(
            x.view(-1, n_vecs, c).permute(0, 2, 1),
            output_size=output_size,
            kernel_size=self.t2t_params["kernel_size"],
            padding=self.t2t_params["padding"],
            stride=self.t2t_params["stride"],
        )

        x = (
            F.unfold(
                x / normalizer,
                kernel_size=self.t2t_params["kernel_size"],
                padding=self.t2t_params["padding"],
                stride=self.t2t_params["stride"],
            )
            .permute(0, 2, 1)
            .contiguous()
            .view(b, n, c)
        )
        x = self.fc2(x)
        return x


def window_partition(x, window_size, n_head):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, n_head, T, window_size, window_size, C//n_head)
    """
    B, T, H, W, C = x.shape
    x = x.view(
        B,
        T,
        H // window_size[0],
        window_size[0],
        W // window_size[1],
        window_size[1],
        n_head,
        C // n_head,
    )
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return windows


class SparseWindowAttention(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size=(4,4), qkv_bias=True, attn_drop=0., proj_drop=0., 
                pooling_token=True):
        super().__init__()
        assert dim % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim, qkv_bias)
        self.query = nn.Linear(dim, dim, qkv_bias)
        self.value = nn.Linear(dim, dim, qkv_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # output projection
        self.proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.window_size = window_size
        self.pooling_token = pooling_token
        if self.pooling_token:
            ks, stride = pool_size, pool_size
            self.pool_layer = nn.Conv2d(
                dim, dim, kernel_size=ks, stride=stride, padding=(0, 0), groups=dim
            )
            self.pool_layer.weight.data.fill_(1.0 / (pool_size[0] * pool_size[1]))
            self.pool_layer.bias.data.fill_(0)
        # self.expand_size = tuple(i // 2 for i in window_size)
        self.expand_size = tuple((i + 1) // 2 for i in window_size)

        if any(i > 0 for i in self.expand_size):
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[: -self.expand_size[0], : -self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[: -self.expand_size[0], self.expand_size[1] :] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0] :, : -self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0] :, self.expand_size[1] :] = 0
            masrool_k = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
            self.register_buffer(
                "valid_ind_rolled", masrool_k.nonzero(as_tuple=False).view(-1)
            )

        self.max_pool = nn.MaxPool2d(window_size, window_size, (0, 0))

    def forward(self, x, mask=None, T_ind=None, attn_mask=None):
        b, t, h, w, c = x.shape  # 20 36
        w_h, w_w = self.window_size[0], self.window_size[1]
        c_head = c // self.n_head
        n_wh = math.ceil(h / self.window_size[0])
        n_ww = math.ceil(w / self.window_size[1])
        new_h = n_wh * self.window_size[0]  # 20
        new_w = n_ww * self.window_size[1]  # 36
        pad_r = new_w - w
        pad_b = new_h - h
        # reverse order
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode="constant", value=0)
            mask = F.pad(
                mask, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode="constant", value=0
            )

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        win_q = window_partition(q.contiguous(), self.window_size, self.n_head).view(
            b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head
        )
        win_k = window_partition(k.contiguous(), self.window_size, self.n_head).view(
            b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head
        )
        win_v = window_partition(v.contiguous(), self.window_size, self.n_head).view(
            b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head
        )
        # roll_k and roll_v
        if any(i > 0 for i in self.expand_size):
            (k_tl, v_tl) = map(
                lambda a: torch.roll(
                    a, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(2, 3)
                ),
                (k, v),
            )
            (k_tr, v_tr) = map(
                lambda a: torch.roll(
                    a, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(2, 3)
                ),
                (k, v),
            )
            (k_bl, v_bl) = map(
                lambda a: torch.roll(
                    a, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(2, 3)
                ),
                (k, v),
            )
            (k_br, v_br) = map(
                lambda a: torch.roll(
                    a, shifts=(self.expand_size[0], self.expand_size[1]), dims=(2, 3)
                ),
                (k, v),
            )

            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
                lambda a: window_partition(a, self.window_size, self.n_head).view(
                    b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head
                ),
                (k_tl, k_tr, k_bl, k_br),
            )
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda a: window_partition(a, self.window_size, self.n_head).view(
                    b, n_wh * n_ww, self.n_head, t, w_h * w_w, c_head
                ),
                (v_tl, v_tr, v_bl, v_br),
            )
            rool_k = torch.cat(
                (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), 4
            ).contiguous()
            rool_v = torch.cat(
                (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows), 4
            ).contiguous()  # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            # mask out tokens in current window
            rool_k = rool_k[:, :, :, :, self.valid_ind_rolled]
            rool_v = rool_v[:, :, :, :, self.valid_ind_rolled]
            roll_N = rool_k.shape[4]
            rool_k = rool_k.view(
                b, n_wh * n_ww, self.n_head, t, roll_N, c // self.n_head
            )
            rool_v = rool_v.view(
                b, n_wh * n_ww, self.n_head, t, roll_N, c // self.n_head
            )
            win_k = torch.cat((win_k, rool_k), dim=4)
            win_v = torch.cat((win_v, rool_v), dim=4)
        else:
            win_k = win_k
            win_v = win_v

        # pool_k and pool_v
        if self.pooling_token:
            pool_x = self.pool_layer(x.view(b * t, new_h, new_w, c).permute(0, 3, 1, 2))
            _, _, p_h, p_w = pool_x.shape
            pool_x = pool_x.permute(0, 2, 3, 1).view(b, t, p_h, p_w, c)
            # pool_k
            pool_k = (
                self.key(pool_x).unsqueeze(1).repeat(1, n_wh * n_ww, 1, 1, 1, 1)
            )  # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_k = pool_k.view(
                b, n_wh * n_ww, t, p_h, p_w, self.n_head, c_head
            ).permute(0, 1, 5, 2, 3, 4, 6)
            pool_k = pool_k.contiguous().view(
                b, n_wh * n_ww, self.n_head, t, p_h * p_w, c_head
            )
            win_k = torch.cat((win_k, pool_k), dim=4)
            # pool_v
            pool_v = (
                self.value(pool_x).unsqueeze(1).repeat(1, n_wh * n_ww, 1, 1, 1, 1)
            )  # [b, n_wh*n_ww, t, p_h, p_w, c]
            pool_v = pool_v.view(
                b, n_wh * n_ww, t, p_h, p_w, self.n_head, c_head
            ).permute(0, 1, 5, 2, 3, 4, 6)
            pool_v = pool_v.contiguous().view(
                b, n_wh * n_ww, self.n_head, t, p_h * p_w, c_head
            )
            win_v = torch.cat((win_v, pool_v), dim=4)

        # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
        out = torch.zeros_like(win_q)
        l_t = mask.size(1)

        mask = self.max_pool(mask.view(b * l_t, new_h, new_w))
        mask = mask.view(b, l_t, n_wh * n_ww)
        mask = torch.sum(mask, dim=1)  # [b, n_wh*n_ww]
        for i in range(win_q.shape[0]):
            ### For masked windows
            mask_ind_i = mask[i].nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            mask_n = len(mask_ind_i)
            if mask_n > 0:
                win_q_t = win_q[i, mask_ind_i].view(
                    mask_n, self.n_head, t * w_h * w_w, c_head
                )
                win_k_t = win_k[i, mask_ind_i]
                win_v_t = win_v[i, mask_ind_i]
                # mask out key and value
                if T_ind is not None:
                    # key [n_wh*n_ww, n_head, t, w_h*w_w, c_head]
                    win_k_t = win_k_t[:, :, T_ind.view(-1)].view(
                        mask_n, self.n_head, -1, c_head
                    )
                    # value
                    win_v_t = win_v_t[:, :, T_ind.view(-1)].view(
                        mask_n, self.n_head, -1, c_head
                    )
                else:
                    win_k_t = win_k_t.view(
                        n_wh * n_ww, self.n_head, t * w_h * w_w, c_head
                    )
                    win_v_t = win_v_t.view(
                        n_wh * n_ww, self.n_head, t * w_h * w_w, c_head
                    )

                att_t = (win_q_t @ win_k_t.transpose(-2, -1)) * (
                    1.0 / math.sqrt(win_q_t.size(-1))
                )
                att_t = F.softmax(att_t, dim=-1)
                att_t = self.attn_drop(att_t)
                y_t = att_t @ win_v_t

                out[i, mask_ind_i] = y_t.view(-1, self.n_head, t, w_h * w_w, c_head)

            ### For unmasked windows
            unmask_ind_i = (mask[i] == 0).nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            # [b, n_wh*n_ww, n_head, t, w_h*w_w, c_head]
            win_q_s = win_q[i, unmask_ind_i]
            win_k_s = win_k[i, unmask_ind_i, :, :, : w_h * w_w]
            win_v_s = win_v[i, unmask_ind_i, :, :, : w_h * w_w]

            att_s = (win_q_s @ win_k_s.transpose(-2, -1)) * (
                1.0 / math.sqrt(win_q_s.size(-1))
            )
            att_s = F.softmax(att_s, dim=-1)
            att_s = self.attn_drop(att_s)
            y_s = att_s @ win_v_s
            out[i, unmask_ind_i] = y_s

        # re-assemble all head outputs side by side
        out = out.view(b, n_wh, n_ww, self.n_head, t, w_h, w_w, c_head)
        out = (
            out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(b, t, new_h, new_w, c)
        )

        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]

        # output projection
        out = self.proj_drop(self.proj(out))
        return out


# class MyConv1D(nn.Module):
#     """
#     (Linear) 1D-convolutional layer that can be reparameterised into skip (see Eq. 6 of paper).

#     Args:
#         nf (int): The number of output features.
#         nx (int): The number of input features.
#         resid_gain (float): Residual weight.
#         skip_gain (float): Skip weight, if None then defaults to standard Linear layer.
#         trainable_gains (bool): Whether or not gains are trainable.
#         init_type (one of ["orth", "id", "normal"]): Type of weight initialisation.
#         bias (bool): Whether or not to use bias parameters.
#     """

#     def __init__(
#         self,
#         nf,
#         nx,
#         resid_gain=None,
#         skip_gain=None,
#         trainable_gains=False,
#         init_type="normal",
#         bias=True,
#     ):
#         super().__init__()
#         self.nf = nf

#         if bias:
#             self.bias = nn.Parameter(torch.zeros(nf))
#         else:
#             self.bias = nn.Parameter(torch.zeros(nf), requires_grad=False)

#         if skip_gain is None:
#             # Standard linear layer
#             self.weight = nn.Parameter(torch.empty(nx, nf))
#             if init_type == "orth":
#                 nn.init.orthogonal_(self.weight)
#             elif init_type == "id":
#                 self.weight.data = torch.eye(nx)
#             elif init_type == "normal":
#                 nn.init.normal_(self.weight, std=0.02)
#             else:
#                 raise NotImplementedError
#             self.skip = False

#         elif skip_gain is not None:
#             # Reparameterised linear layer
#             assert nx == nf
#             self.resid_gain = nn.Parameter(
#                 torch.Tensor([resid_gain]), requires_grad=trainable_gains
#             )
#             self.skip_gain = nn.Parameter(
#                 torch.Tensor([skip_gain]),
#                 requires_grad=trainable_gains,
#             )

#             self.weight = nn.Parameter(torch.zeros(nx, nx))
#             if init_type == "orth":
#                 self.id = nn.init.orthogonal_(torch.empty(nx, nx)).cuda()
#             elif init_type == "id":
#                 self.id = torch.eye(nx).cuda()
#             elif init_type == "normal":
#                 self.id = nn.init.normal_(
#                     torch.empty(nx, nx), std=1 / math.sqrt(nx)
#                 ).cuda()
#             else:
#                 raise NotImplementedError
#             self.skip = True
#             self.init_type = init_type

#     def forward(self, x):
#         size_out = x.size()[:-1] + (self.nf,)
#         if self.skip:
#             if self.resid_gain == 0 and self.init_type == "id":
#                 x = torch.add(self.bias, x * self.skip_gain)
#             else:
#                 x = torch.addmm(
#                     self.bias,
#                     x.view(-1, x.size(-1)),
#                     self.resid_gain * self.weight + self.skip_gain * self.id,
#                 )
#         else:
#             x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
#         x = x.view(size_out)

#         return x


# class myGPT2Attention(nn.Module):
#     """
#     A customisable Attn sub-block that can implement Shaped Attention, and identity value/projection weights.

#         self.attention = myGPT2Attention(dim, n_head, window_size, pool_size)

#         def __init__(self, dim, n_head, window_size, pool_size=(4,4), qkv_bias=True, attn_drop=0., proj_drop=0.,pooling_token=True):


#     """
#     def __init__(self, dim, n_head, window_size, pool_size=(4,4), qkv_bias=True, attn_drop=0., proj_drop=0.,pooling_token=True, is_cross_attention=False, layer_idx=None):

#         super().__init__()
#         assert is_cross_attention == False
#         max_positions = config.max_position_embeddings

#         self.embed_dim = dim
#         self.num_heads = n_head
#         self.head_dim = self.embed_dim // self.num_heads
#         if self.head_dim * self.num_heads != self.embed_dim:
#             raise ValueError(
#                 f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
#                 f" {self.num_heads})."
#             )

#         assert self.head_dim  % self.num_heads == 0

#         self.scale_attn_weights = config.scale_attn_weights

#         self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
#         self.layer_idx = layer_idx

#         self.qk_attn = MyConv1D(
#             2 * self.embed_dim,
#             self.embed_dim,
#         )

#         if config.first_layer_value_resid_gain is not None and layer_idx == 0:
#             value_resid_gain = config.first_layer_value_resid_gain
#         else:
#             value_resid_gain = config.value_resid_gain
#         if (
#             config.value_skip_gain != 1
#             or value_resid_gain != 0
#             or config.val_init_type != "id"
#         ):
#             self.v_attn = MyConv1D(
#                 self.embed_dim,
#                 self.embed_dim,
#                 resid_gain=value_resid_gain,
#                 skip_gain=config.value_skip_gain,
#                 trainable_gains=config.trainable_value_gains,
#                 init_type=config.val_init_type,
#                 bias=False,
#             )
#         else:
#             self.v_attn = nn.Identity()

#         if (
#             config.last_layer_proj_resid_gain is not None
#             and layer_idx == config.n_layer - 1
#         ):
#             proj_resid_gain = config.last_layer_proj_resid_gain
#         else:
#             proj_resid_gain = config.proj_resid_gain
#         if (
#             config.proj_skip_gain != 1
#             or proj_resid_gain != 0
#             or config.proj_init_type != "id"
#         ):
#             self.c_proj = MyConv1D(
#                 self.embed_dim,
#                 self.embed_dim,
#                 resid_gain=proj_resid_gain,
#                 skip_gain=config.proj_skip_gain,
#                 trainable_gains=config.trainable_proj_gains,
#                 init_type=config.proj_init_type,
#                 bias=False,
#             )
#         else:
#             self.c_proj = nn.Identity()

#         self.split_size = self.embed_dim
#         query_weight, key_weight = self.qk_attn.weight.data.split(
#             self.split_size, dim=1
#         )

#         if config.query_init_std is not None:
#             query_weight.normal_(mean=0.0, std=config.query_init_std)

#         if config.key_init_std is not None:
#             key_weight.normal_(mean=0.0, std=config.key_init_std)

#         if config.val_proj_init_std is not None:
#             self.v_attn.weight.data.normal_(mean=0.0, std=config.val_proj_init_std)
#             self.c_proj.weight.data.normal_(mean=0.0, std=config.val_proj_init_std)

#         self.attn_dropout = nn.Dropout(config.attn_pdrop)
#         self.resid_dropout = nn.Dropout(config.resid_pdrop)

#         self.pruned_heads = set()

#         self.attn_mat_resid_gain = nn.Parameter(
#             config.attn_mat_resid_gain * torch.ones((1, self.num_heads, 1, 1)),
#             requires_grad=config.trainable_attn_mat_gains,
#         )
#         self.attn_mat_skip_gain = nn.Parameter(
#             config.attn_mat_skip_gain * torch.ones((1, self.num_heads, 1, 1)),
#             requires_grad=config.trainable_attn_mat_gains,
#         )

#         self.centre_attn = config.centre_attn
#         # Centered attention, from https://arxiv.org/abs/2306.17759
#         uniform_causal_attn_mat = torch.ones(
#             (max_positions, max_positions), dtype=torch.float32
#         ) / torch.arange(1, max_positions + 1).view(-1, 1)
#         self.register_buffer(
#             "uniform_causal_attn_mat",
#             torch.tril(
#                 uniform_causal_attn_mat,
#             ).view(1, 1, max_positions, max_positions),
#             persistent=False,
#         )
#         self.centre_attn_gain = nn.Parameter(
#             config.centre_attn_gain * torch.ones((1, self.num_heads, 1, 1)),
#             requires_grad=config.trainable_attn_mat_gains
#             and config.centre_attn_gain != 0,
#         )
#         self.register_buffer(
#             "diag",
#             torch.eye(max_positions).view(1, 1, max_positions, max_positions),
#             persistent=False,
#         )
#         self.register_buffer(
#             "bias",
#             torch.tril(
#                 torch.ones((max_positions, max_positions), dtype=torch.bool)
#             ).view(1, 1, max_positions, max_positions),
#             persistent=False,
#         )

#     def _attn(self, query, key, value, attention_mask=None, head_mask=None):
#         attn_weights = torch.matmul(query, key.transpose(-1, -2))

#         if self.scale_attn_weights:
#             attn_weights = attn_weights / torch.full(
#                 [],
#                 value.size(-1) ** 0.5,
#                 dtype=attn_weights.dtype,
#                 device=attn_weights.device,
#             )

#         # Layer-wise attention scaling
#         if self.scale_attn_by_inverse_layer_idx:
#             attn_weights = attn_weights / float(self.layer_idx + 1)

#         query_length, key_length = query.size(-2), key.size(-2)
#         causal_mask = self.bias[
#             :, :, key_length - query_length : key_length, :key_length
#         ]
#         mask_value = torch.finfo(attn_weights.dtype).min
#         # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
#         # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
#         mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
#             attn_weights.device
#         )
#         attn_weights = torch.where(
#             causal_mask, attn_weights.to(attn_weights.dtype), mask_value
#         )

#         if attention_mask is not None:
#             # Apply the attention mask
#             attn_weights = attn_weights + attention_mask

#         attn_weights = nn.functional.softmax(attn_weights, dim=-1)

#         # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
#         new_attn_weights = self.attn_mat_resid_gain * attn_weights.type(value.dtype)

#         if self.centre_attn:
#             post_sm_bias_matrix = (
#                 self.attn_mat_skip_gain * self.diag[:, :, :key_length, :key_length]
#             ) - self.centre_attn_gain * (
#                 self.uniform_causal_attn_mat[
#                     :, :, key_length - query_length : key_length, :key_length
#                 ]
#             )
#             new_attn_weights = new_attn_weights + post_sm_bias_matrix

#         new_attn_weights = self.attn_dropout(new_attn_weights)

#         # Mask heads if we want to
#         if head_mask is not None:
#             new_attn_weights = new_attn_weights * head_mask

#         attn_output = torch.matmul(new_attn_weights, value)

#         return attn_output, attn_weights

#     def _split_heads(self, tensor, num_heads, attn_head_size):
#         """
#         Splits hidden_size dim into attn_head_size and num_heads
#         """
#         new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
#         tensor = tensor.view(new_shape)
#         return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

#     def _merge_heads(self, tensor, num_heads, attn_head_size):
#         """
#         Merges attn_head_size dim and num_attn_heads dim into hidden_size
#         """
#         tensor = tensor.permute(0, 2, 1, 3).contiguous()
#         new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
#         return tensor.view(new_shape)

#     def forward(
#         self,
#         hidden_states: Optional[Tuple[torch.FloatTensor]],
#         layer_past: Optional[Tuple[torch.Tensor]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = False,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
#         assert encoder_hidden_states is None
#         (query, key) = self.qk_attn(hidden_states).split(self.split_size, dim=2)
#         value = self.v_attn(hidden_states)

#         query = self._split_heads(query, self.num_heads, self.head_dim)
#         key = self._split_heads(key, self.num_heads, self.head_dim)
#         value = self._split_heads(value, self.num_heads, self.head_dim)

#         if layer_past is not None:
#             past_key, past_value = layer_past
#             key = torch.cat((past_key, key), dim=-2)
#             value = torch.cat((past_value, value), dim=-2)

#         if use_cache is True:
#             present = (key, value)
#         else:
#             present = None

#         attn_output, attn_weights = self._attn(
#             query, key, value, attention_mask, head_mask
#         )

#         attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

#         proj_output = self.c_proj(attn_output)
#         proj_output = self.resid_dropout(proj_output)

#         outputs = (proj_output, present)
#         if output_attentions:
#             outputs += (attn_weights,)

#         return outputs  # a, present, (attentions)


class TemporalSparseTransformer(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        window_size,
        pool_size,
        norm_layer=nn.LayerNorm,
        t2t_params=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.attention = SparseWindowAttention(dim, n_head, window_size, pool_size)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = FusionFeedForward(dim, t2t_params=t2t_params)

    def forward(self, x, fold_x_size, mask=None, T_ind=None):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            mask: mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        B, T, H, W, C = x.shape  # 20 36

        shortcut = x
        x = self.norm1(x)
        att_x = self.attention(x, mask, T_ind)

        # FFN
        x = shortcut + att_x
        y = self.norm2(x)
        x = x + self.mlp(y.view(B, T * H * W, C), fold_x_size).view(B, T, H, W, C)

        return x


class TemporalSparseTransformerBlock(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, depths, t2t_params=None):
        super().__init__()
        blocks = []
        for i in range(depths):
            blocks.append(
                TemporalSparseTransformer(
                    dim, n_head, window_size, pool_size, t2t_params=t2t_params
                )
            )
        self.transformer = nn.Sequential(*blocks)
        self.depths = depths

    def forward(self, x, fold_x_size, l_mask=None, t_dilation=2):
        """
        Args:
            x: image tokens, shape [B T H W C]
            fold_x_size: fold feature size, shape [60 108]
            l_mask: local mask tokens, shape [B T H W 1]
        Returns:
            out_tokens: shape [B T H W C]
        """
        assert self.depths % t_dilation == 0, "wrong t_dilation input."
        T = x.size(1)
        T_ind = [torch.arange(i, T, t_dilation) for i in range(t_dilation)] * (
            self.depths // t_dilation
        )

        for i in range(0, self.depths):
            x = self.transformer[i](x, fold_x_size, l_mask, T_ind[i])

        return x


