import torch
import torch.nn as nn

from RAFT import RAFT


def initialize_RAFT(model_path='weights/raft.pth', device='cuda'):
    """Initializes the RAFT model."""
    model = RAFT()
    ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
    # checkpoint was saved from a DataParallel wrapper; strip the prefix
    model.load_state_dict({k.removeprefix('module.'): v for k, v in ckpt.items()})
    model.to(device)

    return model


class RAFT_bi(nn.Module):
    """Bidirectional flow estimation with a frozen RAFT."""
    def __init__(self, model_path='weights/raft.pth', device='cuda'):
        super().__init__()
        self.raft = initialize_RAFT(model_path, device=device)

        for p in self.raft.parameters():
            p.requires_grad = False

        self.eval()

    def forward(self, gt_local_frames, iters=20):
        b, l_t, c, h, w = gt_local_frames.size()

        with torch.no_grad():
            gtlf_1 = gt_local_frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            gtlf_2 = gt_local_frames[:, 1:, :, :, :].reshape(-1, c, h, w)

            _, gt_flows_forward = self.raft(gtlf_1, gtlf_2, iters=iters, test_mode=True)
            _, gt_flows_backward = self.raft(gtlf_2, gtlf_1, iters=iters, test_mode=True)

        gt_flows_forward = gt_flows_forward.view(b, l_t-1, 2, h, w)
        gt_flows_backward = gt_flows_backward.view(b, l_t-1, 2, h, w)

        return gt_flows_forward, gt_flows_backward
