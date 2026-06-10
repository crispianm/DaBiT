import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (differentiable L1 variant), used for the
    super-resolution term in the DaBiT training objective."""

    def __init__(self, epsilon=0.001):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, gt):
        return torch.mean(torch.sqrt((output - gt) ** 2 + self.epsilon**2))
