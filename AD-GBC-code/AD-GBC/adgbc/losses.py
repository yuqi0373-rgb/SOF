import torch
from torch import nn
from torch.nn import functional as F


class BCEDiceLoss(nn.Module):
    def forward(self, logits, target):
        probability = logits.sigmoid().flatten(1)
        truth = target.flatten(1)
        intersection = (probability * truth).sum(1)
        dice = (2 * intersection + 1e-5) / (probability.sum(1) + truth.sum(1) + 1e-5)
        return .5 * F.binary_cross_entropy_with_logits(logits, target) + 1 - dice.mean()
