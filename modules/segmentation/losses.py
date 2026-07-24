import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)[:, 1]
        targets_f = targets.float()
        intersection = (probs * targets_f).sum()
        union = probs.sum() + targets_f.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


def dice_coef(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)[:, 1]
    targets_f = targets.float()
    inter = (probs * targets_f).sum()
    union = probs.sum() + targets_f.sum()
    return (2 * inter + smooth) / (union + smooth)


class MultiLabelDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets_f = targets.float()
        dims = (0, 2, 3)
        intersection = (probs * targets_f).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_f.sum(dim=dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def multilabel_dice_coef(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets_f = targets.float()
    dims = (0, 2, 3)
    inter = (probs * targets_f).sum(dim=dims)
    union = probs.sum(dim=dims) + targets_f.sum(dim=dims)
    dice = (2 * inter + smooth) / (union + smooth)
    return dice.mean()
