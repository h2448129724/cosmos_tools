import torch
import torch.nn as nn


def _make_norm(norm: str, num_channels: int) -> nn.Module:
    norm = (norm or "bn").lower()
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm == "gn":
        # 小 batch（例如 1-4）时比 BN 稳定
        groups = 8
        if num_channels < groups:
            groups = 1
        while num_channels % groups != 0 and groups > 1:
            groups //= 2
        return nn.GroupNorm(groups, num_channels)
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    raise ValueError(f"Unsupported norm: {norm}")


class MicroDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm: str = "bn", dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _make_norm(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            _make_norm(norm, out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class MicroUNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, norm: str = "bn", dropout: float = 0.0):
        super().__init__()
        self.down1 = MicroDoubleConv(in_channels, 16, norm=norm, dropout=dropout)
        self.down2 = MicroDoubleConv(16, 32, norm=norm, dropout=dropout)
        self.down3 = MicroDoubleConv(32, 64, norm=norm, dropout=dropout)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = MicroDoubleConv(32+64, 32, norm=norm, dropout=dropout)
        self.up1 = MicroDoubleConv(16+32, 16, norm=norm, dropout=dropout)
        self.final = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))

        x = self.up(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        return self.final(x)
