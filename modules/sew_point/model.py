import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention gate for decoder skip connections."""
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Conv2d(g_ch, inter_ch, 1, bias=False)
        self.W_x = nn.Conv2d(x_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Always express alignment as a tensor operation so dynamic ONNX exports
        # do not freeze a Python shape comparison from the probe input.
        g1 = F.interpolate(g1, size=x1.shape[2:])
        att = self.psi(self.relu(g1 + x1))
        return x * att


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ca_downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_ch
        for f in features:
            self.downs.append(ConvBlock(ch, f))
            self.ca_downs.append(ChannelAttention(f))
            ch = f

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.att_gates.append(SpatialAttention(f, f, f // 2))
            self.ups.append(ConvBlock(f * 2, f))

        self.final = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for down, ca in zip(self.downs, self.ca_downs):
            x = down(x)
            x = ca(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skips = skips[::-1]
        att_idx = 0
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            x = F.interpolate(x, size=skip.shape[2:])
            skip = self.att_gates[att_idx](x, skip)
            att_idx += 1
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return torch.sigmoid(self.final(x))
