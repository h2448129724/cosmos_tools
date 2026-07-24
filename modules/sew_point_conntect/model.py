from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act(out)
        return out


class EdgeGraphNet(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.patch_encoder = nn.Sequential(
            ConvNormAct(3, 24, kernel_size=3, stride=1),
            ResidualConvBlock(24),
            ConvNormAct(24, 48, kernel_size=3, stride=2),
            ResidualConvBlock(48),
            ConvNormAct(48, 64, kernel_size=3, stride=2),
            ResidualConvBlock(64),
            ConvNormAct(64, 96, kernel_size=3, stride=2),
            ResidualConvBlock(96),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.msg_layers = nn.ModuleList(
            [MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.update_layers = nn.ModuleList(
            [MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_x, edge_index, edge_attr, edge_patch):
        h = self.node_proj(node_x)
        e = self.edge_proj(edge_attr)
        p = self.patch_encoder(edge_patch)
        src, dst = edge_index

        for msg_layer, update_layer, norm_layer in zip(self.msg_layers, self.update_layers, self.norm_layers):
            msg_to_dst = msg_layer(torch.cat([h[src], e], dim=1))
            msg_to_src = msg_layer(torch.cat([h[dst], e], dim=1))

            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg_to_dst)
            agg.index_add_(0, src, msg_to_src)

            deg = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
            ones = torch.ones(src.shape[0], device=h.device, dtype=h.dtype)
            deg.index_add_(0, src, ones)
            deg.index_add_(0, dst, ones)
            agg = agg / deg.clamp_min(1.0).unsqueeze(1)

            update = update_layer(torch.cat([h, agg], dim=1))
            h = norm_layer(h + update)

        edge_feat = torch.cat([h[src], h[dst], torch.abs(h[src] - h[dst]), e, p], dim=1)
        logits = self.edge_head(edge_feat).squeeze(1)
        return logits
