#!/usr/bin/env python3
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class ResidualStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int, stride: int):
        super().__init__()
        blocks = [ResidualConvBlock(in_ch, out_ch, stride=stride)]
        blocks.extend(ResidualConvBlock(out_ch, out_ch, stride=1) for _ in range(depth - 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class UNetBackbone(nn.Module):
    """Residual U-Net encoder for 400x400 Hi-C O/E maps."""

    def __init__(self, in_ch: int = 1, widths=(48, 96, 192, 384, 512), depths=(3, 3, 4, 4, 2)):
        super().__init__()
        if len(widths) != len(depths):
            raise ValueError("widths and depths must have the same length")
        stages = []
        cur = in_ch
        for i, (width, depth) in enumerate(zip(widths, depths)):
            stages.append(ResidualStage(cur, width, depth=depth, stride=1 if i == 0 else 2))
            cur = width
        self.stages = nn.ModuleList(stages)
        self.out_channels = list(widths)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class UNetDecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, depth: int):
        super().__init__()
        self.stage = ResidualStage(in_ch + skip_ch, out_ch, depth=depth, stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.stage(torch.cat([x, skip], dim=1))


class TrueUNetDecoder(nn.Module):
    """Residual U-Net decoder that restores a 400x400 feature map."""

    def __init__(self, in_channels: List[int], hidden: int = 128, encoder_depths=(3, 3, 4, 4, 2), decoder_depths=(4, 4, 3, 3)):
        super().__init__()
        if len(in_channels) != len(encoder_depths):
            raise ValueError("in_channels and encoder_depths must have the same length")
        skip_channels = list(reversed(in_channels[:-1]))
        if len(decoder_depths) != len(skip_channels):
            raise ValueError("decoder_depths must match the number of decoder skip stages")
        cur_ch = in_channels[-1]
        stages = []
        for skip_ch, depth in zip(skip_channels, decoder_depths):
            stages.append(UNetDecoderStage(cur_ch, skip_ch, out_ch=skip_ch, depth=depth))
            cur_ch = skip_ch
        self.stages = nn.ModuleList(stages)
        self.out = ResidualStage(cur_ch, hidden, depth=2, stride=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        for stage, skip in zip(self.stages, reversed(feats[:-1])):
            x = stage(x, skip)
        return self.out(x)


class DiagonalProjection1D(nn.Module):
    """Extract a learnable near-diagonal band and map it to 1D genomic features."""

    def __init__(self, in_ch: int, hidden: int = 128, out_size: int = 400, diag_radius: int = 50):
        super().__init__()
        if diag_radius < 0:
            raise ValueError("diag_radius must be non-negative")
        self.out_size = out_size
        self.diag_radius = int(diag_radius)
        self.dw = 2 * self.diag_radius + 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.fc = nn.Linear(hidden * self.dw, hidden)
        pos = torch.arange(self.out_size, dtype=torch.long)
        offsets = torch.arange(-self.diag_radius, self.diag_radius + 1, dtype=torch.long)
        rows = pos.unsqueeze(0).expand(self.dw, self.out_size)
        cols = pos.unsqueeze(0) + offsets.unsqueeze(1)
        valid = (cols >= 0) & (cols < self.out_size)
        flat_idx = rows * self.out_size + cols.clamp(0, self.out_size - 1)
        self.register_buffer("band_flat_idx", flat_idx.reshape(-1), persistent=False)
        self.register_buffer("band_mask", valid.reshape(1, 1, self.dw, self.out_size), persistent=False)

    def forward(self, feat2d: torch.Tensor) -> torch.Tensor:
        if feat2d.shape[-1] != self.out_size or feat2d.shape[-2] != self.out_size:
            feat2d = F.interpolate(feat2d, (self.out_size, self.out_size), mode="bilinear", align_corners=False)
        x = self.conv(feat2d)
        bsz, channels, _h, _w = x.shape
        flat = x.flatten(2)
        idx = self.band_flat_idx.view(1, 1, -1).expand(bsz, channels, -1)
        band = flat.gather(2, idx).view(bsz, channels, self.dw, self.out_size)
        band = band * self.band_mask.to(dtype=band.dtype)
        band = band.permute(0, 2, 1, 3).reshape(bsz, self.dw * channels, self.out_size)
        return self.fc(band.transpose(1, 2))


class BoundaryDecoder(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.left_head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, 3, padding=1),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, 1),
        )
        self.right_head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, 3, padding=1),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, 1),
        )

    def forward(self, diag_feat: torch.Tensor) -> dict:
        x = self.shared(diag_feat.transpose(1, 2))
        return {
            "left_boundary": torch.sigmoid(self.left_head(x).squeeze(1)),
            "right_boundary": torch.sigmoid(self.right_head(x).squeeze(1)),
        }


class ContextTADBackboneModel(nn.Module):
    def __init__(self, hidden: int = 128, diag_radius: int = 50, backbone: str = "unet_decoder"):
        super().__init__()
        if backbone != "unet_decoder":
            raise ValueError(f"This slimmed archive only supports backbone='unet_decoder', got {backbone!r}")
        self.backbone_name = backbone
        self.diag_radius = int(diag_radius)
        self.backbone = UNetBackbone(in_ch=1)
        self.neck = TrueUNetDecoder(self.backbone.out_channels, hidden=hidden)
        self.diag_proj = DiagonalProjection1D(hidden, hidden=hidden, out_size=400, diag_radius=self.diag_radius)
        self.decoder = BoundaryDecoder(hidden=hidden)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.neck(self.backbone(x))
        diag = self.diag_proj(feat)
        return self.decoder(diag)
