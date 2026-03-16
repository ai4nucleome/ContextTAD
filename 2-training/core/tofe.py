from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DepthwiseMix(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dw9 = nn.Conv2d(channels, channels, kernel_size=9, padding=4, groups=channels)
        self.pw = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.dw3(x), self.dw5(x), self.dw9(x)], dim=1)
        return self.act(self.pw(y))


class _EdgeGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        gy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        edge = self.proj(gx + gy)
        return x + torch.sigmoid(self.alpha) * edge


class _FreqGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.2))
        self.gamma = nn.Parameter(torch.tensor(1.5))
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = x.float()
        f = torch.fft.rfft2(x_fft, norm="ortho")
        amp = torch.abs(f)
        amp = torch.pow(amp + 1e-6, torch.sigmoid(self.gamma) * 2.0)
        f_new = f * (amp / (torch.abs(f) + 1e-6))
        y = torch.fft.irfft2(f_new, s=x.shape[-2:], norm="ortho")
        y = y.to(dtype=x.dtype)
        y = self.proj(y)
        return x + torch.sigmoid(self.beta) * y


class TOFE(nn.Module):
    """Learnable enhancer for TAD O/E maps.

    Input:  [B, 1, H, W] in [0, 1]
    Output: [B, 3, H, W] in [0, 1]
    """

    def __init__(self, in_channels: int = 1, hidden: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden),
            nn.GELU(),
        )
        self.ms = _DepthwiseMix(hidden)
        self.edge = _EdgeGate(hidden)
        self.freq = _FreqGate(hidden)
        self.out_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"TOFE expects 4D tensor [B,1,H,W], got shape {tuple(x.shape)}")
        z = self.stem(x)
        z = self.ms(z)
        z = self.edge(z)
        z = self.freq(z)
        y = self.out_head(z)
        return torch.sigmoid(y)
