"""Compact TCN encoder over ordered historical movements."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.n1 = nn.GroupNorm(8, channels)
        self.n2 = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout(dropout)
        self.crop = pad

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        # Causal left-pad equivalent: drop the extra right tail from explicit padding.
        if self.crop <= 0:
            return x
        return x[..., : -self.crop]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)  mask: (B, L) True=valid
        m = mask.unsqueeze(1).to(x.dtype)
        y = self._crop(self.conv1(x * m))
        y = self.drop(F.gelu(self.n1(y)))
        y = self._crop(self.conv2(y * m))
        y = self.drop(F.gelu(self.n2(y)))
        return (x + y) * m


class TCNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        dilations: tuple[int, ...],
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, hidden, kernel_size=1)
        self.proj_n = nn.GroupNorm(8, hidden)
        self.blocks = nn.ModuleList(
            [TemporalBlock(hidden, kernel_size, d, dropout) for d in dilations]
        )
        self.out_dim = hidden * 3

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        h = F.gelu(self.proj_n(self.proj(x.transpose(1, 2))))
        for blk in self.blocks:
            h = blk(h, mask)
        # h: (B, C, L)
        m = mask.to(h.dtype).unsqueeze(1)
        h_m = h * m
        summed = h_m.sum(dim=-1)
        denom = m.sum(dim=-1).clamp_min(1.0)
        mean = summed / denom
        # masked max
        neg = torch.finfo(h.dtype).min
        mx = torch.where(mask.unsqueeze(1), h, torch.full_like(h, neg)).max(dim=-1).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        # last valid (most recent) timestep
        lengths = mask.long().sum(dim=1).clamp(min=1)
        last_i = (lengths - 1).clamp(min=0)
        last = h[torch.arange(h.size(0), device=h.device), :, last_i]
        return torch.cat([last, mean, mx], dim=-1)
