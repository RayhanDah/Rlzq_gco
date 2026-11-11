"""Critique PPO v0 : MLP 3×512 → V(s), séparé de la politique (`docs/01_papers.md` §1.5, §2 ; WM §3.2, PPG §1).

Initialisation orthogonale gain √2, tête gain 1 (« 37 details », `docs/research/paper_logo_2202_and_refs.md` §12).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from models.policy import build_mlp

VALUE_HIDDEN: tuple[int, ...] = (512, 512, 512)
VALUE_HEAD_GAIN = 1.0


class ValueNet(nn.Module):
    """V(s) : `forward(obs (B, D) f32) -> (B,) f32`."""

    def __init__(self, obs_dim: int, hidden: Sequence[int] = VALUE_HIDDEN) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.net = build_mlp(obs_dim, hidden, 1, VALUE_HEAD_GAIN)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(obs).squeeze(-1)
        return out
