"""Politique PPO v0 : MLP 3×256 → logits sur la table d'actions discrète (ADR-003 §4, `docs/01_papers.md` §2).

Initialisation (`docs/01_papers.md` §1.1, « 37 details ») : orthogonale, gain √2 sur les couches cachées, gain 0,01 sur
la tête (politique initiale quasi uniforme sur les 90 actions), biais nuls. Activations ReLU, comme le prototype mesuré
(`scripts/bench_learner.py`, ADR-001 M3/M4).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

N_ACTIONS = 90  # ADR-003 §4 ; `actions.table` (T1) fait foi, ce module ne l'importe pas pour rester autonome
POLICY_HIDDEN: tuple[int, ...] = (256, 256, 256)
HIDDEN_GAIN = math.sqrt(2.0)
POLICY_HEAD_GAIN = 0.01


def init_linear(layer: nn.Linear, gain: float) -> None:
    """Init orthogonale de gain `gain`, biais nul."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def build_mlp(n_in: int, hidden: Sequence[int], n_out: int, head_gain: float) -> nn.Sequential:
    """MLP Linear/ReLU ; couches cachées gain √2, dernière couche gain `head_gain`."""
    if n_in <= 0 or n_out <= 0 or any(h <= 0 for h in hidden):
        raise ValueError(f"dimensions invalides : n_in={n_in}, hidden={tuple(hidden)}, n_out={n_out}")
    layers: list[nn.Module] = []
    d = n_in
    for h in hidden:
        lin = nn.Linear(d, h)
        init_linear(lin, HIDDEN_GAIN)
        layers += [lin, nn.ReLU()]
        d = h
    head = nn.Linear(d, n_out)
    init_linear(head, head_gain)
    layers.append(head)
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    """π(a | s) : `forward(obs (B, D) f32) -> logits (B, n_actions) f32`."""

    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS, hidden: Sequence[int] = POLICY_HIDDEN) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.net = build_mlp(obs_dim, hidden, n_actions, POLICY_HEAD_GAIN)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(obs)
        return out


def log_prob_entropy(logits: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """log π(act | s) et entropie H(π(· | s)) pour des logits (B, A) et des actions (B,) i64.

    `gather` sur la dernière dimension : même opération que `Categorical.log_prob` (utilisée par le prototype sur XPU).
    """
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, act.unsqueeze(-1)).squeeze(-1)
    entropy = -(logp_all.exp() * logp_all).sum(-1)
    return logp, entropy


def sample_actions(logits: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Tire a ~ π(· | s) (B,) i64 et renvoie (a, log π(a | s)).

    `torch.multinomial` (même noyau que `Categorical.sample`) avec un générateur explicite : la collecte a son propre
    flux aléatoire, indépendant de celui du thread learner (reproductibilité en recouvrement).
    """
    probs = torch.softmax(logits, dim=-1)
    act = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    logp, _ = log_prob_entropy(logits, act)
    return act, logp
