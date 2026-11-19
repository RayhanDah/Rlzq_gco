"""ADR-009 D7-8 avec le vrai `torch.optim.Adam` (fusionné et non fusionné, CPU) : la décomposition du déplacement de π
en parts PPO et SIL (moments fantômes, `training.sil`) redonne le déplacement réel à 1e-5 relatif près.

En intégration et non en unitaire : le premier `torch.optim.Adam` du processus importe `torch._dynamo` (12 à 47 s,
mesuré le 2026-11-11 ; 14,3 s le 2026-11-19 sur ce laptop). Formule d'Adam vérifiée dans
`torch/optim/adam.py` (`_single_tensor_adam`, torch 2.15.0) : exp_avg.lerp_(g, 1 − β₁), exp_avg_sq ← β₂·v + (1 − β₂)·g²,
θ ← θ − (lr / (1 − β₁^t))·exp_avg / (√exp_avg_sq / √(1 − β₂^t) + ε).
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models import PolicyNet, ValueNet  # noqa: E402
from training import Collector, Learner, PPOConfig  # noqa: E402
from training.sil import SILConfig  # noqa: E402

CPU = torch.device("cpu")


class _Corridor:
    """Env jouet minimal (contrat `VecEnv`) : marche aléatoire sur 0..4, +1 terminal à 4, −1 terminal à 0."""

    obs_dim = 7

    def __init__(self, n: int) -> None:
        self._n, self.pos, self.t = n, np.full(n, 2), np.zeros(n, np.int64)

    @property
    def n_agents(self) -> int:
        return self._n

    def _obs(self) -> np.ndarray:
        o = np.zeros((self._n, 7), np.float32)
        o[np.arange(self._n), self.pos] = 1.0
        return o

    def reset(self, seed: int) -> np.ndarray:
        self.pos[:], self.t[:] = 2, 0
        return self._obs()

    def step(self, act: np.ndarray) -> tuple[Any, ...]:
        self.pos = np.clip(self.pos + np.where(act == 0, -1, np.where(act == 1, 1, 0)), 0, 4)
        self.t += 1
        term = (self.pos == 0) | (self.pos == 4)
        rew = np.where(term, np.where(self.pos == 4, 1.0, -1.0), 0.0).astype(np.float32)
        trunc = ~term & (self.t >= 3)
        done = np.flatnonzero(term | trunc)
        final = self._obs()[done]
        self.pos[done], self.t[done] = 2, 0
        return self._obs(), rew, term, trunc, {"final_obs": final, "final_agent": done.astype(np.int64)}


@pytest.mark.parametrize("fused", [False, True])
def test_decomposition_du_deplacement_avec_torch_adam(fused: bool) -> None:
    # Arrange
    torch.manual_seed(0)
    cfg = PPOConfig(total_iterations=10, minibatch_size=64, target_batch_size=128, lr_start=1e-2, lr_end=1e-2)
    policy, value = PolicyNet(7, 3, (16,)), ValueNet(7, (16,))
    params = [*policy.parameters(), *value.parameters()]
    opt = torch.optim.Adam(params, lr=1e-2, betas=(0.9, 0.999), eps=1e-5, fused=fused)
    sil = SILConfig(
        enabled=True, anneal_start_timesteps=10**12, anneal_end_timesteps=2 * 10**12, batch_size=4, capacity=64,
        horizon_steps=8, store_float16=False,
    )  # fmt: skip
    learner = Learner(policy, value, cfg, CPU, opt, sil=sil)
    assert learner.sil is not None and learner.sil.buffer is not None and learner.sil.optimizer is None
    obs = np.random.default_rng(0).normal(size=(32, 7)).astype(np.float32)
    learner.sil.buffer.add(obs, np.zeros(32, np.int16), np.ones(32, np.float32), 0)
    col = Collector(_Corridor(16), PolicyNet(7, 3, (16,)), 8, CPU, seed=1)
    for _ in range(3):
        before = [p.detach().clone() for p in policy.parameters()]
        # Act
        out = learner.update(dataclasses.replace(copy.deepcopy(col.collect(0)), policy_version=learner.version))
        # Assert
        assert learner.sil.last_disp is not None
        d_ppo, d_sil = learner.sil.last_disp
        real = [p.detach() - b for p, b in zip(policy.parameters(), before, strict=True)]
        err = math.sqrt(sum(float((a + b - r).pow(2).sum()) for a, b, r in zip(d_ppo, d_sil, real, strict=True)))
        norm = math.sqrt(sum(float(r.pow(2).sum()) for r in real))
        assert norm > 0 and err / norm <= 1e-5, err / norm
        assert out.sil["steps_done"] > 0 and 0.0 < out.sil["disp_share"] < 1.0
