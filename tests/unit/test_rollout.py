"""Tampon de rollout (ADR-003 §6) : lignes finales des troncatures seulement, validation, double tampon."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models import PolicyNet  # noqa: E402
from training import Collector, RolloutBuffer  # noqa: E402

N, D = 3, 2


def _step(buf: RolloutBuffer, term: list[bool], trunc: list[bool], final_agent: list[int], fill: float) -> None:
    fa = np.array(final_agent, np.int64)
    buf.add(
        obs=np.full((N, D), fill, np.float32),
        act=np.zeros(N, np.int64),
        logp=np.zeros(N, np.float32),
        rew=np.zeros(N, np.float32),
        term=np.array(term),
        trunc=np.array(trunc),
        final_obs=np.stack([np.full(D, 100.0 * fill + a, np.float32) for a in fa]) if fa.size else np.zeros((0, D)),
        final_agent=fa,
    )


def test_buffer_keeps_only_truncated_non_terminal_final_rows() -> None:
    # Arrange
    buf = RolloutBuffer(n_steps=2, n_agents=N, obs_dim=D)
    # Act : t0 → agent 0 but, agent 2 tronqué ; t1 → agent 1 but et timeout (le but prime), agent 2 tronqué.
    _step(buf, [True, False, False], [False, False, True], [0, 2], fill=1.0)
    _step(buf, [False, True, False], [False, True, True], [1, 2], fill=2.0)
    batch = buf.finish(np.zeros((N, D), np.float32), np.ones(N, np.bool_), policy_version=7)
    # Assert
    assert batch.final_t.tolist() == [0, 1] and batch.final_agent.tolist() == [2, 2]
    np.testing.assert_array_equal(batch.final_obs, [[102.0, 102.0], [202.0, 202.0]])
    assert batch.policy_version == 7 and batch.obs.shape == (2, N, D)
    batch.validate()


def test_finish_rejects_incomplete_buffer_and_missing_final_rows() -> None:
    buf = RolloutBuffer(n_steps=2, n_agents=N, obs_dim=D)
    _step(buf, [False] * 3, [False] * 3, [], fill=0.0)
    with pytest.raises(RuntimeError):
        buf.finish(np.zeros((N, D), np.float32), np.ones(N, np.bool_), 0)
    # Une troncature sans ligne finale rendue par l'env : la validation refuse le lot.
    _step(buf, [False] * 3, [False, True, False], [], fill=0.0)
    with pytest.raises(ValueError):
        buf.finish(np.zeros((N, D), np.float32), np.ones(N, np.bool_), 0)


class _CountingEnv:
    """obs = numéro du step (constante par ligne) ; troncature de l'agent 0 à chaque step."""

    n_agents, obs_dim = N, D

    def __init__(self) -> None:
        self.k = 0

    def reset(self, seed: int) -> np.ndarray:
        self.k = 0
        return np.zeros((N, D), np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
        self.k += 1
        obs = np.full((N, D), float(self.k), np.float32)
        trunc = np.array([True, False, False])
        info = {"final_obs": np.full((1, D), -float(self.k), np.float32), "final_agent": np.array([0], np.int64)}
        return obs, np.zeros(N, np.float32), np.zeros(N, np.bool_), trunc, info


def test_collector_double_buffer_keeps_previous_batch_intact() -> None:
    # Arrange
    torch.manual_seed(0)
    col = Collector(_CountingEnv(), PolicyNet(D, 3, (4,)), n_steps=2, device=torch.device("cpu"), seed=0)
    # Act
    b0 = col.collect(policy_version=0)
    snapshot = b0.obs.copy()
    b1 = col.collect(policy_version=1)
    # Assert : le lot 0 n'est pas réécrit par la collecte du lot 1 ; les épisodes enjambent la frontière de lot.
    np.testing.assert_array_equal(b0.obs, snapshot)
    assert b0.obs[:, 0, 0].tolist() == [0.0, 1.0] and b1.obs[:, 0, 0].tolist() == [2.0, 3.0]
    assert b0.last_obs[0, 0] == 2.0 and b1.last_obs[0, 0] == 4.0
    assert b1.final_t.tolist() == [0, 1] and b1.final_obs[:, 0].tolist() == [-3.0, -4.0]
    assert col.agent_steps == 4 * N
