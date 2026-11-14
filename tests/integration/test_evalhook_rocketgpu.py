"""Évaluation en cours de run (`training.evalhook`) sur RocketGPU CPU : simulateur dédié réutilisé d'une évaluation à
l'autre sans changer les résultats, instantanés relisibles, durées journalisées. Sauté si RocketGPU est absent
(`ROCKETGPU_PY_DIR`, `ROCKETGPU_COLLISION_MESH_DIR`, voir `test_env_rocketgpu.py`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from actions import N_ACTIONS  # noqa: E402
from env import RocketGPUSimulator, SimulatorUnavailableError  # noqa: E402
from models import PolicyNet  # noqa: E402
from obs import obs_dim  # noqa: E402
from rewards import RewardConfig  # noqa: E402
from training.checkpoint import CheckpointIdentity  # noqa: E402
from training.evalhook import EVALS_FILE, EvalHook, EvalHookConfig  # noqa: E402

try:
    _PROBE = RocketGPUSimulator(2)
    _SKIP: str | None = None
except (SimulatorUnavailableError, OSError) as exc:
    _SKIP = f"RocketGPU indisponible : {exc}"
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

D = obs_dim(2)
CFG = EvalHookConfig(every_timesteps=1, frozen_lag_timesteps=2, episodes_random=32, episodes_frozen=32, n_envs=16)


def _hook(tmp: Path, sims: list[Any]) -> tuple[EvalHook, PolicyNet]:
    def sim_factory(n: int) -> Any:
        sims.append(RocketGPUSimulator(n, 2, 8))
        return sims[-1]

    hook = EvalHook(
        CFG,
        batch_agent_steps=1,
        run_seed=0,
        out_dir=tmp / "out",
        snapshot_dir=tmp / "snap",
        policy_factory=lambda: PolicyNet(D, N_ACTIONS, (32, 32)),
        sim_factory=sim_factory,
        identity=CheckpointIdentity.current(reward_config_hash="r", obs_dim=D, tick_skip=8),
        reward_config=RewardConfig(),
        healthy_probe=lambda: None,
    )
    torch.manual_seed(0)
    live = PolicyNet(D, N_ACTIONS, (32, 32))
    hook.bind_params(list(live.parameters()))
    return hook, live


def _rows(tmp: Path) -> list[dict[str, Any]]:
    lines = (tmp / "out" / EVALS_FILE).read_text(encoding="utf-8").splitlines()
    return [r for r in (json.loads(x) for x in lines) if r["event"] == "eval"]


def test_simulateur_reutilise_resultats_identiques_a_un_simulateur_neuf(tmp_path: Path) -> None:
    # Arrange : hook A réutilise son simulateur pour 3 évaluations ; hook B (neuf) ne joue que la 3e
    sims_a: list[Any] = []
    a, _ = _hook(tmp_path / "a", sims_a)
    for it in (1, 2, 3):
        a(it)
    sims_b: list[Any] = []
    b, _ = _hook(tmp_path / "b", sims_b)
    # Act : B rejoue les mêmes indices d'évaluation avec un simulateur neuf à chaque fois
    for it in (1, 2, 3):
        b._sim = None
        b(it)
    # Assert
    ra, rb = _rows(tmp_path / "a"), _rows(tmp_path / "b")
    assert len(sims_a) == 1 and len(sims_b) == 3
    for x, y in zip(ra, rb, strict=True):
        for side in ("random", "frozen"):
            keys = ("wins", "draws", "losses", "minutes")
            assert {k: x[side][k] for k in keys} == {k: y[side][k] for k in keys}
    assert all(r["timing"]["total_s"] > 0 for r in ra)
