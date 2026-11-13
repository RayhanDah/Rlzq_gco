"""Mini-run réel de `training.run` (T6) : RocketGPU CPU, 64 arènes, vrai Adam (fused), checkpoint, arrêt, reprise.

Sauté si RocketGPU est absent. Environnement : `ROCKETGPU_PY_DIR`, `ROCKETGPU_COLLISION_MESH_DIR`
(voir `test_env_rocketgpu.py`). Le premier `torch.optim.Adam` du processus importe `torch._dynamo` (12 à 47 s).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from env import RocketGPUSimulator, SimulatorUnavailableError  # noqa: E402
from training import CheckpointManager, RunConfig, run  # noqa: E402

try:
    _PROBE = RocketGPUSimulator(2)
    _SKIP: str | None = None
except (SimulatorUnavailableError, OSError) as exc:
    _SKIP = f"RocketGPU indisponible : {exc}"
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

CONFIG: dict[str, Any] = {
    "name": "mini",
    "seed": 1,
    "device": "cpu",
    "env": {"n_envs": 64, "n_cars": 2, "tick_skip": 8},
    "reward": {"variant": "GCO-std"},
    "initial_states": {"p_kick_start": 0.2, "p_kick_end": 0.8, "anneal_timesteps": 3072},
    "model": {"policy_hidden": [32, 32], "value_hidden": [32, 32]},
    "ppo": {"epochs": 1, "minibatch_size": 256, "target_batch_size": 512},
    "collection": {"steps_per_batch": 4, "overlap": True},  # 64 x 2 x 4 = 512 timesteps agents par lot
    "budget": {"total_timesteps": 6 * 512},
    "checkpoint": {"every_iterations": 2, "keep": 2},
}


def test_mini_run_reel_checkpoint_arret_reprise_sans_nan(tmp_path: Path) -> None:
    # Arrange
    cfg = RunConfig.from_dict(CONFIG)
    out = tmp_path / "run"
    # Act : 3 itérations, arrêt, reprise dans de nouveaux objets (nouveau simulateur), fin du budget
    first = run(cfg, out, max_iterations=3, handle_sigint=False)
    second = run(cfg, out, resume=True, handle_sigint=False)
    # Assert
    assert first.status == "paused" and first.end_iteration == 3
    assert second.status == "completed" and second.start_iteration == 3 and second.end_iteration == 6
    assert second.exact_env is True  # état du simulateur restauré bit à bit
    assert second.nan_detected == 0 and second.restores == 0 and second.decisions == []
    assert first.checkpoints[0]["iteration"] == 0 and first.checkpoints[0]["healthy"] is True  # checkpoint initial (A1)
    assert second.p_kick_final == pytest.approx(0.8)  # annealing de 6 itérations : la cible est atteinte (A1)
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5, 6]
    assert all(isinstance(r["ppo/policy_loss"], float) and isinstance(r["ppo/entropy"], float) for r in rows)
    assert all(r["perf/agent_steps_per_s"] > 0 for r in rows)
    ck = CheckpointManager(out / "checkpoints").load_latest(healthy_only=True)
    assert ck is not None and ck[1]["iteration"] == 6 and ck[1]["env"]["sim"] is not None
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["config_hash"] == cfg.trajectory_hash() and meta["torch"] == torch.__version__
