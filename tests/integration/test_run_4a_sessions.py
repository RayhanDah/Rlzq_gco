"""Lanceur des runs 4a (`scripts/run_4a.py`) sur la vraie boucle avec simulateur factice : deux sessions enchaînées
reprennent le même run et donnent les mêmes poids qu'une session unique ; l'échéance murale arrête proprement."""

from __future__ import annotations

import dataclasses
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "unit"))

from test_env import fake_obs  # noqa: E402
from test_run import StatefulFakeSim, adam_factory, cfg_of, make_factory  # noqa: E402
from test_run_4a import r4  # noqa: E402

from training.checkpoint import CheckpointManager  # noqa: E402

POOL = {"past_fraction": 0.5, "snapshot_every": 2, "max_size": 2, "quality_lr": 0.01}
EVAL_YAML = "every_timesteps: 48\nfrozen_lag_timesteps: 96\nepisodes_random: 8\nepisodes_frozen: 8\nn_envs: 2\n"


def _config_dir(tmp_path: Path) -> Path:
    d = tmp_path / "configs"
    d.mkdir()
    shutil.copy(ROOT / "configs" / "run_4a_std.yaml", d / "run_4a_std.yaml")  # la vraie config est chargée et validée
    (d / "run_4a_eval.yaml").write_text(EVAL_YAML, encoding="utf-8")
    return d


def _session(tmp_path: Path, root: str, cfg_dir: Path, **kw: Any) -> tuple[int, dict[str, Any]]:
    def tiny(cfg: Any) -> Any:  # config jouet, nom et seed du lanceur conservés
        return dataclasses.replace(cfg_of(pool=POOL), name=cfg.name, seed=cfg.seed)

    code, report = r4.launch(
        "std",
        1,
        runs_root=tmp_path / root / "runs",
        checkpoints_root=tmp_path / root / "ck",
        config_dir=cfg_dir,
        config_edit=tiny,
        run_kwargs={"env_factory": make_factory(), "optimizer_factory": adam_factory, "handle_sigint": False},
        hook_kwargs={"sim_factory": lambda n: StatefulFakeSim(n, 2), "build_obs": fake_obs, "obs_dim": 7},
        **kw,
    )
    return int(code), dict(report)


def _policy(tmp_path: Path, root: str) -> dict[str, Any]:
    last = CheckpointManager(tmp_path / root / "ck" / "run_4a_std_s1").load_latest()
    assert last is not None
    return dict(last[1]["trainer"]["learner"]["policy"])


def test_deux_sessions_reprennent_le_meme_run(tmp_path: Path) -> None:
    # Arrange
    cfg_dir = _config_dir(tmp_path)
    # Act : une session unique ; puis le même run en deux sessions (arrêt à l'itération 3, même commande relancée)
    code_a, rep_a = _session(tmp_path, "a", cfg_dir, max_hours=None)
    code_1, rep_1 = _session(tmp_path, "b", cfg_dir, max_hours=None, max_iterations=3)
    code_2, rep_2 = _session(tmp_path, "b", cfg_dir, max_hours=None)
    # Assert
    assert (code_a, code_1, code_2) == (0, 0, 0)
    assert rep_a["session_mode"] == "new" and rep_a["status"] == "completed" and rep_a["remaining_iterations"] == 0
    assert rep_1["session_mode"] == "new" and rep_1["status"] == "paused" and rep_1["end_iteration"] == 3
    assert rep_1["remaining_iterations"] == rep_a["total_iterations"] - 3
    assert rep_2["session_mode"] == "resume" and rep_2["start_iteration"] == 3 and rep_2["status"] == "completed"
    assert rep_a["run"] == rep_2["run"] == "run_4a_std_s1"
    pa, pb = _policy(tmp_path, "a"), _policy(tmp_path, "b")
    assert all(torch.equal(pa[k], pb[k]) for k in pa)
    assert rep_2["evaluation"]["evaluations_retained"] == rep_a["evaluation"]["evaluations_retained"] > 0


def test_echeance_murale_arrete_proprement_puis_reprise(tmp_path: Path) -> None:
    cfg_dir = _config_dir(tmp_path)
    code, rep = _session(tmp_path, "c", cfg_dir, max_hours=1e-12)  # échéance déjà passée à la 1re frontière
    assert code == 0 and rep["status"] == "interrupted" and rep["end_iteration"] == 0
    assert rep["remaining_iterations"] == rep["total_iterations"]
    code, rep = _session(tmp_path, "c", cfg_dir, max_hours=None)
    assert code == 0 and rep["session_mode"] == "resume" and rep["status"] == "completed"
