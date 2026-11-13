"""Verdict et résumé du smoke test (`scripts/train.py --smoke`), sans lancer de run."""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from training import RunConfig, RunSummary

ROOT = Path(__file__).resolve().parents[2]


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_script"] = mod
    spec.loader.exec_module(mod)
    return mod


def _segment(cfg: RunConfig, start: int, end: int, **over: Any) -> RunSummary:
    total = cfg.total_iterations
    base = RunSummary(
        status="completed" if end == total else "paused",
        cause=None,
        start_iteration=start,
        end_iteration=end,
        total_iterations=total,
        trained_timesteps=end * cfg.batch_agent_steps,
        collected_timesteps=(end + 1) * cfg.batch_agent_steps,
        segment_collected_timesteps=(end - start) * cfg.batch_agent_steps,
        wall_s=10.0,
        loop_s=8.0,
        mean_steps_per_s=1000.0,
        steps_per_s=[1000.0] * (end - start),
        nan_detected=0,
        restores=0,
        decisions=[],
        checkpoints=[{"bytes": 1, "write_s": 0.1, "capture_s": 0.1}],
        goals_total=0.0,
        p_kick_final=0.5,
        out_dir="out",
        checkpoint_dir="ckpt",
    )
    return dataclasses.replace(base, **over)


def _report(**second_over: Any) -> dict[str, Any]:
    script = _load_script()
    cfg = RunConfig.from_yaml(ROOT / "configs" / "smoke.yaml")
    half = cfg.total_iterations // 2
    first = _segment(cfg, 0, half)
    second = _segment(cfg, half, cfg.total_iterations, resumed_from="ckpt/x.pt", exact_env=True, **second_over)
    report: dict[str, Any] = script.smoke_report(cfg, first, second, ["python", "scripts/train.py"])
    return report


def test_smoke_propre_passe() -> None:
    report = _report()

    assert report["verdict"] == "PASS"


def test_smoke_avec_restauration_echoue() -> None:
    # Revue T6 (2026-11-13) : un RESTORE sur KL, sans NaN, n'était pas compté contre le verdict.
    report = _report(restores=1)

    assert report["checks"]["no_restore"] is False
    assert report["verdict"] == "FAIL"


def test_resume_contient_la_provenance() -> None:
    # Règle 1 du projet : une mesure porte commande, seed, commit, matériel et date ; un arbre modifié doit se voir.
    report = _report()

    assert report["seed"] == 0
    assert isinstance(report["git_dirty"], bool)
    assert report["date_utc"].endswith("+00:00")
    assert report["command"] == ["python", "scripts/train.py"]
