"""Run long de démo (ADR-014) : `configs/long_run.yaml` ne diffère de GCO-S (`run_4a_s.yaml`) que par le nom et le
budget ; `scripts/run_4a.py --variant long` lance `long_s<seed>` avec l'éval `configs/long_eval.yaml`."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_run_4a import ROOT, r4
from test_run_4a_interventions import _capture_launch, _diff, _Stop

from training import RunConfig
from training.evalhook import EvalHookConfig

CONFIGS = ROOT / "configs"


def test_long_run_ne_differe_de_gco_s_que_par_le_nom_et_le_budget() -> None:
    # Arrange
    s_raw = yaml.safe_load((CONFIGS / "run_4a_s.yaml").read_text(encoding="utf-8"))
    l_raw = yaml.safe_load((CONFIGS / "long_run.yaml").read_text(encoding="utf-8"))
    s_cfg = RunConfig.from_yaml(CONFIGS / "run_4a_s.yaml").to_dict()
    l_cfg = RunConfig.from_yaml(CONFIGS / "long_run.yaml").to_dict()
    # Act
    raw_diff, cfg_diff = _diff(s_raw, l_raw), _diff(s_cfg, l_cfg)
    # Assert : GCO strict (aucun bloc sil/aux), seuls le nom et le budget changent, kill criteria conservés
    expected = {
        ("name",): ("run_4a_s", "long_run"),
        ("budget", "total_timesteps"): (1_000_000_000, 25_000_000_000),
    }
    assert raw_diff == expected
    # config construite : plus le champ dérivé du budget (⌈budget / 51 611⌉, interdit dans le YAML)
    assert cfg_diff == {**expected, ("ppo", "total_iterations"): (19_376, 484_393)}
    assert l_raw["reward"]["variant"] == "GCO-S" and "sil" not in l_raw and "aux" not in l_raw
    assert l_raw["kill"] == s_raw["kill"]


def test_long_run_budget_et_calendriers_sur_25b() -> None:
    # Act
    cfg = RunConfig.from_yaml(CONFIGS / "long_run.yaml")
    # Assert : ⌈25e9 / 51 611⌉ itérations, calendrier PPO étalé sur tout le run
    assert cfg.batch_learned_steps == 51_611
    assert cfg.total_iterations == 484_393
    assert cfg.ppo.total_iterations == cfg.total_iterations


def test_long_eval_ne_differe_de_s1v2_que_par_la_frequence() -> None:
    # Arrange
    v2 = yaml.safe_load((CONFIGS / "run_s1v2_eval.yaml").read_text(encoding="utf-8"))
    lg = yaml.safe_load((CONFIGS / "long_eval.yaml").read_text(encoding="utf-8"))
    # Act
    diff = _diff(v2, lg)
    hook = EvalHookConfig.from_yaml(CONFIGS / "long_eval.yaml")
    # Assert
    assert diff == {
        ("every_timesteps",): (100_000_000, 500_000_000),
        ("frozen_lag_timesteps",): (100_000_000, 500_000_000),
    }
    assert hook.protocol == "s1_v2"


def test_nom_config_et_eval_par_defaut_du_run_long() -> None:
    assert r4.run_name("long", 0) == "long_s0"
    assert r4.config_file("long") == "long_run.yaml"
    assert r4.default_eval_config("long") == "long_eval.yaml"
    assert r4.run_name("s", 0) == "run_4a_s_s0" and r4.default_eval_config("s") == "run_4a_eval.yaml"  # inchangés


def test_launch_du_run_long(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    seen: list[tuple[Any, EvalHookConfig]] = []
    _capture_launch(monkeypatch, seen)
    kw: dict[str, Any] = {"max_hours": None, "runs_root": tmp_path / "r", "checkpoints_root": tmp_path / "c"}
    # Act
    with pytest.raises(_Stop):
        r4.launch("long", 0, config_dir=CONFIGS, **kw)
    # Assert
    ((cfg, hook),) = seen
    assert (cfg.name, cfg.seed, cfg.reward.variant) == ("long_s0", 0, "GCO-S")
    assert cfg.sil is None and cfg.aux is None
    assert (hook.protocol, hook.every_timesteps) == ("s1_v2", 500_000_000)


def test_run_long_refuse_si_la_config_n_est_pas_gco_s(tmp_path: Path) -> None:
    # Arrange : la config du run long remplacée par une config GCO-std
    cdir = tmp_path / "configs"
    cdir.mkdir()
    shutil.copy(CONFIGS / "run_4a_std.yaml", cdir / "long_run.yaml")
    shutil.copy(CONFIGS / "long_eval.yaml", cdir / "long_eval.yaml")
    # Act / Assert
    with pytest.raises(ValueError, match="GCO-S sans SIL"):
        r4.launch("long", 0, max_hours=None, runs_root=tmp_path / "r", checkpoints_root=tmp_path / "c",
                  config_dir=cdir)  # fmt: skip
    assert not (tmp_path / "r").exists() and not (tmp_path / "c").exists()


def test_parseur_et_ligne_enfant_du_run_long(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    captured: list[argparse.Namespace] = []

    def fake_supervise(args: argparse.Namespace, max_hours: float | None) -> int:
        captured.append(args)
        return 0

    monkeypatch.setattr(r4, "_supervise_main", fake_supervise)
    # Act
    code = r4.main(["--variant", "long", "--seed", "0", "--max-hours", "0", "--allow-dirty"])
    # Assert
    assert code == 0
    (args,) = captured
    assert (args.variant, args.eval_config) == ("long", "long_eval.yaml")
    args.runs_root, args.checkpoints_root = Path("R"), Path("C")
    cmd = r4.child_command(args, 0, False, None)
    assert cmd[2:7] == ["--child", "--variant", "long", "--seed", "0"]
    assert cmd[cmd.index("--max-hours") + 1] == "0.0"
    assert cmd[cmd.index("--eval-config") + 1] == "long_eval.yaml"
    assert "--std-s1-observed" not in cmd
