"""Interventions préparées après GCO-S (ADR-006 D5, ADR-007, ADR-008) : configs à une seule modification par rapport à
`configs/run_4a_std.yaml` et lancement par `scripts/run_4a.py --variant int_*`."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_run_4a import ROOT, r4

from training import RunConfig
from training.evalhook import EvalHookConfig

CONFIGS = ROOT / "configs"
EXPECTED_DIFF: dict[str, dict[tuple[str, ...], tuple[Any, Any]]] = {
    "int_gamma998": {("ppo", "gamma"): (0.995, 0.998)},
    "int_slow_anneal": {("initial_states", "anneal_timesteps"): (500_000_000, 1_000_000_000)},
}


def _flatten(d: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if not isinstance(d, dict):
        return {prefix: d}
    out: dict[tuple[str, ...], Any] = {}
    for k, v in d.items():
        out.update(_flatten(v, (*prefix, str(k))))
    return out


def _diff(a: dict[str, Any], b: dict[str, Any]) -> dict[tuple[str, ...], tuple[Any, Any]]:
    fa, fb = _flatten(a), _flatten(b)
    return {k: (fa.get(k), fb.get(k)) for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k)}


@pytest.mark.parametrize("variant", sorted(EXPECTED_DIFF))
def test_config_d_intervention_ne_differe_de_std_que_d_un_champ(variant: str) -> None:
    # Arrange : YAML brut (le parseur ignore les commentaires) et config construite (valeurs par défaut comprises)
    std_raw = yaml.safe_load((CONFIGS / "run_4a_std.yaml").read_text(encoding="utf-8"))
    int_raw = yaml.safe_load((CONFIGS / r4.INTERVENTIONS[variant]).read_text(encoding="utf-8"))
    std_cfg = RunConfig.from_yaml(CONFIGS / "run_4a_std.yaml").to_dict()
    int_cfg = RunConfig.from_yaml(CONFIGS / r4.INTERVENTIONS[variant]).to_dict()
    # Act
    raw_diff, cfg_diff = _diff(std_raw, int_raw), _diff(std_cfg, int_cfg)
    # Assert : hors `name`, exactement le champ de l'ADR, avec les valeurs de l'ADR
    expected = {**EXPECTED_DIFF[variant], ("name",): ("run_4a_std", variant)}
    assert raw_diff == expected
    assert cfg_diff == expected
    assert int_raw["reward"]["variant"] == "GCO-std"


def test_gamma_0998_et_calendrier_inchange() -> None:
    std = RunConfig.from_yaml(CONFIGS / "run_4a_std.yaml")
    cfg = RunConfig.from_yaml(CONFIGS / "int_gamma998.yaml")
    assert (cfg.ppo.gamma, cfg.ppo.gae_lambda) == (0.998, 0.95)  # λ inchangé (ADR-007)
    assert cfg.total_iterations == std.total_iterations == 19_376
    assert [cfg.p_kick_at(i) for i in (0, 9_687, 19_375)] == [std.p_kick_at(i) for i in (0, 9_687, 19_375)]


def test_annealing_lent_0_2_a_0_8_sur_tout_le_run() -> None:
    # ADR-008 : linéaire 0,2 -> 0,8 sur 1B appris = toute la durée du run ; cible atteinte à la dernière itération
    std = RunConfig.from_yaml(CONFIGS / "run_4a_std.yaml")
    cfg = RunConfig.from_yaml(CONFIGS / "int_slow_anneal.yaml")
    n = cfg.total_iterations
    assert cfg.anneal_iterations == n == 19_376 and std.anneal_iterations == 9_688
    assert cfg.p_kick_at(0) == pytest.approx(0.2) and cfg.p_kick_at(n - 1) == pytest.approx(0.8)
    assert cfg.p_kick_at(n - 2) < 0.8
    # à 0,5B (itération 9 687) : 0,5 au lieu de 0,8 ; pente moitié de celle de std
    assert cfg.p_kick_at(9_687) == pytest.approx(0.2 + 0.6 * 9_687 / (n - 1))
    assert std.p_kick_at(9_687) == pytest.approx(0.8)
    # jamais au-dessus de std, et p_kick moyen sur le run 0,5 contre 0,65
    ks = range(0, n, 97)
    assert all(cfg.p_kick_at(i) <= std.p_kick_at(i) + 1e-12 for i in ks)
    mean_slow = sum(cfg.p_kick_at(i) for i in range(n)) / n
    mean_std = sum(std.p_kick_at(i) for i in range(n)) / n
    assert mean_slow == pytest.approx(0.5, abs=1e-3) and mean_std == pytest.approx(0.65, abs=1e-3)


def test_nom_config_et_eval_par_defaut_des_interventions() -> None:
    assert r4.run_name("int_gamma998", 0) == "int_gamma998_s0"
    assert r4.run_name("int_slow_anneal", 2) == "int_slow_anneal_s2"
    assert r4.run_name("std", 0) == "run_4a_std_s0"  # inchangé
    assert r4.config_file("int_gamma998") == "int_gamma998.yaml" and r4.config_file("std") == "run_4a_std.yaml"
    assert r4.default_eval_config("int_slow_anneal") == "run_s1v2_eval.yaml"
    assert r4.default_eval_config("std") == r4.default_eval_config("s") == "run_4a_eval.yaml"
    with pytest.raises(ValueError):
        r4.run_name("int_absente", 0)
    with pytest.raises(ValueError):
        r4.run_name("int_gamma998", -1)


class _Stop(Exception):
    pass


def _capture_launch(monkeypatch: pytest.MonkeyPatch, seen: list[tuple[Any, EvalHookConfig]]) -> None:
    import train as train_script

    def fake_build(cfg: Any, hook_cfg: EvalHookConfig, *a: Any, **k: Any) -> Any:
        seen.append((cfg, hook_cfg))
        raise _Stop

    monkeypatch.setattr(train_script, "build_eval_hook", fake_build)


def test_launch_d_une_intervention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    seen: list[tuple[Any, EvalHookConfig]] = []
    _capture_launch(monkeypatch, seen)
    kw: dict[str, Any] = {"max_hours": None, "runs_root": tmp_path / "r", "checkpoints_root": tmp_path / "c"}
    # Act
    for variant in ("int_gamma998", "int_slow_anneal"):
        with pytest.raises(_Stop):
            r4.launch(variant, 0, config_dir=CONFIGS, **kw)
    with pytest.raises(_Stop):
        r4.launch("int_gamma998", 1, config_dir=CONFIGS, eval_config="run_4a_eval.yaml", **kw)
    with pytest.raises(_Stop):
        r4.launch("std", 0, config_dir=CONFIGS, **kw)
    # Assert : nom distinct, config de l'intervention, éval v2 par défaut, choix explicite respecté, std inchangé
    (g, hg), (a, ha), (g1, hg1), (s, hs) = seen
    assert (g.name, g.seed, g.ppo.gamma) == ("int_gamma998_s0", 0, 0.998)
    assert g.initial_states.anneal_timesteps == 500_000_000
    assert (a.name, a.ppo.gamma, a.initial_states.anneal_timesteps) == ("int_slow_anneal_s0", 0.995, 1_000_000_000)
    assert (hg.protocol, ha.protocol) == ("s1_v2", "s1_v2")
    assert (g1.name, hg1.protocol) == ("int_gamma998_s1", "s1_light")
    assert (s.name, s.ppo.gamma, hs.protocol) == ("run_4a_std_s0", 0.995, "s1_light")


def test_intervention_refusee_si_la_config_n_est_pas_gco_std(tmp_path: Path) -> None:
    # Arrange : une config d'intervention remplacée par une config GCO-S
    cdir = tmp_path / "configs"
    cdir.mkdir()
    shutil.copy(CONFIGS / "run_4a_s.yaml", cdir / "int_gamma998.yaml")
    shutil.copy(CONFIGS / "run_s1v2_eval.yaml", cdir / "run_s1v2_eval.yaml")
    # Act / Assert
    with pytest.raises(ValueError, match="GCO-std attendu"):
        r4.launch("int_gamma998", 0, max_hours=None, runs_root=tmp_path / "r", checkpoints_root=tmp_path / "c",
                  config_dir=cdir)  # fmt: skip
    assert not (tmp_path / "r").exists() and not (tmp_path / "c").exists()


def test_parseur_et_ligne_enfant_d_une_intervention(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    captured: list[argparse.Namespace] = []

    def fake_supervise(args: argparse.Namespace, max_hours: float | None) -> int:
        captured.append(args)
        return 0

    monkeypatch.setattr(r4, "_supervise_main", fake_supervise)
    # Act : sans --std-s1-observed (pas de garde-fou D3 : GCO-std), éval par défaut puis explicite
    assert r4.main(["--variant", "int_gamma998", "--seed", "0", "--allow-dirty"]) == 0
    assert r4.main(["--variant", "int_slow_anneal", "--seed", "0", "--allow-dirty", "--eval-config", "x.yaml"]) == 0
    assert r4.main(["--variant", "std", "--seed", "0", "--allow-dirty"]) == 0
    # Assert
    a, b, std = captured
    assert (a.variant, a.eval_config) == ("int_gamma998", "run_s1v2_eval.yaml")
    assert (b.variant, b.eval_config) == ("int_slow_anneal", "x.yaml")
    assert std.eval_config == r4.EVAL_CONFIG  # std inchangé
    a.runs_root, a.checkpoints_root = Path("R"), Path("C")
    cmd = r4.child_command(a, 0, False, None)
    assert cmd[2:7] == ["--child", "--variant", "int_gamma998", "--seed", "0"]
    assert cmd[cmd.index("--eval-config") + 1] == "run_s1v2_eval.yaml"
    assert "--std-s1-observed" not in cmd


@pytest.mark.parametrize(
    ("variant", "explicit", "expected"),
    [
        ("int_gamma998", ["--eval-config", "run_4a_eval.yaml"], "run_4a_eval.yaml"),
        ("int_slow_anneal", [], "run_s1v2_eval.yaml"),
        ("std", [], "run_4a_eval.yaml"),
        ("std", ["--eval-config", "run_s1v2_eval.yaml"], "run_s1v2_eval.yaml"),
    ],
)
def test_config_d_eval_transmise_telle_quelle_a_la_session_enfant(
    monkeypatch: pytest.MonkeyPatch, variant: str, explicit: list[str], expected: str
) -> None:
    """Régression (revue de 933a8e8, majeur) : `child_command` n'ajoutait `--eval-config` que s'il différait de
    `run_4a_eval.yaml` ; pour une intervention, dont le défaut est `run_s1v2_eval.yaml`, un `run_4a_eval.yaml` explicite
    était perdu et l'enfant retombait silencieusement sur le protocole v2."""
    # Arrange : superviseur -> ligne de commande enfant -> session enfant (launch factice)
    supervised: list[argparse.Namespace] = []
    launched: list[str | Path | None] = []

    def fake_supervise(args: argparse.Namespace, max_hours: float | None) -> int:
        supervised.append(args)
        return 0

    def fake_launch(variant: str, seed: int, **kw: Any) -> tuple[int, dict[str, Any]]:
        launched.append(kw["eval_config"])
        return 0, {"status": "paused", "remaining_iterations": 1}

    import bench_run

    monkeypatch.setattr(r4, "_supervise_main", fake_supervise)
    monkeypatch.setattr(r4, "launch", fake_launch)
    monkeypatch.setattr(bench_run, "_set_rocketgpu_defaults", lambda: {})
    base = ["--variant", variant, "--seed", "0", "--allow-dirty", "--no-keep-awake"]
    # Act
    assert r4.main(base + explicit) == 0
    child = r4.child_command(supervised[0], 0, False, None)
    assert r4.main(child[2:]) == 0  # l'enfant reparse sa ligne de commande
    # Assert : la session enfant évalue avec la config choisie par l'utilisateur (ou le défaut de la variante)
    assert supervised[0].eval_config == expected
    assert launched == [expected]


@pytest.mark.parametrize("variant", sorted(r4.INTERVENTIONS))
def test_reprise_et_refus_apres_stop_pour_une_intervention(tmp_path: Path, variant: str) -> None:
    # Arrange : run <intervention>_s0 existant, dernière session STOP, un checkpoint
    name = f"{variant}_s0"
    out, ck = tmp_path / "runs" / name, tmp_path / "ck" / name
    out.mkdir(parents=True)
    (out / "segments.jsonl").write_text(
        '{"status": "paused"}\n{"status": "stopped", "cause": "K2"}\n', encoding="utf-8"
    )
    ck.mkdir(parents=True)
    (ck / "ckpt_000000_it00000000.pt").write_bytes(b"")
    kw: dict[str, Any] = {"max_hours": None, "runs_root": tmp_path / "runs", "checkpoints_root": tmp_path / "ck"}
    before = sorted(p.name for p in out.iterdir())
    # Act / Assert : même chemin que std (session_mode puis check_resumable), sur le nom de l'intervention
    assert r4.session_mode(out, ck) == "resume"
    with pytest.raises(r4.SessionRefused, match="stopped"):
        r4.launch(variant, 0, config_dir=CONFIGS, **kw)
    assert sorted(p.name for p in out.iterdir()) == before
    # run terminé : rien n'est relancé
    (out / "segments.jsonl").write_text('{"status": "completed"}\n', encoding="utf-8")
    code, report = r4.launch(variant, 0, config_dir=CONFIGS, **kw)
    assert (code, report["run"], report["status"], report["session_mode"]) == (0, name, "completed", "none")
