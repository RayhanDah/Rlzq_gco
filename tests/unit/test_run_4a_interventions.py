"""Interventions préparées après GCO-S (ADR-006 D5, ADR-007, ADR-008, ADR-009) : configs à une seule modification par
rapport à `configs/run_4a_std.yaml` et lancement par `scripts/run_4a.py --variant int_*`."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_env import fake_obs
from test_run import StatefulFakeSim, adam_factory, cfg_of, make_factory
from test_run_4a import ROOT, r4

from rewards import RewardFunction
from training import CheckpointManager, RunConfig
from training.evalhook import EVALS_FILE, EvalHookConfig
from training.run import _spec_label

CONFIGS = ROOT / "configs"
EXPECTED_DIFF: dict[str, dict[tuple[str, ...], tuple[Any, Any]]] = {
    "int_gamma998": {("ppo", "gamma"): (0.995, 0.998)},
    "int_slow_anneal": {("initial_states", "anneal_timesteps"): (500_000_000, 1_000_000_000)},
    # ADR-009 amendé : un seul bloc ajouté, `sil` (absent de std), valeurs de l'ADR (D1–D5, KS1, KS2) ; mode joint :
    # `updates_per_iteration` et `max_grad_norm` absents (None des deux côtés, donc hors du diff)
    "int_sil": {
        ("sil", k): (None, v)
        for k, v in {
            "enabled": True,
            "coef": 0.1,
            "anneal_start_timesteps": 500_000_000,
            "anneal_end_timesteps": 900_000_000,
            "value_coef": 0.01,
            "mode": "joint",
            "batch_size": 4096,
            "capacity": 250_000,
            "horizon_steps": 150,  # règle RAM de D7 (banc du 2026-11-21)
            "priority_alpha": 0.6,
            "priority_eps": 1e-3,
            "is_beta": 0.1,
            "keep_conceded": False,
            "store_float16": True,
            "share_alert": 0.5,
            "share_alert_patience": 50,
            "share_stop": 0.8,
            "share_stop_patience": 500,
            "ram_min_available_mb": 1000.0,
            "ram_low_max_s": 600.0,
        }.items()
    },
    # ADR-010 D8-2 : un seul bloc ajouté, `aux` ; λ constant (calendrier absent : None des deux côtés, hors du diff)
    "int_aux": {
        ("aux", k): (None, v)
        for k, v in {
            "enabled": True,
            "coef": 0.1,  # provisoire : fixé par les bancs D5 avant le GO
            "horizon_steps": 100,  # D2, repli H = 100 (banc D5 du 2026-11-22 : minimum 807 Mo < 1,5 Go)
            "batch_size": 8192,  # D3
            "capacity_per_class": 125_000,  # D3
            "store_float16": False,  # D3 amendé le 2026-11-23 : float32 à H = 100 (parité D8-5 en échec, 0,0137 > 0,01)
            "share_alert": 0.5,  # D6 (C4)
            "share_alert_patience": 50,
            "ram_min_available_mb": 1000.0,  # KS1
            "ram_low_max_s": 600.0,
        }.items()
    },
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
    assert r4.run_name("int_sil", 0) == "int_sil_s0" and r4.config_file("int_sil") == "int_sil.yaml"
    assert r4.default_eval_config("int_sil") == "run_s1v2_eval.yaml"
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
    for variant in ("int_gamma998", "int_slow_anneal", "int_sil"):
        with pytest.raises(_Stop):
            r4.launch(variant, 0, config_dir=CONFIGS, **kw)
    with pytest.raises(_Stop):
        r4.launch("int_gamma998", 1, config_dir=CONFIGS, eval_config="run_4a_eval.yaml", **kw)
    with pytest.raises(_Stop):
        r4.launch("std", 0, config_dir=CONFIGS, **kw)
    # Assert : nom distinct, config de l'intervention, éval v2 par défaut, choix explicite respecté, std inchangé
    (g, hg), (a, ha), (sil, hsil), (g1, hg1), (s, hs) = seen
    assert (g.name, g.seed, g.ppo.gamma) == ("int_gamma998_s0", 0, 0.998)
    assert g.initial_states.anneal_timesteps == 500_000_000
    assert (a.name, a.ppo.gamma, a.initial_states.anneal_timesteps) == ("int_slow_anneal_s0", 0.995, 1_000_000_000)
    assert (hg.protocol, ha.protocol, hsil.protocol) == ("s1_v2", "s1_v2", "s1_v2")
    assert sil.name == "int_sil_s0" and sil.sil is not None and sil.sil.enabled and sil.ppo.gamma == 0.995
    assert g.sil is None and a.sil is None and s.sil is None  # les autres variantes sans SIL
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


# ------------------------------------------------------------------------------ témoin dense HORS SPEC (ADR-011)


def test_diag_dense_ne_differe_de_std_que_par_la_recompense() -> None:
    std_raw = yaml.safe_load((CONFIGS / "run_4a_std.yaml").read_text(encoding="utf-8"))
    dd_raw = yaml.safe_load((CONFIGS / r4.DIAGNOSTICS["diag_dense"]).read_text(encoding="utf-8"))
    std_cfg = RunConfig.from_yaml(CONFIGS / "run_4a_std.yaml").to_dict()
    dd_cfg = RunConfig.from_yaml(CONFIGS / "diag_dense.yaml").to_dict()
    expected = {("reward", "variant"): ("GCO-std", "dense-diag"), ("name",): ("run_4a_std", "diag_dense")}
    assert _diff(std_raw, dd_raw) == expected
    assert _diff(std_cfg, dd_cfg) == expected


def test_diag_dense_nom_config_eval_et_parseur(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[argparse.Namespace] = []

    def fake_supervise(args: argparse.Namespace, max_hours: float | None) -> int:
        captured.append(args)
        return 0

    monkeypatch.setattr(r4, "_supervise_main", fake_supervise)
    assert r4.run_name("diag_dense", 0) == "diag_dense_s0" and r4.config_file("diag_dense") == "diag_dense.yaml"
    assert r4.default_eval_config("diag_dense") == "run_s1v2_eval.yaml"
    assert "diag_dense" not in r4.INTERVENTIONS  # pas une intervention GCO-std
    assert r4.main(["--variant", "diag_dense", "--seed", "0", "--allow-dirty"]) == 0
    assert (captured[0].variant, captured[0].eval_config) == ("diag_dense", "run_s1v2_eval.yaml")


def test_launch_diag_dense_et_refus_si_la_config_n_est_pas_dense(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[Any, EvalHookConfig]] = []
    _capture_launch(monkeypatch, seen)
    kw: dict[str, Any] = {"max_hours": None, "runs_root": tmp_path / "r", "checkpoints_root": tmp_path / "c"}
    with pytest.raises(_Stop):
        r4.launch("diag_dense", 0, config_dir=CONFIGS, **kw)
    ((cfg, hook),) = seen
    assert (cfg.name, cfg.seed, cfg.reward.variant, hook.protocol) == ("diag_dense_s0", 0, "dense-diag", "s1_v2")
    cdir = tmp_path / "configs"
    cdir.mkdir()
    shutil.copy(CONFIGS / "run_4a_std.yaml", cdir / "diag_dense.yaml")
    shutil.copy(CONFIGS / "run_s1v2_eval.yaml", cdir / "run_s1v2_eval.yaml")
    with pytest.raises(ValueError, match="dense-diag attendu"):
        r4.launch("diag_dense", 0, config_dir=cdir, **kw)


# ------------------------------------------------------------------ audit des réglages PPO, LR ×3 (ADR-012 D1, D2, P2)

LR3_DIFF: dict[tuple[str, ...], tuple[Any, Any]] = {
    ("name",): ("diag_dense", "diag_dense_lr3"),
    ("ppo", "lr_start"): (2.0e-4, 6.0e-4),
    ("ppo", "lr_end"): (5.0e-5, 1.5e-4),
}
DENSE_DIAG_REWARD_HASH = "275f3f883e786a28"  # préfixe consigné dans ADR-011 (Implémentation) et ADR-012 D2


def _ppo_text(path: Path, key: str) -> str:
    """Texte de la valeur `ppo.<key>` dans le YAML (une seule occurrence, indentée, commentaire retiré)."""
    (line,) = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith(f"  {key}:")]
    return line.split(":", 1)[1].split("#", 1)[0].strip()


def test_diag_dense_lr3_ne_differe_du_temoin_que_par_la_lr() -> None:
    # Arrange : YAML brut et configs construites (valeurs par défaut et dérivées comprises)
    base_path, lr3_path = CONFIGS / "diag_dense.yaml", CONFIGS / r4.DIAGNOSTICS["diag_dense_lr3"]
    base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    lr3_raw = yaml.safe_load(lr3_path.read_text(encoding="utf-8"))
    base, lr3 = RunConfig.from_yaml(base_path), RunConfig.from_yaml(lr3_path)
    # Act
    raw_diff, cfg_diff = _diff(base_raw, lr3_raw), _diff(base.to_dict(), lr3.to_dict())
    h_base = RewardFunction(base.reward.build()).config_hash  # même calcul que `training.run` (run.py:422)
    h_lr3 = RewardFunction(lr3.reward.build()).config_hash
    # Assert : hors `name`, exactement les deux clés de D1
    assert raw_diff == LR3_DIFF
    assert cfg_diff == LR3_DIFF
    # facteur 3 exact en décimal sur le texte du YAML (en flottant, 3 * 2e-4 != 6e-4 d'un ulp)
    for key in ("lr_start", "lr_end"):
        assert Decimal(_ppo_text(lr3_path, key)) == 3 * Decimal(_ppo_text(base_path, key))
    # même calendrier (longueur 1B, forme linéaire) : LR(t) = 3 × LR témoin(t) ; tolérance 1e-12 = arrondi flottant
    assert base.total_iterations == lr3.total_iterations == 19_376 and lr3.batch_learned_steps == 51_611
    for t in (0, 1, 1_938, 9_687, 9_688, 19_375, 19_376):
        assert lr3.ppo.lr_at(t) == pytest.approx(3 * base.ppo.lr_at(t), rel=1e-12, abs=0)
        assert lr3.ppo.ent_coef_at(t) == base.ppo.ent_coef_at(t)
    # récompense inchangée (hash du témoin) et étiquette HORS SPEC (ADR-011 D2)
    assert h_lr3 == h_base and h_lr3.startswith(DENSE_DIAG_REWARD_HASH)
    assert _spec_label(lr3) == {"hors_spec": True, "label": "HORS SPEC", "reward_variant": "dense-diag"}


def test_diag_dense_lr3_nom_config_eval_parseur_et_ligne_enfant(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    captured: list[argparse.Namespace] = []

    def fake_supervise(args: argparse.Namespace, max_hours: float | None) -> int:
        captured.append(args)
        return 0

    monkeypatch.setattr(r4, "_supervise_main", fake_supervise)
    argv = ["--variant", "diag_dense_lr3", "--seed", "0", "--max-hours", "0", "--max-iterations", "9688"]
    # Act : commande de D2 (arrêt à 0,5B)
    code = r4.main([*argv, "--allow-dirty"])
    # Assert
    assert code == 0
    assert r4.run_name("diag_dense_lr3", 0) == "diag_dense_lr3_s0"
    assert r4.config_file("diag_dense_lr3") == "diag_dense_lr3.yaml"
    assert r4.default_eval_config("diag_dense_lr3") == r4.default_eval_config("diag_dense") == "run_s1v2_eval.yaml"
    assert "diag_dense_lr3" not in r4.INTERVENTIONS  # pas une intervention GCO-std
    (args,) = captured
    assert (args.variant, args.eval_config, args.max_iterations) == ("diag_dense_lr3", "run_s1v2_eval.yaml", 9688)
    args.runs_root, args.checkpoints_root = Path("R"), Path("C")
    cmd = r4.child_command(args, 0, False, None)  # chaque session enfant (reprise après perte du GPU) garde le plafond
    assert cmd[2:7] == ["--child", "--variant", "diag_dense_lr3", "--seed", "0"]
    assert cmd[cmd.index("--max-iterations") + 1] == "9688"
    assert cmd[cmd.index("--eval-config") + 1] == "run_s1v2_eval.yaml"


def test_diag_dense_lr3_eval_due_a_l_iteration_9688(tmp_path: Path) -> None:
    # Arrange : hook réel de la config réelle (D2 : instantané de l'itération 9 688) ; simulateur jamais construit
    import train as train_script

    cfg = RunConfig.from_yaml(CONFIGS / "diag_dense_lr3.yaml")
    hook_cfg = EvalHookConfig.from_yaml(CONFIGS / r4.default_eval_config("diag_dense_lr3"))
    hook = train_script.build_eval_hook(
        cfg,
        hook_cfg,
        tmp_path / "runs",
        tmp_path / "ck",
        resume=False,
        sim_factory=lambda n: StatefulFakeSim(n, 2),
        build_obs=fake_obs,
        obs_dim=7,
    )
    # Act
    due = [it for it in range(1, 10_001) if hook.is_due(it)]
    # Assert : 0,1B à 0,5B appris = itérations du tableau M1 d'ADR-012 ; 9 688 est la 5e éval, 9 687 ne l'est pas
    assert hook.learned_steps_per_iteration == 51_611 and hook_cfg.every_timesteps == 100_000_000
    assert due == [1_938, 3_876, 5_813, 7_751, 9_688]
    assert hook.eval_index(9_688) == 5 and hook.eval_index(9_687) == 4


def _weights_sha256(path: Path) -> str:
    """SHA-256 des poids de politique d'un instantané d'évaluation (noms triés, octets des tenseurs), comme ADR-006."""
    weights = CheckpointManager.load(path)["trainer"]["learner"]["policy"]
    h = hashlib.sha256()
    for name in sorted(weights):
        h.update(name.encode("utf-8"))
        h.update(weights[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_diag_dense_lr3_theta0_identique_au_temoin_et_instantane_au_plafond(tmp_path: Path) -> None:
    # Arrange : vraie boucle (`launch` -> `training.run` -> hook) en CPU, simulateur factice. De chaque config
    # versionnée on garde le modèle, la graine, la variante de récompense et les deux LR ; le reste est réduit.
    cdir = tmp_path / "configs"
    cdir.mkdir()
    for name in ("diag_dense.yaml", "diag_dense_lr3.yaml"):
        shutil.copy(CONFIGS / name, cdir / name)
    # éval due toutes les 3 itérations (96 / 32) : le plafond 3 tombe sur une itération d'éval, comme 9 688 en D2
    (cdir / "run_s1v2_eval.yaml").write_text(
        "protocol: s1_v2\nevery_timesteps: 96\nfrozen_lag_timesteps: 96\nepisodes_random: 4\nepisodes_frozen: 4\n"
        "n_envs: 2\n",
        encoding="utf-8",
    )

    def shrink(cfg: Any) -> Any:
        # checkpoint 0 et final seulement : moins d'écritures (la reprise n'est pas l'objet de ce test)
        tiny = cfg_of(reward={"variant": cfg.reward.variant}, checkpoint={"every_iterations": 100})
        ppo = dataclasses.replace(tiny.ppo, lr_start=cfg.ppo.lr_start, lr_end=cfg.ppo.lr_end)
        return dataclasses.replace(tiny, name=cfg.name, seed=cfg.seed, model=cfg.model, ppo=ppo)

    kw: dict[str, Any] = {
        "max_hours": None,
        "runs_root": tmp_path / "r",
        "checkpoints_root": tmp_path / "c",
        "config_dir": cdir,
        "config_edit": shrink,
        "max_iterations": 3,
        "run_kwargs": {"env_factory": make_factory(), "optimizer_factory": adam_factory, "handle_sigint": False},
        "hook_kwargs": {"sim_factory": lambda n: StatefulFakeSim(n, 2), "build_obs": fake_obs, "obs_dim": 7},
    }
    # Act
    reports = {v: r4.launch(v, 0, **kw) for v in ("diag_dense", "diag_dense_lr3")}
    # Assert
    snaps: dict[str, Path] = {}
    for variant, (code, rep) in reports.items():
        name = f"{variant}_s0"
        assert (code, rep["run"], rep["status"], rep["end_iteration"]) == (0, name, "paused", 3)
        meta = json.loads((tmp_path / "r" / name / "run_meta.json").read_text(encoding="utf-8"))
        assert (meta["hors_spec"], meta["label"], meta["reward_variant"]) == (True, "HORS SPEC", "dense-diag")
        snaps[variant] = tmp_path / "c" / name / "eval_snapshots"
        lines = (tmp_path / "r" / name / EVALS_FILE).read_text(encoding="utf-8").splitlines()
        evals = [json.loads(ln) for ln in lines if ln.strip()]
        # plafond absolu = itération d'éval : l'éval et son instantané sont écrits avant l'arrêt `paused`
        at_cap = [e for e in evals if e.get("event") == "eval" and e.get("iteration") == 3]
        assert len(at_cap) == 1 and at_cap[0]["snapshot"]
        assert (snaps[variant] / at_cap[0]["snapshot"]).is_file()
    theta0 = {v: _weights_sha256(snaps[v] / "ckpt_000000_it00000000.pt") for v in snaps}
    at_3 = {v: _weights_sha256(next(snaps[v].glob("ckpt_*_it00000003.pt"))) for v in snaps}
    assert theta0["diag_dense"] == theta0["diag_dense_lr3"]  # θ₀ ne dépend pas de la LR (ADR-012 D7 P2)
    assert at_3["diag_dense"] != at_3["diag_dense_lr3"]  # sensibilité du test : la LR agit bien après θ₀


# ----------------------------------------------------------------------- SIL à fondu tardif (ADR-013, proposé)

SIL_LATE_DIFF: dict[tuple[str, ...], tuple[Any, Any]] = {
    ("name",): ("int_sil", "int_sil_late"),
    ("sil", "anneal_start_timesteps"): (500_000_000, 800_000_000),
}


def test_int_sil_late_ne_differe_de_int_sil_que_par_le_debut_du_fondu() -> None:
    # Arrange : YAML brut et configs construites (valeurs par défaut et dérivées comprises)
    base_path, late_path = CONFIGS / "int_sil.yaml", CONFIGS / r4.INTERVENTIONS["int_sil_late"]
    base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    late_raw = yaml.safe_load(late_path.read_text(encoding="utf-8"))
    base, late = RunConfig.from_yaml(base_path), RunConfig.from_yaml(late_path)
    std = RunConfig.from_yaml(CONFIGS / "run_4a_std.yaml")
    # Act
    raw_diff, cfg_diff = _diff(base_raw, late_raw), _diff(base.to_dict(), late.to_dict())
    # même calcul que `training.run` (run.py:426)
    h_std, h_base, h_late = (RewardFunction(c.reward.build()).config_hash for c in (std, base, late))
    # Assert : hors `name`, la seule clé de l'ADR-013 ; fin du fondu inchangée
    assert raw_diff == SIL_LATE_DIFF
    assert cfg_diff == SIL_LATE_DIFF
    assert late.sil is not None and late.sil.anneal_end_timesteps == 900_000_000
    # récompense GCO-std inchangée (même hash que int_sil et std), dans la spec (aucune étiquette HORS SPEC)
    assert late_raw["reward"]["variant"] == "GCO-std" and h_late == h_base == h_std
    assert _spec_label(late) == {}
    a, b = base.trajectory_dict(), late.trajectory_dict()
    assert a.pop("name") != b.pop("name") and a != b  # le fondu fait partie de la trajectoire (hash distinct)


def test_int_sil_late_calendrier_de_c_sil() -> None:
    # Arrange : vrai calendrier (`SILConfig.coef_at`, src/training/sil.py) des configs versionnées
    late = RunConfig.from_yaml(CONFIGS / "int_sil_late.yaml").sil
    base = RunConfig.from_yaml(CONFIGS / "int_sil.yaml").sil
    assert late is not None and base is not None and late.enabled
    # Act / Assert : 0,1 jusqu'à 0,8B, 0,05 à 0,85B, 0 à 0,9B et après
    assert [late.coef_at(t) for t in (0, 500_000_000, 799_999_999, 800_000_000)] == [0.1] * 4
    # tolérance 1e-12 : arrondi flottant de 0,1 × 50e6 / 100e6
    assert late.coef_at(850_000_000) == pytest.approx(0.05, rel=1e-12, abs=0)
    assert late.coef_at(899_999_999) == pytest.approx(0.1 / 100_000_000, rel=1e-9, abs=0)
    assert late.coef_at(900_000_000) == 0.0 and late.coef_at(10**9) == 0.0
    # par rapport à int_sil : jamais en dessous, strictement au-dessus pendant le fondu d'int_sil
    for t in range(0, 1_000_000_001, 10_000_000):
        assert late.coef_at(t) >= base.coef_at(t)
    assert late.coef_at(700_000_000) == 0.1 > base.coef_at(700_000_000)


def test_int_sil_late_nom_config_eval_parseur_et_ligne_enfant(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    captured: list[argparse.Namespace] = []

    def fake_supervise(args: argparse.Namespace, max_hours: float | None) -> int:
        captured.append(args)
        return 0

    monkeypatch.setattr(r4, "_supervise_main", fake_supervise)
    # Act : lancement d'ADR-013 (sans plafond de durée)
    code = r4.main(["--variant", "int_sil_late", "--seed", "0", "--max-hours", "0", "--allow-dirty"])
    # Assert
    assert code == 0
    assert r4.run_name("int_sil_late", 0) == "int_sil_late_s0"
    assert r4.config_file("int_sil_late") == "int_sil_late.yaml"
    assert r4.default_eval_config("int_sil_late") == r4.default_eval_config("int_sil") == "run_s1v2_eval.yaml"
    assert "int_sil_late" in r4.INTERVENTIONS and "int_sil_late" not in r4.DIAGNOSTICS  # GCO-std, dans la spec
    (args,) = captured
    assert (args.variant, args.eval_config) == ("int_sil_late", "run_s1v2_eval.yaml")
    args.runs_root, args.checkpoints_root = Path("R"), Path("C")
    cmd = r4.child_command(args, 0, False, None)
    assert cmd[2:7] == ["--child", "--variant", "int_sil_late", "--seed", "0"]
    assert cmd[cmd.index("--max-hours") + 1] == "0.0"
    assert cmd[cmd.index("--eval-config") + 1] == "run_s1v2_eval.yaml"
    assert "--std-s1-observed" not in cmd and "--max-iterations" not in cmd


def test_launch_int_sil_late(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    seen: list[tuple[Any, EvalHookConfig]] = []
    _capture_launch(monkeypatch, seen)
    kw: dict[str, Any] = {"max_hours": None, "runs_root": tmp_path / "r", "checkpoints_root": tmp_path / "c"}
    # Act
    with pytest.raises(_Stop):
        r4.launch("int_sil_late", 0, config_dir=CONFIGS, **kw)
    # Assert : nom du run, graine, SIL active au fondu tardif, éval S1-seed v2
    ((cfg, hook),) = seen
    assert (cfg.name, cfg.seed, cfg.reward.variant, hook.protocol) == ("int_sil_late_s0", 0, "GCO-std", "s1_v2")
    assert cfg.sil is not None and cfg.sil.enabled and cfg.sil.anneal_start_timesteps == 800_000_000


def test_int_sil_late_theta0_identique_a_std_s0(tmp_path: Path) -> None:
    # Arrange : vraie boucle (`launch` -> `training.run`) en CPU, simulateur factice. De chaque config versionnée on
    # garde le modèle, la graine et le bloc `sil` (fondu aux mêmes fractions du budget réduit, tailles réduites).
    cdir = tmp_path / "configs"
    cdir.mkdir()
    for name in ("run_4a_std.yaml", "int_sil.yaml", "int_sil_late.yaml"):
        shutil.copy(CONFIGS / name, cdir / name)
    (cdir / "run_s1v2_eval.yaml").write_text(
        "protocol: s1_v2\nevery_timesteps: 96\nfrozen_lag_timesteps: 96\nepisodes_random: 4\nepisodes_frozen: 4\n"
        "n_envs: 2\n",
        encoding="utf-8",
    )

    def shrink(cfg: Any) -> Any:
        tiny = cfg_of(reward={"variant": cfg.reward.variant}, checkpoint={"every_iterations": 100})
        sil = None
        if cfg.sil is not None:
            scale = tiny.budget.total_timesteps / cfg.budget.total_timesteps
            sil = dataclasses.replace(
                cfg.sil,
                anneal_start_timesteps=round(cfg.sil.anneal_start_timesteps * scale),
                anneal_end_timesteps=round(cfg.sil.anneal_end_timesteps * scale),
                batch_size=4,
                capacity=64,
                horizon_steps=8,
            )
        return dataclasses.replace(tiny, name=cfg.name, seed=cfg.seed, model=cfg.model, sil=sil)

    kw: dict[str, Any] = {
        "max_hours": None,
        "runs_root": tmp_path / "r",
        "checkpoints_root": tmp_path / "c",
        "config_dir": cdir,
        "config_edit": shrink,
        "eval_config": "run_s1v2_eval.yaml",
        "max_iterations": 1,
        "run_kwargs": {
            "env_factory": make_factory(),
            "optimizer_factory": adam_factory,
            "handle_sigint": False,
            "memory_probe": lambda: (4096.0, 512.0),
        },
        "hook_kwargs": {"sim_factory": lambda n: StatefulFakeSim(n, 2), "build_obs": fake_obs, "obs_dim": 7},
    }
    # Act
    reports = {v: r4.launch(v, 0, **kw) for v in ("std", "int_sil", "int_sil_late")}
    # Assert : θ₀ (checkpoint 0) identique ; la SIL ne touche pas à l'initialisation (π créée avant la SIL, run.py:438)
    theta0: dict[str, str] = {}
    for variant, (code, rep) in reports.items():
        assert (code, rep["status"], rep["end_iteration"]) == (0, "paused", 1)
        meta = json.loads((tmp_path / "r" / rep["run"] / "run_meta.json").read_text(encoding="utf-8"))
        assert "hors_spec" not in meta  # dans la spec
        theta0[variant] = _weights_sha256(tmp_path / "c" / rep["run"] / "ckpt_000000_it00000000.pt")
    assert reports["int_sil_late"][1]["run"] == "int_sil_late_s0"
    assert theta0["int_sil_late"] == theta0["int_sil"] == theta0["std"]
    # même modèle et même graine que std s0 : aux dimensions réelles, θ₀ = SHA-256 7fab4c74…cb8a (test_auxtask.py:686)
    std, late = (RunConfig.from_yaml(CONFIGS / n) for n in ("run_4a_std.yaml", "int_sil_late.yaml"))
    assert late.model == std.model and late.seed == std.seed == 0
