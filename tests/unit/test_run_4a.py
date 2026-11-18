"""Partie pure du lanceur des runs 4a (`scripts/run_4a.py`, A6) et validité des configs versionnées."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("run_4a", ROOT / "scripts" / "run_4a.py")
assert _spec is not None and _spec.loader is not None
r4 = importlib.util.module_from_spec(_spec)
sys.modules["run_4a"] = r4
_spec.loader.exec_module(r4)

from training import RunConfig  # noqa: E402
from training.evalhook import EvalHookConfig  # noqa: E402


def test_nom_du_run_et_refus() -> None:
    assert r4.run_name("std", 0) == "run_4a_std_s0" and r4.run_name("s", 2) == "run_4a_s_s2"
    with pytest.raises(ValueError):
        r4.run_name("ext", 0)
    with pytest.raises(ValueError):
        r4.run_name("std", -1)


def test_mode_de_session(tmp_path: Path) -> None:
    out, ck = tmp_path / "runs" / "x", tmp_path / "ck" / "x"
    assert r4.session_mode(out, ck) == "new"
    out.mkdir(parents=True)
    assert r4.session_mode(out, ck) == "new"  # dossier vide : rien à reprendre, rien à écraser
    (out / "events.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="aucun checkpoint"):
        r4.session_mode(out, ck)  # journaux sans checkpoint : ni neuf ni reprenable
    ck.mkdir(parents=True)
    (ck / "notes.txt").write_text("x", encoding="utf-8")
    assert not r4.has_checkpoint(ck)
    (ck / "ckpt_000000_it00000000.pt").write_bytes(b"")
    assert r4.session_mode(out, ck) == "resume"


def test_echeance_murale() -> None:
    now = [100.0]
    stop = r4.deadline(0.5, clock=lambda: now[0])
    assert stop is not None and not stop()
    now[0] = 100.0 + 1799.0
    assert not stop()
    now[0] = 100.0 + 1800.0
    assert stop()
    assert r4.deadline(None) is None
    with pytest.raises(ValueError):
        r4.deadline(0.0)


def test_codes_de_sortie() -> None:
    assert [r4.exit_code(s) for s in ("completed", "paused", "interrupted")] == [0, 0, 0]
    assert r4.exit_code("stopped") == 3 and r4.exit_code("error") == 1


def test_gco_s_refuse_sans_attestation(capsys: pytest.CaptureFixture[str]) -> None:
    assert r4.main(["--variant", "s", "--seed", "0"]) == 1
    assert "S1-léger" in capsys.readouterr().err


def test_configs_versionnees_conformes_a_l_adr_005() -> None:
    std = RunConfig.from_yaml(ROOT / "configs" / "run_4a_std.yaml")
    s = RunConfig.from_yaml(ROOT / "configs" / "run_4a_s.yaml")
    hook = EvalHookConfig.from_yaml(ROOT / "configs" / "run_4a_eval.yaml")
    # D3 : 1B appris ; D5 : pool ; D2 : rotation keep 2
    for cfg in (std, s):
        assert cfg.budget.total_timesteps == 1_000_000_000 and cfg.batch_learned_steps == 51_611
        assert cfg.pool is not None
        assert (cfg.pool.past_fraction, cfg.pool.snapshot_every, cfg.pool.max_size) == (0.2, 50, 100)
        assert cfg.checkpoint.keep == 2
    assert (std.reward.variant, s.reward.variant) == ("GCO-std", "GCO-S")
    # D4 : p_kick 0,2 -> 0,8 sur 0,5B (std) ; 1,0 imposé (S)
    st = std.initial_states
    assert (st.p_kick_start, st.p_kick_end, st.anneal_timesteps) == (0.2, 0.8, 500_000_000)
    assert (s.initial_states.p_kick_start, s.initial_states.p_kick_end) == (1.0, 1.0)
    # D3 : config identique hors variante et états initiaux
    a, b = std.to_dict(), s.to_dict()
    for d in (a, b):
        for key in ("name", "reward", "initial_states"):
            d.pop(key)
    assert a == b
    # mêmes paramètres d'entraînement que le banc mesuré (ADR-001 M8), hors sections opérationnelles
    bench = RunConfig.from_yaml(ROOT / "configs" / "bench_4a.yaml")
    ta, tb = std.trajectory_dict(), bench.trajectory_dict()
    assert ta.pop("name") != tb.pop("name") and ta == tb
    # D2 et précisions validées ; R0 = borne haute de P0 complet
    assert (hook.every_timesteps, hook.frozen_lag_timesteps, hook.n_envs) == (50_000_000, 100_000_000, 256)
    assert (hook.episodes_random, hook.episodes_frozen, hook.min_decisive, hook.consecutive) == (512, 512, 20, 3)
    assert hook.garwood_two_sided and hook.r0_goals_per_min == 0.027


def test_config_d_eval_s1_v2_conforme_a_l_adr_006() -> None:
    h = EvalHookConfig.from_yaml(ROOT / "configs" / "run_s1v2_eval.yaml")
    assert h.protocol == "s1_v2" and (h.every_timesteps, h.n_envs, h.seed) == (100_000_000, 256, None)
    assert (h.episodes_random, h.episodes_frozen) == (3072, 3072)
    assert (h.sprt_elo1, h.sprt_alpha, h.sprt_beta) == (100.0, 0.05, 0.05)
    assert (h.consecutive, h.k7_consecutive, h.r0_goals_per_min) == (2, 2, 0.027) and h.garwood_two_sided
    # la config par défaut des runs 4a reste S1-léger, inchangée
    old = EvalHookConfig.from_yaml(ROOT / "configs" / "run_4a_eval.yaml")
    assert old.protocol == "s1_light" and old.consecutive == 3 and old.episodes_random == 512


def test_option_eval_config_choisit_la_config_d_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import train as train_script

    seen: list[EvalHookConfig] = []

    class Stop(Exception):
        pass

    def fake_build(cfg: Any, hook_cfg: EvalHookConfig, *a: Any, **k: Any) -> Any:
        seen.append(hook_cfg)
        raise Stop

    monkeypatch.setattr(train_script, "build_eval_hook", fake_build)
    kw: dict[str, Any] = {
        "max_hours": None,
        "runs_root": tmp_path / "r",
        "checkpoints_root": tmp_path / "c",
        "config_dir": ROOT / "configs",
    }
    with pytest.raises(Stop):
        r4.launch("std", 0, **kw)  # défaut inchangé
    with pytest.raises(Stop):
        r4.launch("std", 0, eval_config="run_s1v2_eval.yaml", **kw)
    assert [h.protocol for h in seen] == ["s1_light", "s1_v2"]


def test_resolution_de_la_config_d_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : un nom dans config_dir, un fichier ailleurs (relatif au répertoire courant et absolu)
    cdir = tmp_path / "configs"
    cdir.mkdir()
    (cdir / "a.yaml").write_text("protocol: s1_v2\n", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "b.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # Assert : nom -> config_dir ; `configs/a.yaml` (relatif au cwd) n'est pas résolu en configs/configs/a.yaml
    assert r4.resolve_eval_config(cdir, "a.yaml") == cdir / "a.yaml"
    assert r4.resolve_eval_config(cdir, "configs/a.yaml") == Path("configs/a.yaml")
    assert r4.resolve_eval_config(cdir, "other/b.yaml") == Path("other/b.yaml")
    assert r4.resolve_eval_config(cdir, other / "b.yaml") == other / "b.yaml"
    with pytest.raises(FileNotFoundError, match="introuvable"):
        r4.resolve_eval_config(cdir, "absent.yaml")
    with pytest.raises(FileNotFoundError):
        r4.resolve_eval_config(cdir, tmp_path / "absent.yaml")


def _segments(out: Path, statuses: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"status": st, "cause": "K2 : entropie" if st == "stopped" else None}) for st in statuses]
    (out / "segments.jsonl").write_text("".join(x + chr(10) for x in lines), encoding="utf-8")


def test_dernier_segment_lu_et_absent(tmp_path: Path) -> None:
    assert r4.last_segment(tmp_path / "absent") is None
    _segments(tmp_path / "x", ["paused", "stopped"])
    last = r4.last_segment(tmp_path / "x")
    assert last is not None and last["status"] == "stopped"


@pytest.mark.parametrize("status", ["stopped", "error"])
def test_reprise_refusee_apres_un_stop_ou_une_erreur(tmp_path: Path, status: str) -> None:
    """Régression (revue A6, M1) : après un STOP (K2, K3, K5), relancer la même commande reprenait le run depuis un
    checkpoint sain antérieur, contre l'ADR-005 D6 (« STOP = fin du run »)."""
    # Arrange : un run existant dont la dernière session s'est terminée par un STOP (ou une erreur)
    out, ck = tmp_path / "runs" / "run_4a_std_s0", tmp_path / "ck" / "run_4a_std_s0"
    _segments(out, ["paused", status])
    ck.mkdir(parents=True)
    (ck / "ckpt_000000_it00000000.pt").write_bytes(b"")
    # Act / Assert : refus avant toute construction (ni env ni hook), aucun fichier touché
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(r4.SessionRefused, match=status):
        r4.check_resumable(out, force=False)
    with pytest.raises(r4.SessionRefused):
        r4.launch(
            "std",
            0,
            max_hours=None,
            runs_root=tmp_path / "runs",
            checkpoints_root=tmp_path / "ck",
            config_dir=ROOT / "configs",
        )
    assert sorted(p.name for p in out.iterdir()) == before
    r4.check_resumable(out, force=True)  # levée explicite (--resume-after-stop) : décision humaine


def test_run_termine_n_est_pas_relance(tmp_path: Path) -> None:
    out = tmp_path / "runs" / "run_4a_std_s0"
    _segments(out, ["paused", "completed"])
    assert r4.check_resumable(out, force=False) == "completed"
    _segments(out, ["paused", "interrupted"])
    assert r4.check_resumable(out, force=False) == "resumable"
