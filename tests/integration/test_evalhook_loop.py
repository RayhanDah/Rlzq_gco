"""`training.evalhook` branché sur la vraie boucle (`training.run`) avec le simulateur factice de `tests/unit`
(sans RocketGPU) : S1 armé après un checkpoint sain, évaluations poursuivies après S1 (D7) et K7, évaluation de
l'itération divergente écartée après restauration, pas en timesteps appris avec le pool, reprise exacte avec pool et
hook, trajectoire et RNG globaux identiques avec ou sans évaluation, erreur d'évaluation tracée sans tuer le run.
En intégration pour garder `tests/unit` sous 60 s
(≈ 10 s pour le fichier, mesuré le 2026-11-14)."""

from __future__ import annotations

import dataclasses
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))

from test_env import fake_obs  # noqa: E402
from test_evalhook import (  # noqa: E402
    PASS_FROZEN,
    Script,
    _result,
)
from test_run import StatefulFakeSim, cfg_of, events, go, inject, rows  # noqa: E402

from eval.match import RandomPolicy  # noqa: E402
from models import PolicyNet  # noqa: E402
from training.checkpoint import CheckpointIdentity, CheckpointManager  # noqa: E402
from training.evalhook import EVALS_FILE, EvalHook, EvalHookConfig  # noqa: E402

hook_mod = importlib.import_module("training.evalhook")


def run_hook(out: Path, cfg: Any, hook_cfg: EvalHookConfig, sim_factory: Any = None) -> EvalHook:
    return EvalHook(
        hook_cfg,
        learned_steps_per_iteration=cfg.batch_learned_steps,
        run_seed=cfg.seed,
        out_dir=out,
        snapshot_dir=out / "checkpoints" / "eval_snapshots",
        policy_factory=lambda: PolicyNet(7, 90, (16, 16)),
        sim_factory=sim_factory or (lambda n: StatefulFakeSim(n, 2)),
        identity=CheckpointIdentity.current(reward_config_hash="r", obs_dim=7, tick_skip=8),
        reward_config=cfg.reward.build(),
        build_obs=fake_obs,
        obs_dim=7,
    )


def log(out: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (out / EVALS_FILE).read_text(encoding="utf-8").splitlines()]


EVERY = EvalHookConfig(every_timesteps=32, frozen_lag_timesteps=64)


def test_boucle_reelle_s1_arme_apres_checkpoint_sain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : 32 transitions par itération, une évaluation par itération, checkpoint sain toutes les 2 itérations
    monkeypatch.setattr(hook_mod, "play_match", Script("PPPPPP"))
    cfg = cfg_of()
    hook = run_hook(tmp_path / "run", cfg, EVERY)
    # Act
    s = go(cfg, tmp_path / "run", eval_hook=hook)
    # Assert : réussites à 1, 2, 3 ; 3 couverte par le checkpoint de l'itération 4 ; armé à l'appel 5
    assert s.status == "completed" and s.s1_reached and s.s1_iteration == 5 and hook.s1_eval_iteration == 3
    assert [e["iteration"] for e in events(tmp_path / "run") if e["event"] == "s1_reached"] == [5]
    assert len(list((tmp_path / "run" / "checkpoints" / "eval_snapshots").iterdir())) == 1 + 6
    last = CheckpointManager(tmp_path / "run" / "checkpoints").load_latest()
    assert last is not None and last[1]["extra"]["evalhook"]["s1_iteration"] == 5


def test_boucle_reelle_evaluations_continuent_apres_s1_et_k7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : 3 réussites puis effondrement contre l'aléatoire (0 V / 30 D) ; budget de 8 itérations
    script = Script("PPP")

    def match(a: Any, b: Any, sim: Any, n: int, **kw: Any) -> Any:
        if isinstance(b, RandomPolicy) and script.n_eval >= 3:
            script.n_eval += 1
            return _result(0, 30)
        return script(a, b, sim, n, **kw)

    monkeypatch.setattr(hook_mod, "play_match", match)
    cfg = cfg_of(budget={"total_timesteps": 256})
    hook = run_hook(tmp_path / "run", cfg, EVERY)
    # Act
    s = go(cfg, tmp_path / "run", eval_hook=hook)
    # Assert : une évaluation à CHAQUE itération, y compris après S1 (armé à 5) ; K7 à la 3e mauvaise (itération 6)
    ev = log(tmp_path / "run")
    assert s.s1_iteration == 5 and [e["iteration"] for e in ev if e["event"] == "eval"] == list(range(1, 9))
    assert [e["iteration"] for e in ev if e["event"] == "eval" and e["iteration"] > 5] == [6, 7, 8]
    assert [e["iteration"] for e in ev if e["event"] == "k7_alert"] == [6]


def test_boucle_reelle_evaluation_de_l_iteration_divergente_ecartee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : NaN au 3e update -> RESTORE vers l'itération 2 (checkpoint sain tous les 2) ; toutes les évaluations
    # réussissent. Sans retour en arrière, la réussite de l'itération divergente compterait.
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, grad_norm=float("nan")) if n == 3 else st)
    monkeypatch.setattr(hook_mod, "play_match", Script("P" * 20))
    cfg = cfg_of()
    hook = run_hook(tmp_path / "run", cfg, EVERY)
    # Act
    s = go(cfg, tmp_path / "run", eval_hook=hook)
    # Assert : l'état est rechargé du checkpoint 2 (l'évaluation de 3 disparaît), la nouvelle 3 est rejouée
    its = [r["iteration"] for r in hook.history]
    assert s.restores == 1 and its == [1, 2, 3, 4, 5, 6]
    assert sum(1 for e in log(tmp_path / "run") if e["event"] == "eval" and e["iteration"] == 3) == 2
    assert s.s1_reached and hook.s1_eval_iteration == 3 and s.s1_iteration == 5  # série 1, 2, 3 rejouée ; sain 4


def test_pas_d_evaluation_en_timesteps_appris_avec_le_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : pool 50 % : 24 appris pour 32 collectés par itération ; une évaluation tous les 48 appris (2 itérations)
    monkeypatch.setattr(hook_mod, "play_match", Script("F" * 10))
    cfg = cfg_of(pool={"past_fraction": 0.5, "snapshot_every": 2, "max_size": 2, "quality_lr": 0.01})
    assert (cfg.batch_agent_steps, cfg.batch_learned_steps) == (32, 24)
    hook = run_hook(tmp_path / "run", cfg, EvalHookConfig(every_timesteps=48, frozen_lag_timesteps=96))
    # Act
    go(cfg, tmp_path / "run", eval_hook=hook)
    # Assert : itérations 2, 4, 6, 8 (en collecté, ce serait tous les 1,5 itérations : 2, 3, 5, 6, 8)
    ev = [e for e in log(tmp_path / "run") if e["event"] == "eval"]
    assert [e["iteration"] for e in ev] == [2, 4, 6, 8] and [e["trained_timesteps"] for e in ev] == [48, 96, 144, 192]
    assert [e["frozen"]["eval_index"] for e in ev] == [0, 0, 1, 2]
    with pytest.raises(Exception, match="batch_learned_steps"):
        bad = EvalHook(**{**hook_kwargs(tmp_path / "bad", cfg), "learned_steps_per_iteration": 32})
        go(cfg, tmp_path / "bad", eval_hook=bad)


def hook_kwargs(out: Path, cfg: Any) -> dict[str, Any]:
    h = run_hook(out, cfg, EVERY)
    return {
        "cfg": h.cfg,
        "learned_steps_per_iteration": h.learned_steps_per_iteration,
        "run_seed": cfg.seed,
        "out_dir": out,
        "snapshot_dir": out / "snap",
        "policy_factory": lambda: PolicyNet(7, 90, (16, 16)),
        "sim_factory": lambda n: StatefulFakeSim(n, 2),
        "identity": h.identity,
    }


def _stable(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if k not in ("t", "perf/agent_steps_per_s")}


def test_reprise_exacte_avec_pool_et_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : vrai `play_match` (simulateur factice dédié), pool actif ; continu contre arrêt à 3 puis reprise
    cfg = cfg_of(pool={"past_fraction": 0.5, "snapshot_every": 2, "max_size": 2, "quality_lr": 0.01})
    small = EvalHookConfig(every_timesteps=48, frozen_lag_timesteps=96, episodes_random=8, episodes_frozen=8, n_envs=2)
    a, b = tmp_path / "a", tmp_path / "b"
    ha = run_hook(a, cfg, small)
    go(cfg, a, eval_hook=ha)
    go(cfg, b, eval_hook=run_hook(b, cfg, small), max_iterations=3)
    hb = run_hook(b, cfg, small)  # nouveau processus : l'état vient du checkpoint repris
    s2 = go(cfg, b, eval_hook=hb, resume=True)
    # Assert : mêmes lignes de journal, mêmes poids, mêmes évaluations retenues (résultats identiques)
    assert s2.start_iteration == 3 and s2.status == "completed"  # checkpoint final de la pause : 3
    assert [_stable(x) for x in rows(a)] == [_stable(x) for x in rows(b)]
    la, lb = CheckpointManager(a / "checkpoints").load_latest(), CheckpointManager(b / "checkpoints").load_latest()
    assert la is not None and lb is not None
    for k, v in la[1]["trainer"]["learner"]["policy"].items():
        assert torch.equal(v, lb[1]["trainer"]["learner"]["policy"][k])

    def strip(h: EvalHook) -> list[Any]:
        keep = ("iteration", "passed", "random", "frozen")
        return [
            {
                k: ({kk: vv for kk, vv in r[k].items() if kk != "snapshot"} if isinstance(r[k], dict) else r[k])
                for k in keep
            }
            for r in h.history
        ]

    assert strip(ha) == strip(hb) and len(ha.history) == 4


def test_trajectoire_et_rng_globaux_identiques_avec_ou_sans_evaluation(tmp_path: Path) -> None:
    # vrai `play_match` sur un simulateur factice dédié : aucun générateur global consommé
    cfg = cfg_of()
    a = go(cfg, tmp_path / "a")
    small = EvalHookConfig(every_timesteps=64, frozen_lag_timesteps=128, episodes_random=8, episodes_frozen=8, n_envs=2)
    hook = run_hook(tmp_path / "b", cfg, small)
    b = go(cfg, tmp_path / "b", eval_hook=hook)
    assert len(hook.history) == 3 and all(r["random"]["episodes"] == 8 for r in hook.history)
    la = CheckpointManager(tmp_path / "a" / "checkpoints").load_latest()
    lb = CheckpointManager(tmp_path / "b" / "checkpoints").load_latest()
    assert la is not None and lb is not None
    pa, pb = la[1]["trainer"]["learner"]["policy"], lb[1]["trainer"]["learner"]["policy"]
    assert all(torch.equal(pa[k], pb[k]) for k in pa)
    assert a.goals_total == b.goals_total
    ra, rb = la[1]["rng"], lb[1]["rng"]  # python, numpy, torch : état global capturé par le checkpoint
    assert json.dumps(ra, default=repr, sort_keys=True) == json.dumps(rb, default=repr, sort_keys=True)
    assert torch.equal(torch.as_tensor(ra["torch"]), torch.as_tensor(rb["torch"]))


def test_erreur_d_evaluation_tracee_le_run_continue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : la 2e évaluation lève ; puis une erreur à chaque appel au-delà du plafond (3 d'affilée tolérées)
    calls = {"n": 0}

    def flaky(a: Any, b: Any, sim: Any, n: int, **kw: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 3:  # 2e évaluation, match contre l'aléatoire
            raise RuntimeError("simulateur d'évaluation perdu")
        return _result(30, 0) if isinstance(b, RandomPolicy) else PASS_FROZEN

    monkeypatch.setattr(hook_mod, "play_match", flaky)
    cfg = cfg_of(budget={"total_timesteps": 256})  # 8 itérations : la série 3-4-5 est couverte au checkpoint 6
    hook = run_hook(tmp_path / "run", cfg, EVERY)
    # Act
    s = go(cfg, tmp_path / "run", eval_hook=hook)
    # Assert : événement tracé, évaluation comptée comme échouée (série remise à zéro), run terminé
    ev = log(tmp_path / "run")
    err = [e for e in ev if e["event"] == "eval_error"]
    assert s.status == "completed" and len(err) == 1 and err[0]["iteration"] == 2 and "perdu" in err[0]["message"]
    assert [r["passed"] for r in hook.history] == [True, False] + [True] * 6
    assert hook.s1_eval_iteration == 5
    # erreurs répétées : au-delà de max_consecutive_errors, l'exception remonte et `run` la trace
    monkeypatch.setattr(hook_mod, "play_match", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cassé")))
    one = EvalHookConfig(every_timesteps=32, frozen_lag_timesteps=64, max_consecutive_errors=1)
    hook2 = run_hook(tmp_path / "run2", cfg, one)
    with pytest.raises(Exception, match="erreurs d'évaluation"):
        go(cfg, tmp_path / "run2", eval_hook=hook2)
    assert any(e["event"] == "exception" for e in events(tmp_path / "run2"))


def test_restauration_vers_l_iteration_0_conserve_theta0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : NaN au 1er update -> RESTORE vers le checkpoint de l'itération 0 (seconde revue A4, majeur)
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, grad_norm=float("nan")) if n == 1 else st)
    monkeypatch.setattr(hook_mod, "play_match", Script("P" * 20))
    cfg = cfg_of()
    hook = run_hook(tmp_path / "run", cfg, EVERY)
    # Act
    s = go(cfg, tmp_path / "run", eval_hook=hook)
    # Assert : aucune erreur d'évaluation, θ₀ connu, le checkpoint 0 contient déjà θ₀
    assert s.status == "completed" and s.restores == 1
    assert not [e for e in log(tmp_path / "run") if e["event"] == "eval_error"]
    assert hook.theta0 is not None and all(r["passed"] for r in hook.history)
    assert [e["to_iteration"] for e in events(tmp_path / "run") if e["event"] == "restore"] == [0]
    assert [r["iteration"] for r in hook.history] == [1, 2, 3, 4, 5, 6]  # le checkpoint 0 a été tourné ensuite


def test_reprise_depuis_le_seul_checkpoint_0_gele_theta0_aux_bons_poids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : run arrêté avant la 1re itération (seul le checkpoint 0 existe), reprise dans de nouveaux objets
    seen: list[dict[str, Any]] = []

    def match(a: Any, b: Any, sim: Any, n: int, **kw: Any) -> Any:
        if not isinstance(b, RandomPolicy):
            seen.append({k: v.clone() for k, v in b.net.state_dict().items()})
        return _result(30, 0) if isinstance(b, RandomPolicy) else PASS_FROZEN

    monkeypatch.setattr(hook_mod, "play_match", match)
    cfg = cfg_of()
    out = tmp_path / "run"
    go(cfg, out, eval_hook=run_hook(out, cfg, EVERY), max_iterations=0)
    ck0 = CheckpointManager(out / "checkpoints").load_latest()
    assert ck0 is not None and ck0[1]["iteration"] == 0
    assert ck0[1]["extra"]["evalhook"]["theta0"] is not None  # checkpoint 0 autosuffisant
    theta0 = ck0[1]["trainer"]["learner"]["policy"]
    # Act : nouveau hook dont la politique liée a d'autres poids que θ₀ au moment de bind_policy
    torch.manual_seed(1234)
    go(cfg, out, eval_hook=run_hook(out, cfg, EVERY), resume=True, max_iterations=1)
    # Assert : le gelé de la 1re évaluation a exactement les poids de θ₀
    assert len(seen) == 1 and all(torch.equal(seen[0][k], theta0[k]) for k in theta0)


def test_reprise_sans_etat_d_evaluation_avertit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : run sans hook, puis reprise avec hook (le checkpoint n'a pas extra["evalhook"])
    monkeypatch.setattr(hook_mod, "play_match", Script("F" * 10))
    cfg = cfg_of()
    out = tmp_path / "run"
    go(cfg, out, max_iterations=2)
    # Act
    go(cfg, out, eval_hook=run_hook(out, cfg, EVERY), resume=True)
    # Assert
    w = [e for e in events(out) if e["event"] == "evalhook_warning"]
    assert len(w) == 1 and "θ₀" in w[0]["message"] and w[0]["iteration"] == 2
