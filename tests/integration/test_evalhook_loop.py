"""`training.evalhook` branché sur la vraie boucle (`training.run`) avec le simulateur factice de `tests/unit`
(sans RocketGPU) : S1 armé après un checkpoint sain, évaluation de l'itération divergente écartée après restauration,
trajectoire d'entraînement identique avec ou sans évaluation. En intégration pour garder `tests/unit` sous 60 s
(≈ 3,3 s à eux trois, mesuré le 2026-11-14)."""

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
from test_evalhook import Script  # noqa: E402
from test_run import StatefulFakeSim, adam_factory, cfg_of, events, go, inject  # noqa: E402

from models import PolicyNet  # noqa: E402
from training.checkpoint import CheckpointIdentity, CheckpointManager  # noqa: E402
from training.evalhook import EVALS_FILE, EvalHook, EvalHookConfig, latest_healthy_iteration  # noqa: E402

hook_mod = importlib.import_module("training.evalhook")


def run_hook(out: Path, cfg: Any, hook_cfg: EvalHookConfig, sim_factory: Any = None) -> EvalHook:
    ck = out / "checkpoints"
    return EvalHook(
        hook_cfg,
        batch_agent_steps=cfg.batch_agent_steps,
        run_seed=cfg.seed,
        out_dir=out,
        snapshot_dir=ck / "eval_snapshots",
        policy_factory=lambda: PolicyNet(7, 90, (16, 16)),
        sim_factory=sim_factory or (lambda n: StatefulFakeSim(n, 2)),
        identity=CheckpointIdentity.current(reward_config_hash="r", obs_dim=7, tick_skip=8),
        reward_config=cfg.reward.build(),
        build_obs=fake_obs,
        obs_dim=7,
        healthy_probe=lambda: latest_healthy_iteration(ck),
    )


def test_boucle_reelle_s1_arme_apres_checkpoint_sain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : 32 transitions par itération, une évaluation par itération, checkpoint sain toutes les 2 itérations
    monkeypatch.setattr(hook_mod, "play_match", Script("PPPPPP"))
    cfg = cfg_of()
    hook = run_hook(tmp_path / "run", cfg, EvalHookConfig(every_timesteps=32, frozen_lag_timesteps=64))
    # Act
    s = go(cfg, tmp_path / "run", s1_provider=hook, optimizer_factory=hook.wrap_optimizer_factory(adam_factory))
    # Assert : réussites à 1, 2, 3 ; 3 couverte par le checkpoint de l'itération 4 ; armé à l'appel 5
    assert s.status == "completed" and s.s1_reached and s.s1_iteration == 5 and hook.s1_eval_iteration == 3
    assert [e["iteration"] for e in events(tmp_path / "run") if e["event"] == "s1_reached"] == [5]
    # les instantanés ne sont pas des checkpoints de reprise (dossier séparé, hors rotation keep)
    assert len(list((tmp_path / "run" / "checkpoints" / "eval_snapshots").iterdir())) == 1 + 5


def test_boucle_reelle_evaluation_de_l_iteration_divergente_ecartee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : NaN au 3e update -> RESTORE vers l'itération 2 (checkpoint sain tous les 2) ; toutes les évaluations
    # réussissent. Sans retour en arrière, la réussite de l'itération divergente compterait.
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, grad_norm=float("nan")) if n == 3 else st)
    monkeypatch.setattr(hook_mod, "play_match", Script("P" * 20))
    cfg = cfg_of()
    hook = run_hook(tmp_path / "run", cfg, EvalHookConfig(every_timesteps=32, frozen_lag_timesteps=64))
    # Act
    s = go(cfg, tmp_path / "run", s1_provider=hook, optimizer_factory=hook.wrap_optimizer_factory(adam_factory))
    # Assert
    log = [json.loads(x) for x in (tmp_path / "run" / EVALS_FILE).read_text(encoding="utf-8").splitlines()]
    rbs = [e for e in log if e["event"] == "rollback"]
    assert s.restores == 1 and len(rbs) == 1 and rbs[0]["dropped_eval_iterations"] == [3]
    its = [r["iteration"] for r in hook.history]
    assert its == sorted(its) and len(its) == len(set(its))
    assert s.s1_reached and hook.s1_eval_iteration is not None and hook.s1_eval_iteration >= 3


def test_trajectoire_identique_avec_ou_sans_evaluation(tmp_path: Path) -> None:
    # vrai `play_match` sur un simulateur factice dédié : aucun générateur global consommé
    cfg = cfg_of()
    a = go(cfg, tmp_path / "a")
    small = EvalHookConfig(every_timesteps=64, frozen_lag_timesteps=128, episodes_random=8, episodes_frozen=8, n_envs=2)
    hook = run_hook(tmp_path / "b", cfg, small)
    b = go(cfg, tmp_path / "b", s1_provider=hook, optimizer_factory=hook.wrap_optimizer_factory(adam_factory))
    assert len(hook.history) == 3 and all(r["random"]["episodes"] == 8 for r in hook.history)
    la = CheckpointManager(tmp_path / "a" / "checkpoints").load_latest()
    lb = CheckpointManager(tmp_path / "b" / "checkpoints").load_latest()
    assert la is not None and lb is not None
    pa, pb = la[1]["trainer"]["learner"]["policy"], lb[1]["trainer"]["learner"]["policy"]
    assert all(torch.equal(pa[k], pb[k]) for k in pa)
    assert a.goals_total == b.goals_total
