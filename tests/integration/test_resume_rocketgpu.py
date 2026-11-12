"""Reprise exacte avec RocketGPU CPU (T5) : restauration bit à bit du simulateur, reprise PPO de bout en bout avec le
vrai optimiseur `torch.optim.Adam` (fused, CPU) et repli « modulo reset des arènes ». Sauté si RocketGPU est absent.

Environnement : `ROCKETGPU_PY_DIR`, `ROCKETGPU_COLLISION_MESH_DIR` (voir `test_env_rocketgpu.py`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from env import BatchEnv, InitialStateSampler, RocketGPUSimulator, SimulatorUnavailableError  # noqa: E402
from models import PolicyNet, ValueNet  # noqa: E402
from obs.schema import OBS_DIM  # noqa: E402
from rewards import RewardConfig, RewardFunction  # noqa: E402
from training import (  # noqa: E402
    CheckpointIdentity,
    CheckpointManager,
    Collector,
    Learner,
    OverlapTrainer,
    PPOConfig,
    capture,
    make_optimizer,
    restore,
)

try:
    _PROBE = RocketGPUSimulator(2)
    _SKIP: str | None = None
except (SimulatorUnavailableError, OSError) as exc:
    _SKIP = f"RocketGPU indisponible : {exc}"
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

E = 8
CPU = torch.device("cpu")


def _digest(sim: RocketGPUSimulator) -> str:
    h = hashlib.sha256()
    for name, arr in sim.raw_arrays.items():
        h.update(name.encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _env(p_kick: float = 0.3, seed: int = 0) -> tuple[BatchEnv, RocketGPUSimulator]:
    sim = RocketGPUSimulator(E, 2, 8)
    return BatchEnv(sim, RewardFunction(RewardConfig()), InitialStateSampler(2, p_kick, seed)), sim


def _actions(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 90, n).astype(np.int64)


def test_restauration_simulateur_bit_a_bit_et_chronologie_identique() -> None:
    """Mesure T5 : état sauvegardé, K steps, restauration (dans une AUTRE instance), mêmes K steps : identique ?"""
    # Arrange
    rng = np.random.default_rng(5)
    pre = [_actions(rng, E * 2) for _ in range(40)]
    post = [_actions(rng, E * 2) for _ in range(60)]
    env, sim = _env()
    env.reset(1)
    obs0: Any = None
    for a in pre:
        obs0 = env.step(a)[0]
    snap = env.state_dict()
    assert snap["sim"] is not None
    ref: list[Any] = []
    for a in post:
        out = env.step(a)
        ref.append((out[0].copy(), out[1].copy(), out[2].copy(), out[3].copy(), _digest(sim)))
    # Act : nouvelle instance de simulateur et d'env, état rechargé
    env2, sim2 = _env(p_kick=0.9, seed=123)
    env2.reset(77)
    assert env2.load_state_dict(snap) is None  # None = restauration exacte
    got = []
    for a in post:
        out = env2.step(a)
        got.append((out[0].copy(), out[1].copy(), out[2].copy(), out[3].copy(), _digest(sim2)))
    # Assert : observations, récompenses, terminaisons ET octets du simulateur identiques à chaque step
    assert obs0 is not None
    for (o1, r1, t1, u1, d1), (o2, r2, t2, u2, d2) in zip(ref, got, strict=True):
        assert np.array_equal(o1, o2) and np.array_equal(r1, r2) and np.array_equal(t1, t2) and np.array_equal(u1, u2)
        assert d1 == d2


def _ppo_setup(seed: int, p_kick: float = 0.3) -> tuple[BatchEnv, RocketGPUSimulator, OverlapTrainer, PPOConfig]:
    env, sim = _env(p_kick, seed)
    torch.manual_seed(seed)
    cfg = PPOConfig(total_iterations=10, minibatch_size=128, target_batch_size=256, epochs=1)
    pol, val = PolicyNet(OBS_DIM, hidden=(32, 32)), ValueNet(OBS_DIM, hidden=(32, 32))
    actor = PolicyNet(OBS_DIM, hidden=(32, 32))
    learner = Learner(
        pol, val, cfg, CPU, optimizer=make_optimizer([*pol.parameters(), *val.parameters()], cfg, CPU), seed=seed
    )
    collector = Collector(env, actor, cfg.rollout_steps(env.n_agents), CPU, seed=seed)
    return env, sim, OverlapTrainer(learner, collector, overlap=True), cfg


def _ident(env: BatchEnv) -> CheckpointIdentity:
    return CheckpointIdentity.current(
        reward_config_hash=RewardFunction(RewardConfig()).config_hash, obs_dim=env.obs_dim, tick_skip=env.tick_skip
    )


def test_reprise_ppo_de_bout_en_bout_avec_adam_reel_et_rocketgpu(tmp_path: Path) -> None:
    # Arrange : référence de 3 itérations d'affilée (recouvrement, vrai Adam fused)
    _, sim_a, tr_a, cfg = _ppo_setup(11)
    ref = [tr_a.iterate().stats for _ in range(3)]
    ref_sim = _digest(sim_a)
    ref_w = {k: v.clone() for k, v in tr_a.learner.policy.state_dict().items()}
    ref_adam = tr_a.learner.optimizer.state_dict()  # type: ignore[attr-defined]
    # Act : 1 itération, checkpoint (état du sim inclus), objets neufs d'une autre seed, reprise, 2 itérations
    env_b, _, tr_b, _ = _ppo_setup(11)
    got = [tr_b.iterate().stats]
    ident = _ident(env_b)
    mgr = CheckpointManager(tmp_path)
    path = mgr.save(capture(tr_b, identity=ident, train_config=cfg, iteration=1, env_state=env_b.state_dict()))
    tr_b.close()
    env_c, sim_c, tr_c, _ = _ppo_setup(404, p_kick=0.9)
    info = restore(mgr.load(path), tr_c, expected=ident, train_config=cfg, env=env_c)
    assert info.exact_env is True
    got += [tr_c.iterate().stats for _ in range(2)]
    # Assert
    assert got == ref
    assert _digest(sim_c) == ref_sim
    for k, v in ref_w.items():
        assert torch.equal(v, tr_c.learner.policy.state_dict()[k])
    adam_c = tr_c.learner.optimizer.state_dict()  # type: ignore[attr-defined]
    # le format de fichier n'a pas de tuple : betas revient en liste (même valeurs, sans effet sur Adam)
    norm = [{k: list(v) if isinstance(v, tuple) else v for k, v in g.items()} for g in ref_adam["param_groups"]]
    assert adam_c["param_groups"] == norm
    for i, st in ref_adam["state"].items():
        for key, val in st.items():
            assert torch.equal(val, adam_c["state"][i][key]), (i, key)


def test_reprise_modulo_reset_des_arenes_sans_etat_simulateur(tmp_path: Path) -> None:
    # Arrange
    env_b, _, tr_b, cfg = _ppo_setup(5)
    tr_b.iterate()
    env_b.set_p_kick(0.25)
    ident = _ident(env_b)
    mgr = CheckpointManager(tmp_path)
    path = mgr.save(capture(tr_b, identity=ident, train_config=cfg, iteration=1, env_state=env_b.state_dict(False)))
    tr_b.close()
    # Act
    env_c, _, tr_c, _ = _ppo_setup(6, p_kick=0.9)
    info = restore(mgr.load(path), tr_c, expected=ident, train_config=cfg, env=env_c)
    stats = tr_c.iterate().stats
    # Assert : reprise non exacte côté arènes (documenté), tout le reste restauré, entraînement sain
    assert info.exact_env is False
    assert env_c.p_kick == 0.25
    assert tr_c.learner.version == 2 and stats.policy_version == 2 and np.isfinite(stats.policy_loss)
