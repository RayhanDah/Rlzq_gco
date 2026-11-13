"""PPO v0 : réseaux, configuration, mise à jour, apprentissage d'un env jouet, recouvrement collecte/apprentissage.

Tout tourne en CPU. Les optimiseurs `torch.optim` ne sont pas utilisés : leur première création importe `torch._dynamo`
(≈ 12 à 47 s sur ce laptop, mesuré le 2026-11-11) ; `_ManualAdam` implémente la même règle d'Adam (Kingma & Ba,
Algorithme 1) via l'interface injectable `OptimizerLike`.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from _helpers import assert_close  # noqa: E402

from models import N_ACTIONS, PolicyNet, ValueNet, log_prob_entropy, sample_actions  # noqa: E402
from training import (  # noqa: E402
    Collector,
    Learner,
    LearnerError,
    OverlapTrainer,
    PPOConfig,
    RolloutBatch,
    resolve_device,
)

CPU = torch.device("cpu")


class _ManualAdam:
    """Adam minimal (zero_grad, step, param_groups[0]["lr"]), sans torch.optim."""

    def __init__(self, params: list[Any], lr: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-5):
        self.params = params
        self.param_groups: list[dict[str, Any]] = [{"lr": lr}]
        self.b1, self.b2, self.eps, self.t = betas[0], betas[1], eps, 0
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.t += 1
        lr = self.param_groups[0]["lr"]
        with torch.no_grad():
            for p, m, v in zip(self.params, self.m, self.v, strict=True):
                if p.grad is None:
                    continue
                m.mul_(self.b1).add_(p.grad, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(p.grad, p.grad, value=1 - self.b2)
                m_hat = m / (1 - self.b1**self.t)
                v_hat = v / (1 - self.b2**self.t)
                p -= lr * m_hat / (v_hat.sqrt() + self.eps)


class Corridor:
    """Env jouet vectorisé au contrat `BatchEnv` (ADR-003 §5), N agents indépendants.

    Position 0..4, départ en 2 ; contexte c ∈ {0, 1} tiré au reset. obs = one-hot(position) ‖ one-hot(c) (D = 7).
    Actions : 0 gauche, 1 droite, 2+ immobile. Atteindre l'extrémité désignée par c (4 si c = 1, 0 sinon) : +1,
    terminal ; l'autre extrémité : −1, terminal ; `horizon` steps sans terminal : troncature (récompense 0).
    Politique aléatoire uniforme : retour moyen nul par symétrie ; optimale : +1 en 2 steps.
    """

    obs_dim = 7

    def __init__(self, n_agents: int, horizon: int = 6, n_actions: int = 3) -> None:
        self._n, self.horizon, self.n_actions = n_agents, horizon, n_actions
        self.rng = np.random.default_rng(0)
        self.pos = np.zeros(n_agents, np.int64)
        self.ctx = np.zeros(n_agents, np.int64)
        self.t = np.zeros(n_agents, np.int64)
        self.returns: list[float] = []

    @property
    def n_agents(self) -> int:
        return self._n

    def _obs(self) -> np.ndarray:
        o = np.zeros((self._n, 7), np.float32)
        o[np.arange(self._n), self.pos] = 1.0
        o[np.arange(self._n), 5 + self.ctx] = 1.0
        return o

    def _reset(self, idx: np.ndarray) -> None:
        self.pos[idx] = 2
        self.ctx[idx] = self.rng.integers(0, 2, idx.size)
        self.t[idx] = 0

    def reset(self, seed: int) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        self._reset(np.arange(self._n))
        return self._obs()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
        assert actions.shape == (self._n,) and actions.dtype == np.int64
        move = np.where(actions == 0, -1, np.where(actions == 1, 1, 0))
        self.pos = np.clip(self.pos + move, 0, 4)
        self.t += 1
        at_end = (self.pos == 0) | (self.pos == 4)
        good = np.where(self.ctx == 1, self.pos == 4, self.pos == 0)
        rew = np.where(at_end, np.where(good, 1.0, -1.0), 0.0).astype(np.float32)
        term = at_end
        trunc = ~term & (self.t >= self.horizon)
        done = np.flatnonzero(term | trunc)
        final_obs = self._obs()[done]
        self.returns += rew[done].tolist()
        self._reset(done)
        return self._obs(), rew, term, trunc, {"final_obs": final_obs, "final_agent": done.astype(np.int64)}


def _small_cfg(**kw: Any) -> PPOConfig:
    base: dict[str, Any] = {"total_iterations": 30, "minibatch_size": 256, "target_batch_size": 1024}
    return PPOConfig(**(base | kw))


def _setup(
    n_agents: int = 64, seed: int = 0, lr: float = 3e-3, overlap: bool = True, **cfg_kw: Any
) -> tuple[Corridor, OverlapTrainer]:
    torch.manual_seed(seed)
    cfg = _small_cfg(lr_start=lr, lr_end=lr, **cfg_kw)
    env = Corridor(n_agents)
    policy = PolicyNet(env.obs_dim, n_actions=3, hidden=(64, 64))
    value = ValueNet(env.obs_dim, hidden=(64, 64))
    actor = PolicyNet(env.obs_dim, n_actions=3, hidden=(64, 64))
    learner = Learner(policy, value, cfg, CPU, optimizer=_ManualAdam([*policy.parameters(), *value.parameters()], lr))
    collector = Collector(env, actor, cfg.rollout_steps(n_agents), CPU, seed=seed)
    return env, OverlapTrainer(learner, collector, overlap=overlap)


# --- Réseaux et configuration ---------------------------------------------------------------------------------------


def test_networks_follow_ppo_v0_config() -> None:
    # Arrange
    torch.manual_seed(0)
    policy, value = PolicyNet(109), ValueNet(109)
    x = torch.randn(5, 109)
    # Act
    with torch.no_grad():
        logits, v = policy(x), value(x)
    # Assert : π 3×256 → 90, V 3×512 → 1 (`docs/01_papers.md` §2) ; tête π gain 0,01 ⇒ politique quasi uniforme.
    assert logits.shape == (5, N_ACTIONS) and v.shape == (5,)
    assert [m.out_features for m in policy.net if isinstance(m, torch.nn.Linear)] == [256, 256, 256, 90]
    assert [m.out_features for m in value.net if isinstance(m, torch.nn.Linear)] == [512, 512, 512, 1]
    _, ent = log_prob_entropy(logits, torch.zeros(5, dtype=torch.long))
    assert float(ent.min()) > math.log(90) - 0.01  # écart mesuré 3,6e-5 nat (seed 0) ; 0,01 = marge large


def test_sampling_is_reproducible_with_generator_and_logp_consistent() -> None:
    logits = torch.randn(1000, 7)
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    a1, lp1 = sample_actions(logits, g1)
    a2, _ = sample_actions(logits, g2)
    assert torch.equal(a1, a2)
    expected = torch.log_softmax(logits, -1).gather(1, a1[:, None]).squeeze(1)
    assert_close(lp1, expected, atol=1e-5, rtol=1.3e-6)  # défauts float32 de torch.testing.assert_close


def test_config_defaults_match_papers_section_2() -> None:
    cfg = PPOConfig(total_iterations=11)
    assert (cfg.gamma, cfg.gae_lambda, cfg.clip, cfg.max_grad_norm) == (0.995, 0.95, 0.2, 0.5)
    assert (cfg.epochs, cfg.minibatch_size, cfg.target_batch_size) == (2, 25_000, 50_000)
    assert (cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps) == (0.9, 0.999, 1e-5)
    # LR 2e-4 → 5e-5 et entropie 0,01 → 0,002, linéaires sur le run.
    assert cfg.lr_at(0) == pytest.approx(2e-4) and cfg.lr_at(10) == pytest.approx(5e-5)
    assert cfg.lr_at(5) == pytest.approx(1.25e-4) and cfg.lr_at(99) == pytest.approx(5e-5)
    assert cfg.ent_coef_at(0) == pytest.approx(0.01) and cfg.ent_coef_at(10) == pytest.approx(0.002)
    assert cfg.rollout_steps(2048) == 25 and cfg.rollout_steps(8192) == 7
    with pytest.raises(ValueError):
        PPOConfig(total_iterations=0)
    with pytest.raises(ValueError):
        PPOConfig(total_iterations=1, minibatch_size=3)


def test_resolve_device() -> None:
    assert resolve_device("cpu") == CPU
    assert resolve_device("auto").type in ("cpu", "cuda", "xpu")


# --- Mise à jour PPO ------------------------------------------------------------------------------------------------


def _collect_one(n_agents: int = 16, mask: np.ndarray | None = None) -> tuple[Learner, RolloutBatch]:
    torch.manual_seed(0)
    cfg = _small_cfg(minibatch_size=64, target_batch_size=128)
    env = Corridor(n_agents, horizon=3)
    policy, value = PolicyNet(7, 3, (16,)), ValueNet(7, (16,))
    learner = Learner(policy, value, cfg, CPU, _ManualAdam([*policy.parameters(), *value.parameters()], 1e-2))
    actor = PolicyNet(7, 3, (16,))
    actor.load_state_dict(policy.state_dict())
    batch = Collector(env, actor, cfg.rollout_steps(n_agents), CPU, seed=1, learn_mask=mask).collect(0)
    return learner, batch


def test_update_stats_finite_and_version_increments() -> None:
    # Arrange
    learner, batch = _collect_one()
    before = [p.detach().clone() for p in learner.policy.parameters()]
    # Act
    stats = learner.update(batch)
    # Assert
    assert all(math.isfinite(float(v)) for v in stats.as_dict().values())
    assert learner.version == 1 and stats.policy_version == 1 and stats.staleness == 0
    # Les obs finales tronquées ne servent qu'au bootstrap : n_samples = T·N, pas T·N + K.
    assert batch.final_obs.shape[0] > 0
    assert stats.n_samples == batch.n_steps * batch.n_agents
    assert stats.n_minibatches == 2 and stats.n_trunc == batch.final_obs.shape[0]
    assert 0.0 <= stats.clip_frac <= 1.0 and stats.approx_kl >= -1e-6  # k3 ≥ 0 à l'arrondi près
    assert any(not torch.equal(a, b) for a, b in zip(before, learner.policy.parameters(), strict=True))


def test_first_minibatch_has_unit_ratio_and_no_clipping_at_staleness_zero() -> None:
    # Arrange : 1 epoch, 1 minibatch ⇒ ratio évalué avant tout pas d'optimiseur, = 1 exactement (même π).
    learner, batch = _collect_one()
    learner.cfg = _small_cfg(minibatch_size=4096, epochs=1)
    # Act
    stats = learner.update(batch)
    # Assert
    assert stats.clip_frac == 0.0 and abs(stats.approx_kl) < 1e-6


def test_update_rejects_stale_batch() -> None:
    learner, batch = _collect_one()
    learner.version = 2  # staleness 2 > max_staleness 1
    with pytest.raises(ValueError):
        learner.update(batch)


def test_learn_mask_restricts_training_samples() -> None:
    mask = np.zeros(16, np.bool_)
    mask[::2] = True
    learner, batch = _collect_one(mask=mask)
    stats = learner.update(batch)
    assert stats.n_samples == batch.n_steps * 8


# --- Apprentissage sur l'env jouet ----------------------------------------------------------------------------------


def test_ppo_solves_corridor_toy_env() -> None:
    """PPO, recouvrement activé, seed 0 : le retour moyen par épisode passe de ≈ 0 (aléatoire, par symétrie) à ≥ 0,95.

    Seuils : l'optimum vaut 1 ; une politique qui choisit le bon côté avec probabilité p a un retour ≈ 2p − 1, donc
    0,95 exige p ≳ 0,97. Au départ, ≈ 1 000 épisodes de retour ±1 ou 0 : écart-type de la moyenne ≤ 0,032, d'où
    |·| < 0,2 (> 6σ). Mesuré le 2026-11-11 (CPU, script hors dépôt, seeds 0–4) : retour du premier lot −0,064 à 0,015 ;
    ≥ 0,993 dès l'itération 10 et 1,0 à l'itération 30 pour les 5 seeds ; ≈ 1,4–2,0 s par run de 31 itérations.
    Allégé le 2026-11-13 à 15 itérations (seuil 0,95 inchangé) : mesuré seeds 0–4, retour final 1,0 pour les 5 (0,971 à
    0,998 avec 10 itérations seulement), 500 épisodes par lot ; la config garde `total_iterations=30`.
    """
    # Arrange
    env, trainer = _setup()
    # Act
    with trainer:
        trainer.iterate()  # lots 0 et 1 collectés par la politique initiale (v0)
        initial = float(np.mean(env.returns))
        for _ in range(13):
            trainer.iterate()
    env.returns.clear()
    trainer.iterate()  # un lot complet collecté avec la politique finale (version 15)
    # Assert
    assert abs(initial) < 0.2
    assert len(env.returns) > 100
    assert float(np.mean(env.returns)) >= 0.95


# --- Recouvrement ---------------------------------------------------------------------------------------------------


def test_overlap_versions_finite_stats_and_thread_stopped() -> None:
    # Arrange
    _, trainer = _setup(n_agents=16)
    # Act
    results = [trainer.iterate() for _ in range(4)]
    # Assert
    for k, r in enumerate(results):
        assert all(math.isfinite(float(v)) for v in r.stats.as_dict().values())
        assert r.policy_version == k + 1
        # Lot 0 collecté par π v0 ; lot k (k ≥ 1) collecté pendant l'update k−1, donc par π v(k−1) : staleness 1.
        assert r.learned_batch_version == max(k - 1, 0)
        assert r.stats.staleness == (0 if k == 0 else 1)
    pending = trainer.pending_batch
    assert pending is not None and pending.policy_version == 3
    # L'acteur est une copie à jour de π, distincte du module du learner.
    actor = trainer.collector.policy
    assert actor is not trainer.learner.policy
    for a, b in zip(actor.parameters(), trainer.learner.policy.parameters(), strict=True):
        assert torch.equal(a, b)
    assert not any(t.name == "ppo-learner" for t in threading.enumerate())


def test_synchronous_mode_has_zero_staleness() -> None:
    _, trainer = _setup(n_agents=16, overlap=False)
    results = [trainer.iterate() for _ in range(3)]
    assert [r.stats.staleness for r in results] == [0, 0, 0]
    assert [r.learned_batch_version for r in results] == [0, 1, 2]


def test_learner_exception_propagates_and_thread_stops() -> None:
    # Arrange
    _, trainer = _setup(n_agents=16)
    trainer.iterate()

    def boom(batch: RolloutBatch) -> Any:
        raise FloatingPointError("NaN simulé")

    trainer.learner.update = boom  # type: ignore[method-assign]
    # Act / Assert
    with pytest.raises(LearnerError) as info:
        trainer.iterate()
    assert isinstance(info.value.__cause__, FloatingPointError)
    assert not any(t.name == "ppo-learner" for t in threading.enumerate())
    with pytest.raises(RuntimeError):
        trainer.iterate()  # trainer inutilisable après échec


def test_trainer_rejects_shared_policy_module() -> None:
    torch.manual_seed(0)
    cfg = _small_cfg()
    env = Corridor(4)
    policy, value = PolicyNet(7, 3, (8,)), ValueNet(7, (8,))
    learner = Learner(policy, value, cfg, CPU, _ManualAdam([*policy.parameters()], 1e-3))
    with pytest.raises(ValueError):
        OverlapTrainer(learner, Collector(env, policy, 4, CPU, seed=0))
