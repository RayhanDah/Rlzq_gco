"""Self-imitation (`training.sil`, ADR-009) : données admises, retours, priorités, perte, annealing, journal C4, gardes,
reprise, mémoire, parité float16, et régression bit à bit « SIL désactivée = PPO d'avant ». CPU, Adam manuel."""

from __future__ import annotations

import copy
import dataclasses
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from _ppo_reference import ReferenceLearner  # noqa: E402
from test_checkpoint import _Adam  # noqa: E402
from test_env import fake_obs  # noqa: E402
from test_obs import prev_of, random_state  # noqa: E402
from test_ppo import Corridor, _ManualAdam, _small_cfg  # noqa: E402
from test_run import BASE, CheckpointManager, StatefulFakeSim, cfg_of, events, go, inject, make_factory, rows  # noqa: E402

from models import PolicyNet, ValueNet  # noqa: E402
from models.policy import init_linear, log_prob_entropy  # noqa: E402
from obs import build_obs  # noqa: E402
from training import Collector, ConfigError, Learner, OverlapTrainer, RunConfig, make_env  # noqa: E402
from training.checkpoint import decode, encode  # noqa: E402
from training.sil import (  # noqa: E402
    EpisodeTracker,
    ReplayBuffer,
    SelfImitation,
    SILConfig,
    SILGuardState,
    goal_returns,
    sil_guard,
    sil_losses,
)

CPU = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[2]
GAMMA = 0.995


def scfg(**kw: Any) -> SILConfig:
    """Petite config SIL activée, coefficient constant (annealing au-delà de tout test)."""
    base: dict[str, Any] = {
        "enabled": True,
        "anneal_start_timesteps": 10**12,
        "anneal_end_timesteps": 2 * 10**12,
        "batch_size": 4,
        "capacity": 64,
        "horizon_steps": 8,
        "store_float16": False,
    }
    return SILConfig(**(base | kw))


def sep(**kw: Any) -> SILConfig:
    """Mode separate (première version de l'ADR-009) : Adam distinct, 2 pas, clip 0,5."""
    return scfg(mode="separate", updates_per_iteration=2, max_grad_norm=0.5, **kw)


def make_sil(cfg: SILConfig | None = None, obs_dim: int = 3, optimizer: Any = None) -> SelfImitation:
    cfg = cfg or scfg()
    return SelfImitation(cfg, GAMMA, obs_dim, 5, seed=0, optimizer=optimizer)


def steps_arrays(t_len: int, n: int, d: int = 3) -> dict[str, np.ndarray]:
    """Lot synthétique (T, N, …) : obs[t, a] = (t, a, 0), action = t mod 5, aucune fin."""
    obs = np.zeros((t_len, n, d), np.float32)
    obs[..., 0] = np.arange(t_len)[:, None]
    obs[..., 1] = np.arange(n)[None, :]
    return {
        "obs": obs,
        "act": np.broadcast_to((np.arange(t_len) % 5)[:, None], (t_len, n)).astype(np.int64).copy(),
        "rew": np.zeros((t_len, n), np.float32),
        "term": np.zeros((t_len, n), np.bool_),
        "trunc": np.zeros((t_len, n), np.bool_),
    }


def ingest(
    sil: SelfImitation, b: dict[str, np.ndarray], mask: np.ndarray | None = None, it: int = 0
) -> tuple[int, int]:
    n = b["obs"].shape[1]
    m = np.ones(n, np.bool_) if mask is None else mask
    return sil.ingest(b["obs"], b["act"], b["rew"], b["term"], b["trunc"], m, it)


# ---------------------------------------------------------------------------------------------------- données (D1)


def test_retours_monte_carlo_exacts() -> None:
    r = goal_returns(1.0, 4, GAMMA)
    assert r.dtype == np.float32
    np.testing.assert_array_equal(r, np.array([GAMMA**3, GAMMA**2, GAMMA, 1.0], np.float32))
    with pytest.raises(ValueError):
        goal_returns(1.0, 0, GAMMA)


def test_episode_gagne_stocke_dans_l_ordre_avec_retours_exacts() -> None:
    # Arrange : agent 1 marque au step 4 (5 transitions), agent 0 rien
    sil, b = make_sil(), steps_arrays(6, 2)
    b["rew"][4, 1], b["term"][4, 1] = 1.0, True
    # Act
    n_ep, n_tr = ingest(sil, b)
    # Assert
    buf = sil.buffer
    assert buf is not None and (n_ep, n_tr) == (1, 5) and buf.size == 5
    np.testing.assert_array_equal(buf.obs[:5, 0], np.arange(5))  # t = 0..4, du plus ancien au but
    np.testing.assert_array_equal(buf.obs[:5, 1], np.ones(5))  # agent 1
    np.testing.assert_array_equal(buf.act[:5], np.arange(5) % 5)
    np.testing.assert_array_equal(buf.ret[:5], goal_returns(1.0, 5, GAMMA))
    assert buf.ret[4] == 1.0  # la transition du but porte R = r_T


def test_troncature_perdant_et_agent_gele_jamais_admis() -> None:
    # Arrange : agent 0 tronqué, agent 1 encaisse (−1), agent 2 (gelé) marque, agent 3 marque mais r = 0 au terminal
    sil, b = make_sil(), steps_arrays(5, 4)
    b["trunc"][3, 0] = True
    b["rew"][3, 1], b["term"][3, 1] = -1.0, True
    b["rew"][3, 2], b["term"][3, 2] = 1.0, True
    b["term"][3, 3] = True
    mask = np.array([True, True, False, True])
    # Act
    n_ep, _ = ingest(sil, b, mask)
    # Assert
    assert n_ep == 0 and sil.buffer is not None and sil.buffer.size == 0


def test_perdants_admis_seulement_avec_keep_conceded() -> None:
    sil, b = make_sil(scfg(keep_conceded=True)), steps_arrays(4, 1)
    b["rew"][2, 0], b["term"][2, 0] = -1.0, True
    assert ingest(sil, b) == (1, 3)
    assert sil.buffer is not None
    np.testing.assert_array_equal(sil.buffer.ret[:3], goal_returns(-1.0, 3, GAMMA))


def test_episode_plus_long_que_la_fenetre_et_a_cheval_sur_deux_lots() -> None:
    # Arrange : fenêtre 8 ; épisode de 13 transitions sur deux lots (7 + 6), but au dernier step
    sil = make_sil()
    b1, b2 = steps_arrays(7, 1), steps_arrays(6, 1)
    b2["obs"][..., 0] += 7
    b2["rew"][5, 0], b2["term"][5, 0] = 1.0, True
    # Act
    ingest(sil, b1)
    n_ep, n_tr = ingest(sil, b2)
    # Assert : les 8 dernières transitions (t = 5..12), retours γ^7..γ^0, sans biais
    buf = sil.buffer
    assert buf is not None and (n_ep, n_tr) == (1, 8)
    np.testing.assert_array_equal(buf.obs[:8, 0], np.arange(5, 13))
    np.testing.assert_array_equal(buf.ret[:8], goal_returns(1.0, 8, GAMMA))


def test_nouvel_episode_apres_une_fin_ne_reprend_rien_de_l_ancien() -> None:
    # Arrange : troncature au step 2, puis but au step 4 : seules les transitions 3 et 4 sont gardées
    sil, b = make_sil(), steps_arrays(5, 1)
    b["trunc"][2, 0] = True
    b["rew"][4, 0], b["term"][4, 0] = 1.0, True
    # Act / Assert
    assert ingest(sil, b) == (1, 2)
    assert sil.buffer is not None
    np.testing.assert_array_equal(sil.buffer.obs[:2, 0], [3.0, 4.0])


def test_recompense_hors_terminal_refusee_et_learn_mask_fixe() -> None:
    sil, b = make_sil(), steps_arrays(3, 2)
    b["rew"][1, 0] = 1.0  # sans terminal : hypothèse GCO §2.1 violée
    with pytest.raises(ValueError, match="hors terminal"):
        ingest(sil, b)
    sil2 = make_sil()
    ingest(sil2, steps_arrays(2, 2))
    with pytest.raises(ValueError, match="learn_mask"):
        ingest(sil2, steps_arrays(2, 2), np.array([True, False]))


# ---------------------------------------------------------------------------------------------------- priorités (D2)


def test_priorites_nouvelles_au_max_puis_mises_a_jour() -> None:
    buf = ReplayBuffer(8, 2, np.float32)
    buf.add(np.zeros((2, 2), np.float32), np.zeros(2, np.int16), np.ones(2, np.float32), 0)
    assert buf.prio[:2].tolist() == [1.0, 1.0]  # buffer vide : priorité 1
    buf.update_priorities(torch.tensor([0, 1]), torch.tensor([0.0, 2.0]), 0.6, 1e-3)
    assert buf.prio[0] == pytest.approx(1e-3**0.6) and buf.prio[1] == pytest.approx(2.001**0.6)
    buf.add(np.zeros((1, 2), np.float32), np.zeros(1, np.int16), np.ones(1, np.float32), 3)
    assert buf.prio[2] == pytest.approx(2.001**0.6) and buf.born[2] == 3  # neuve = max courant


def test_tirage_proportionnel_et_poids_d_importance() -> None:
    # Arrange : priorités 1 et 3
    buf = ReplayBuffer(4, 1, np.float32)
    buf.add(np.zeros((2, 1), np.float32), np.zeros(2, np.int16), np.zeros(2, np.float32), 0)
    buf.prio[:2] = [1.0, 3.0]
    g = torch.Generator().manual_seed(0)
    # Act
    idx, w = buf.sample(20_000, g, 0.1)
    # Assert : P(1) = 0,75 (écart-type 0,003 sur 20 000 tirages) ; w = (p / p_min)^(−β) ≤ 1
    assert float((idx == 1).float().mean()) == pytest.approx(0.75, abs=0.015)
    assert torch.allclose(w[idx == 0], torch.ones(1)) and torch.allclose(w[idx == 1], torch.tensor(3.0**-0.1))


def test_fifo_ecrase_les_plus_anciennes() -> None:
    buf = ReplayBuffer(3, 1, np.float32)
    for k in range(5):
        buf.add(np.full((1, 1), k, np.float32), np.zeros(1, np.int16), np.zeros(1, np.float32), k)
    assert buf.size == 3 and sorted(buf.obs[:, 0].tolist()) == [2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------------------------------- perte (éq. 1–3)


def _nets() -> tuple[PolicyNet, ValueNet]:
    torch.manual_seed(0)
    return PolicyNet(3, 5, (8,)), ValueNet(3, (8,))


def test_clamp_et_aucun_gradient_si_r_inferieur_ou_egal_a_v() -> None:
    # Arrange
    policy, value = _nets()
    obs, act = torch.randn(6, 3), torch.randint(0, 5, (6,))
    with torch.no_grad():
        v = value(obs)
    # Act : R = V − 0,5 (R < V) et R = V (égalité)
    for ret in (v - 0.5, v.clone()):
        for p in [*policy.parameters(), *value.parameters()]:
            p.grad = None
        pl, vl, gap = sil_losses(policy, value, obs, act, ret, torch.ones(6))
        (pl + 0.01 * vl).backward()
        # Assert : (R − V)₊ = 0, gradients nuls partout
        assert torch.equal(gap, torch.zeros(6))
        assert all(torch.count_nonzero(p.grad) == 0 for p in [*policy.parameters(), *value.parameters()])


def test_signe_r_superieur_a_v_imite_l_action_et_releve_v() -> None:
    # Arrange
    policy, value = _nets()
    obs, act = torch.randn(6, 3), torch.randint(0, 5, (6,))
    with torch.no_grad():
        v0, logp0 = value(obs), log_prob_entropy(policy(obs), act)[0]
    ret = v0 + 1.0
    # Act : un pas de descente de gradient
    pl, vl, gap = sil_losses(policy, value, obs, act, ret, torch.ones(6))
    assert torch.allclose(gap, torch.ones(6))
    assert float(vl.detach()) == pytest.approx(0.5)  # ½·(R − V)₊²
    (pl + vl).backward()
    with torch.no_grad():
        for p in [*policy.parameters(), *value.parameters()]:
            p -= 0.05 * p.grad
        v1, logp1 = value(obs), log_prob_entropy(policy(obs), act)[0]
    # Assert : log π(a|s) monte, V monte vers R
    assert float((logp1 - logp0).mean()) > 0 and float((v1 - v0).mean()) > 0


def test_le_poids_r_moins_v_du_terme_de_politique_est_detache() -> None:
    # Le terme de politique ne doit pas entraîner V (Oh §3) : sans terme de valeur, aucun gradient sur V
    policy, value = _nets()
    obs, act = torch.randn(4, 3), torch.randint(0, 5, (4,))
    with torch.no_grad():
        ret = value(obs) + 1.0
    pl, _, _ = sil_losses(policy, value, obs, act, ret, torch.ones(4))
    pl.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in value.parameters())


# ---------------------------------------------------------------------------------------------------- annealing (C2)


def test_coefficient_aux_bornes_du_calendrier_adr_009() -> None:
    cfg = RunConfig.from_yaml(ROOT / "configs" / "int_sil.yaml").sil
    assert cfg is not None and cfg.enabled
    assert [cfg.coef_at(t) for t in (0, 499_999_999, 500_000_000)] == [0.1, 0.1, 0.1]
    assert cfg.coef_at(700_000_000) == pytest.approx(0.05)
    assert cfg.coef_at(899_999_999) == pytest.approx(0.1 * 1 / 400_000_000)
    assert cfg.coef_at(900_000_000) == 0.0 and cfg.coef_at(10**10) == 0.0
    with pytest.raises(ValueError):
        SILConfig(anneal_start_timesteps=2, anneal_end_timesteps=1)


def test_liberation_en_fin_d_annealing() -> None:
    # Arrange : fin de l'annealing à 10 timesteps appris ; 4 appris par itération
    sil = make_sil(scfg(anneal_start_timesteps=0, anneal_end_timesteps=10))
    b = steps_arrays(2, 2)
    b["rew"][1, 0], b["term"][1, 0] = 1.0, True
    # Act : itérations 0, 1 (t = 0, 4 : coefficient > 0), puis 3 (t = 12 ≥ 10)
    it0 = sil.begin(
        **b, learn_mask=np.ones(2, np.bool_), iteration=0, learned_per_iteration=4, device=CPU, n_ppo_steps=1
    )
    assert it0.coef == pytest.approx(0.1) and sil.buffer is not None and sil.buffer.size == 2
    it3 = sil.begin(
        **b, learn_mask=np.ones(2, np.bool_), iteration=3, learned_per_iteration=4, device=CPU, n_ppo_steps=1
    )
    # Assert : plus de buffer ni d'anneau, rien d'ingéré, état minimal relisible
    assert it3.coef == 0.0 and sil.released and sil.buffer is None and sil.tracker is None and sil.nbytes == 0
    assert it3.n_ep == 0 and it3.stats().released == 1.0
    other = make_sil(scfg(anneal_start_timesteps=0, anneal_end_timesteps=10))
    other.load_state_dict(decode(encode(sil.state_dict())))
    assert other.released and other.buffer is None


# ---------------------------------------------------------------------------------------------------- learner


def _learner(sil: SILConfig | None, lr: float = 1e-2, cls: type[Learner] = Learner) -> Learner:
    torch.manual_seed(0)
    cfg = _small_cfg(minibatch_size=64, target_batch_size=128, lr_start=lr, lr_end=lr)
    policy, value = PolicyNet(7, 3, (16,)), ValueNet(7, (16,))
    params = [*policy.parameters(), *value.parameters()]
    sil_opt = _Adam(params, lr) if sil is not None and sil.enabled and sil.mode == "separate" else None
    return cls(policy, value, cfg, CPU, _Adam(params, lr), sil=sil, sil_optimizer=sil_opt)


def _trainer(learner: Learner, n_agents: int = 16, horizon: int = 3) -> OverlapTrainer:
    env = Corridor(n_agents, horizon=horizon)
    actor = PolicyNet(7, 3, (16,))
    return OverlapTrainer(learner, Collector(env, actor, learner.cfg.rollout_steps(n_agents), CPU, seed=1))


def _params(learner: Learner) -> list[Any]:
    return [p.detach().clone() for p in learner.params]


def _same(a: list[Any], b: list[Any]) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_sil_desactivee_bit_a_bit_identique_au_ppo_d_avant() -> None:
    """ADR-009 D7-1 : poids et statistiques identiques bit à bit, après 6 itérations en recouvrement, entre le code
    d'avant la SIL (`_ppo_reference`, copie figée) et le code actuel sans SIL, avec `enabled: false`, et avec la SIL
    activée mais c_sil = 0 (suivi des épisodes actif, aucun pas)."""
    variants = {
        "reference": _learner(None, cls=ReferenceLearner),
        "absente": _learner(None),
        "enabled_false": _learner(scfg(enabled=False)),
        "coef_nul": _learner(scfg(coef=0.0)),
    }
    stats: dict[str, list[dict[str, Any]]] = {}
    for name, learner in variants.items():
        with _trainer(learner) as tr:
            stats[name] = [tr.iterate().stats.as_dict() for _ in range(6)]
    ref = _params(variants["reference"])
    for name in ("absente", "enabled_false", "coef_nul"):
        assert _same(_params(variants[name]), ref), name
        assert stats[name] == stats["reference"], name
    sil = variants["coef_nul"].sil
    assert sil is not None and sil.episodes_total > 0  # le suivi tournait bien
    assert variants["enabled_false"].sil is None and "sil" not in variants["enabled_false"].state_dict()


def _filled(learner: Learner, ret: float, n: int = 32) -> None:
    sil = learner.sil
    assert sil is not None and sil.buffer is not None
    obs = np.random.default_rng(0).normal(size=(n, 7)).astype(np.float32)
    sil.buffer.add(obs, np.zeros(n, np.int16), np.full(n, ret, np.float32), 0)


def _one_batch() -> Any:
    torch.manual_seed(0)
    env = Corridor(16, horizon=3)
    return Collector(env, PolicyNet(7, 3, (16,)), 8, CPU, seed=1).collect(0)


def test_pas_separes_n_alterent_pas_l_etat_adam_de_ppo() -> None:
    # Arrange : SIL activée (buffer prérempli, R = 1 > V) contre PPO seul, même lot, mêmes poids initiaux
    on, off, batch = _learner(sep()), _learner(None), _one_batch()
    on.policy.load_state_dict(off.policy.state_dict())
    _filled(on, 1.0)
    # Act
    s_on, s_off = on.update(batch), off.update(batch)
    # Assert : moments et compteur de l'Adam PPO identiques ; poids différents (pas SIL appliqués) ; Adam SIL avancé
    po, pf = on.optimizer.state_dict(), off.optimizer.state_dict()  # type: ignore[attr-defined]
    assert po["t"] == pf["t"] and all(
        torch.equal(a, b) for a, b in zip(po["m"] + po["v"], pf["m"] + pf["v"], strict=True)
    )
    assert not _same(_params(on), _params(off))
    assert on.sil is not None and on.sil.optimizer is not None and on.sil.optimizer.t == 2  # type: ignore[attr-defined]
    assert s_on.sil["steps_done"] == 2.0 and s_off.sil == {}
    assert {k: v for k, v in s_on.as_dict().items()} == {k: v for k, v in s_off.as_dict().items()}


def test_aucun_pas_ni_gradient_si_tout_le_buffer_a_r_inferieur_a_v() -> None:
    # Arrange : R = −10 < V partout ; lot sans but gagné (rien de neuf n'entre dans le buffer)
    on, off, batch = _learner(sep()), _learner(None), _one_batch()
    batch = dataclasses.replace(batch, rew=np.minimum(batch.rew, 0.0))
    _filled(on, -10.0)
    # Act
    s = on.update(batch).sil
    off.update(batch)
    # Assert : minibatchs tirés mais sautés, Adam SIL intact, poids = PPO seul
    assert (s["steps_done"], s["steps_inactive"], s["frac_pos"], s["grad_share"]) == (0.0, 2.0, 0.0, 0.0)
    assert s["disp_share"] == 0.0
    assert on.sil is not None and on.sil.optimizer is not None and on.sil.optimizer.t == 0  # type: ignore[attr-defined]
    assert _same(_params(on), _params(off))


def test_pas_de_pas_tant_que_le_buffer_est_sous_batch_size() -> None:
    on = _learner(scfg(batch_size=64, capacity=64))
    _filled(on, 1.0, n=10)
    s = on.update(_one_batch()).sil
    assert s["steps_done"] == 0.0 and s["steps_inactive"] == 0.0 and s["buffer_size"] >= 10


def test_journal_c4_complet_fini_et_coherent() -> None:
    # Arrange
    on = _learner(scfg())
    _filled(on, 1.0)
    # Act
    s = on.update(_one_batch()).sil
    # Assert : toutes les clés de l'ADR-009 D5 (hors RAM, ajoutée par la boucle), finies ; parts dans ]0, 1[
    expected = {
        "coef", "steps_done", "buffer_size", "buffer_new", "buffer_age_mean", "loss_policy", "loss_value",
        "frac_pos", "adv_pos_mean", "adv_pos_max", "grad_norm_pi_sil", "grad_norm_pi_ppo", "grad_share",
        "disp_norm_sil", "disp_norm_ppo", "disp_share", "buffer_generic_frac",
    }  # fmt: skip
    assert expected <= set(s) and all(math.isfinite(v) for v in s.values())
    assert 0.0 < s["grad_share"] < 1.0 and s["grad_share"] == pytest.approx(
        s["grad_norm_pi_sil"] / (s["grad_norm_pi_sil"] + s["grad_norm_pi_ppo"])
    )
    assert 0.0 < s["disp_share"] < 1.0 and s["disp_share"] == pytest.approx(
        s["disp_norm_sil"] / (s["disp_norm_sil"] + s["disp_norm_ppo"])
    )
    assert s["frac_pos"] > 0.0 and s["adv_pos_max"] >= s["adv_pos_mean"] > 0.0 and s["ppo_adv_abs_raw"] > 0.0
    assert s["buffer_bytes"] > 0 and s["coef"] == pytest.approx(0.1)


def test_mode_joint_ajoute_la_perte_aux_minibatchs_ppo_sans_optimiseur() -> None:
    on, off, batch = _learner(scfg()), _learner(None), _one_batch()
    _filled(on, 1.0)
    assert on.sil is not None and on.sil.optimizer is None
    s = on.update(batch)
    off.update(batch)
    assert s.sil["steps_done"] == on.cfg.epochs * s.n_minibatches  # un minibatch SIL par pas PPO
    assert 0.0 < s.sil["disp_share"] < 1.0 and not _same(_params(on), _params(off))


def test_reprise_exacte_de_l_etat_sil_par_le_checkpoint() -> None:
    # Arrange : 5 lots figés ; learner A : 3 mises à jour, état passé par encode/decode (comme un checkpoint) vers un
    # learner B neuf ; puis les 2 mêmes lots pour A et B
    torch.manual_seed(0)
    col = Collector(Corridor(16, horizon=3), PolicyNet(7, 3, (16,)), 8, CPU, seed=1)
    batches = [copy.deepcopy(col.collect(0)) for _ in range(5)]
    a = _learner(scfg())  # mode joint : moments fantômes dans l'état
    _filled(a, 1.0)
    for bt in batches[:3]:
        a.update(dataclasses.replace(bt, policy_version=a.version))
    saved = decode(encode(copy.deepcopy(a.state_dict())))
    b = _learner(scfg())
    b.load_state_dict(saved)
    # Act
    tail_a = [a.update(dataclasses.replace(bt, policy_version=a.version)).sil for bt in batches[3:]]
    tail_b = [b.update(dataclasses.replace(bt, policy_version=b.version)).sil for bt in batches[3:]]
    # Assert : mêmes poids, mêmes journaux SIL (pas appliqués), même buffer et mêmes priorités
    assert sum(x["steps_done"] for x in tail_a) > 0
    assert _same(_params(a), _params(b)) and tail_a == tail_b
    assert a.sil is not None and b.sil is not None and a.sil.buffer is not None and b.sil.buffer is not None
    assert a.sil.buffer.size == b.sil.buffer.size and np.array_equal(a.sil.buffer.prio, b.sil.buffer.prio)
    assert a.sil.tracker is not None and b.sil.tracker is not None
    assert np.array_equal(a.sil.tracker.obs, b.sil.tracker.obs) and a.sil.tracker.cursor == b.sil.tracker.cursor
    assert a.sil.ghost_ppo is not None and b.sil.ghost_ppo is not None and a.sil.ghost_sil is not None
    assert b.sil.ghost_sil is not None and _same(a.sil.ghost_ppo, b.sil.ghost_ppo)
    assert _same(a.sil.ghost_sil, b.sil.ghost_sil) and all(x["disp_share"] > 0 for x in tail_a)
    with pytest.raises(ValueError, match="SIL"):
        _learner(None).load_state_dict(a.state_dict())


# ---------------------------------------------------------------------------------------------------- garde C4


def test_garde_part_de_deplacement_alerte_apres_la_patience_et_stop_ks2() -> None:
    cfg = scfg(share_alert_patience=3, share_stop_patience=2, ram_min_available_mb=0.0)
    st = SILGuardState()
    seen = []
    for _ in range(3):
        st, alerts, stop = sil_guard(cfg, st, share=0.6, coef=0.1, ram_available_mb=None, iteration_s=1.0)
        seen.append((bool(alerts), stop))
    assert seen == [(False, None), (False, None), (True, None)]  # 0,6 : alerte à 3, jamais de stop (≤ 0,8)
    st = SILGuardState()
    st, _, stop = sil_guard(cfg, st, share=0.9, coef=0.1, ram_available_mb=None, iteration_s=1.0)
    assert stop is None
    st, _, stop = sil_guard(cfg, st, share=0.9, coef=0.1, ram_available_mb=None, iteration_s=1.0)
    assert stop is not None and stop[0] == "sil_share"
    # c_sil nul : pas de STOP KS2 ; une itération sous le seuil remet le compteur à 0
    _, _, stop = sil_guard(
        cfg, SILGuardState(share_stop_streak=5), share=0.9, coef=0.0, ram_available_mb=None, iteration_s=1
    )
    assert stop is None
    st, _, _ = sil_guard(
        cfg, SILGuardState(share_alert_streak=9), share=0.1, coef=0.1, ram_available_mb=None, iteration_s=1
    )
    assert st.share_alert_streak == 0


def test_garde_ram_ks1_sur_la_duree_cumulee() -> None:
    cfg = scfg(ram_min_available_mb=1000.0, ram_low_max_s=600.0)
    st = SILGuardState()
    st, _, stop = sil_guard(cfg, st, share=0.0, coef=0.1, ram_available_mb=900.0, iteration_s=400.0)
    assert stop is None and st.ram_low_s == 400.0
    st, _, stop = sil_guard(cfg, st, share=0.0, coef=0.1, ram_available_mb=None, iteration_s=400.0)  # pas de mesure
    assert stop is None and st.ram_low_s == 400.0
    _, _, stop = sil_guard(cfg, st, share=0.0, coef=0.1, ram_available_mb=800.0, iteration_s=300.0)
    assert stop is not None and stop[0] == "sil_ram"
    st, _, _ = sil_guard(cfg, st, share=0.0, coef=0.1, ram_available_mb=1500.0, iteration_s=300.0)
    assert st.ram_low_s == 0.0


# ---------------------------------------------------------------------------------------------------- config


def test_section_sil_absente_ne_change_aucun_hash() -> None:
    # Hashs calculés avec le code de main (f2218a6, avant la SIL) : les runs en cours reprennent sur ce code
    expected = {
        "run_4a_std.yaml": "9791371a6e47f14b1fa38b592700d8d3a9c3670a5d1ab3ad04b383a2e28ca479",
        "run_4a_s.yaml": "e6cdbb70e38450eff11d7a751ee4ec4c005b8d88e0866625310d0bf86cecdf2d",
    }
    for name, h in expected.items():
        cfg = RunConfig.from_yaml(ROOT / "configs" / name)
        assert cfg.sil is None and "sil" not in cfg.to_dict() and cfg.trajectory_hash() == h


def test_sil_refusee_en_gco_s_et_sans_annealing_avant_la_fin() -> None:
    sil = {"enabled": True, "anneal_start_timesteps": 0, "anneal_end_timesteps": 100}
    gco_s = {"variant": "GCO-S", "no_touch_timeout_s": 0.5, "max_episode_s": 0.5}
    kick = {"p_kick_start": 1.0, "p_kick_end": 1.0, "anneal_timesteps": 0}
    with pytest.raises(ConfigError, match="interdite"):
        cfg_of(reward=gco_s, initial_states=kick, sil=sil)
    with pytest.raises(ConfigError, match="C2"):
        cfg_of(sil={**sil, "anneal_end_timesteps": BASE["budget"]["total_timesteps"]})
    with pytest.raises(ConfigError, match="inconnues"):
        cfg_of(sil={**sil, "coef_end": 0.0})
    cfg_of(reward=gco_s, initial_states=kick, sil={**sil, "enabled": False})  # désactivée : acceptée partout


# ---------------------------------------------------------------------------------------------------- mémoire, float16


def test_memoire_aux_dimensions_du_run_int_sil_mesuree_en_octets() -> None:
    """Octets des tableaux alloués aux dimensions de `configs/int_sil.yaml` : 4 096 arènes × 2 voitures, pool 20 %
    (7 373 agents appris), obs 109, fenêtre 150 (règle RAM du banc D7, 2026-11-21), buffer 250 000, float16
    (np.zeros : pages non touchées)."""
    cfg = RunConfig.from_yaml(ROOT / "configs" / "int_sil.yaml")
    assert cfg.sil is not None
    n_learn = cfg.batch_learned_steps // cfg.collection.steps_per_batch
    tracker = EpisodeTracker(n_learn, 109, cfg.sil.horizon_steps, cfg.sil.obs_dtype)
    buf = ReplayBuffer(cfg.sil.capacity, 109, cfg.sil.obs_dtype)
    assert n_learn == 7_373
    assert tracker.nbytes == 7_373 * 150 * (109 * 2 + 2) + 7_373 * 8 == 243_367_984  # fenêtre 150 (règle RAM de D7)
    assert buf.nbytes == 250_000 * (109 * 2 + 2 + 4 + 4 + 4 + 1) == 58_250_000  # obs, act, R, prio, itér., origine
    assert tracker.nbytes + buf.nbytes < 0.6e9  # ADR-009 : ≈ 0,6 Go estimé
    del tracker, buf


def test_taille_serialisee_de_l_etat_sil_proche_des_tableaux() -> None:
    sil = make_sil(scfg(capacity=1000, horizon_steps=50, store_float16=True), obs_dim=109)
    b = steps_arrays(60, 40, 109)
    b["rew"][-1], b["term"][-1] = 1.0, True
    ingest(sil, b)
    f = io.BytesIO()
    torch.save(encode(sil.state_dict()), f)
    assert sil.nbytes <= f.tell() <= sil.nbytes * 1.1 + 50_000


def test_parite_float16_des_obs_sur_politique_et_critique() -> None:
    """ADR-009 D7-5 : |Δ log π| max ≤ 1e-2 et |ΔV| max ≤ 1e-3 entre obs float32 et float16, obs du vrai builder sur
    des états physiques tirés (`test_obs.random_state`), réseaux aux dimensions v0 ; politique initiale (tête gain
    0,01) et politique plus piquée (tête gain 1). Mesuré le 2026-11-18 (CPU, seed 0) : 5e-6 et 4,9e-4 ; ΔV 2,7e-4."""
    rng = np.random.default_rng(0)
    obs = build_obs(random_state(rng, 2048, 2), prev_of(rng, 4096))
    x32 = torch.from_numpy(obs)
    x16 = torch.from_numpy(obs.astype(np.float16)).float()
    torch.manual_seed(0)
    policy, value = PolicyNet(109), ValueNet(109)
    sharp = PolicyNet(109)
    init_linear(sharp.net[-1], 1.0)
    with torch.no_grad():
        for pi in (policy, sharp):
            dlp = (torch.log_softmax(pi(x32), -1) - torch.log_softmax(pi(x16), -1)).abs().max()
            assert float(dlp) <= 1e-2
        assert float((value(x32) - value(x16)).abs().max()) <= 1e-3


# ---------------------------------------------------------------------------------------------------- boucle run()


SIL_RUN = {
    "enabled": True,
    "anneal_start_timesteps": 100,
    "anneal_end_timesteps": 160,
    "batch_size": 4,
    "capacity": 64,
    "horizon_steps": 8,
    "ram_min_available_mb": 0.0,
}
GOALS = {(s, a): ("blue", 1) for s in (3, 11, 19) for a in range(4)}


def sil_factory() -> Any:
    """Simulateur factice exact (`test_run`) avec buts ; obs factices réduites (× 1e-3) pour que V(s) initial reste
    sous R = γ^k ≤ 1 (obs brutes : V > R partout, aucun pas SIL appliqué)."""

    def factory(cfg: RunConfig) -> Any:
        sim = StatefulFakeSim(cfg.env.n_envs, cfg.env.n_cars)
        sim.goals = dict(GOALS)
        return make_env(cfg, sim, build_obs=lambda st, prev: fake_obs(st, prev) * np.float32(1e-3), obs_dim=7)

    return factory


def _weights(out: Path) -> dict[str, Any]:
    found = CheckpointManager(out / "checkpoints").load_latest()
    assert found is not None
    return found[1]["trainer"]["learner"]


def test_run_sil_desactivee_identique_a_sans_section(tmp_path: Path) -> None:
    calls: list[int] = []

    def probe() -> tuple[float, float]:
        calls.append(1)
        return 1.0, 1.0

    fac = make_factory(goals=GOALS)
    go(cfg_of(), tmp_path / "a", env_factory=fac)
    go(
        cfg_of(sil={**SIL_RUN, "enabled": False}),
        tmp_path / "b",
        env_factory=make_factory(goals=GOALS),
        memory_probe=probe,
    )
    wa, wb = _weights(tmp_path / "a"), _weights(tmp_path / "b")
    for net in ("policy", "value"):
        assert all(torch.equal(v, wb[net][k]) for k, v in wa[net].items())
    strip = ("t", "perf/agent_steps_per_s")
    ra = [{k: v for k, v in r.items() if k not in strip} for r in rows(tmp_path / "a")]
    rb = [{k: v for k, v in r.items() if k not in strip} for r in rows(tmp_path / "b")]
    assert ra == rb and not calls and not any(k.startswith("sil/") for k in rb[-1])


def test_run_sil_journal_annealing_et_reprise_exacte(tmp_path: Path) -> None:
    # Arrange : SIL activée, 6 itérations de 32 appris (annealing de 100 à 160 timesteps appris)
    cfg = cfg_of(sil=SIL_RUN)
    probe = lambda: (4096.0, 512.0)  # noqa: E731
    a, b = tmp_path / "a", tmp_path / "b"
    go(cfg, a, env_factory=sil_factory(), memory_probe=probe)
    go(cfg, b, env_factory=sil_factory(), memory_probe=probe, max_iterations=3)
    go(cfg, b, env_factory=sil_factory(), memory_probe=probe, resume=True)
    # Assert : journal C4 à chaque itération, RAM journalisée, SIL active puis libérée ; reprise = run continu
    ra = rows(a)
    assert all("sil/disp_share" in r and r["sil/ram_available_mb"] == 4096.0 for r in ra)
    # mise à jour k faite après 32·k appris : 0, 32, 64, 96 (< 100), 128 (rampe), 160 (= fin : 0, libérée)
    assert [r["sil/coef"] for r in ra] == pytest.approx([0.1, 0.1, 0.1, 0.1, 0.1 * (160 - 128) / 60, 0.0])
    assert all(r["sil/steps_done"] == 2.0 for r in ra[:5]) and ra[-1]["sil/released"] == 1.0
    strip = ("t", "perf/agent_steps_per_s")
    assert [{k: v for k, v in r.items() if k not in strip} for r in ra] == [
        {k: v for k, v in r.items() if k not in strip} for r in rows(b)
    ]
    wa, wb = _weights(a), _weights(b)
    for net in ("policy", "value"):
        assert all(torch.equal(v, wb[net][k]) for k, v in wa[net].items())


def test_run_stop_ks1_si_la_ram_disponible_reste_basse(tmp_path: Path) -> None:
    cfg = cfg_of(sil={**SIL_RUN, "ram_min_available_mb": 1000.0, "ram_low_max_s": 0.0})
    s = go(cfg, tmp_path / "r", memory_probe=lambda: (500.0, 100.0))
    assert s.status == "stopped" and s.cause is not None and s.cause.startswith("sil_ram")
    assert s.end_iteration == 1
    assert any(e.get("event") == "guard" and e.get("rule") == "sil_ram" for e in events(tmp_path / "r"))


def test_run_alerte_part_de_deplacement_journalisee(tmp_path: Path) -> None:
    cfg = cfg_of(sil={**SIL_RUN, "share_alert": 1.0e-9, "share_alert_patience": 1})
    go(cfg, tmp_path / "r", env_factory=sil_factory(), memory_probe=lambda: (4096.0, 1.0))
    assert any(e.get("event") == "alert" and e.get("rule") == "sil_share" for e in events(tmp_path / "r"))


def test_learner_refuse_un_optimiseur_absent_en_mode_separate() -> None:
    with pytest.raises(ValueError):
        SelfImitation(sep(), GAMMA, 3, 5, seed=0, optimizer=None)
    with pytest.raises(ValueError):
        SelfImitation(scfg(), GAMMA, 3, 5, seed=0, optimizer=_ManualAdam([], 1e-3))


# ---------------------------------------------------------------------------------------------------- ADR-009 amendé


def test_mode_joint_par_defaut_sans_optimiseur_sil_et_options_separate_refusees() -> None:
    """ADR-009 D7-9 : en mode joint, aucun optimiseur SIL n'est créé ; `updates_per_iteration` et `max_grad_norm`
    sont refusés ; le mode separate les exige."""
    assert SILConfig(anneal_start_timesteps=0, anneal_end_timesteps=1).mode == "joint"
    learner = _learner(scfg())
    assert learner.sil is not None and learner.sil.optimizer is None
    for bad in ({"updates_per_iteration": 2}, {"max_grad_norm": 0.5}):
        with pytest.raises(ValueError, match="mode joint"):
            scfg(**bad)
    with pytest.raises(ValueError, match="mode separate"):
        scfg(mode="separate")
    with pytest.raises(ConfigError, match="mode joint"):
        cfg_of(sil={**SIL_RUN, "updates_per_iteration": 2})
    cfg = RunConfig.from_yaml(ROOT / "configs" / "int_sil.yaml").sil
    assert cfg is not None and cfg.mode == "joint" and cfg.batch_size == 4_096
    assert cfg.updates_per_iteration is None and cfg.max_grad_norm is None


def _delta(before: list[Any], learner: Learner) -> list[Any]:
    return [p.detach() - b for p, b in zip(learner.policy.parameters(), before, strict=True)]


def test_decomposition_du_deplacement_exacte_adam_manuel() -> None:
    """ADR-009 D7-8 : Σ_k Δθ^ppo_k + Σ_k Δθ^sil_k = Δθ réel sur π, à 1e-5 relatif près, sur 3 mises à jour de suite
    (moments fantômes continus d'une itération à l'autre) ; Adam manuel de `test_ppo` (même formule que torch,
    vérifiée en intégration sur `torch.optim.Adam` fusionné et non fusionné : `tests/integration/test_sil_adam.py`)."""
    # Arrange
    on = _learner(scfg())
    _filled(on, 1.0)
    torch.manual_seed(0)
    col = Collector(Corridor(16, horizon=3), PolicyNet(7, 3, (16,)), 8, CPU, seed=1)
    for _ in range(3):
        before = [p.detach().clone() for p in on.policy.parameters()]
        # Act
        full = on.update(dataclasses.replace(copy.deepcopy(col.collect(0)), policy_version=on.version))
        s = full.sil
        # Assert
        assert on.sil is not None and on.sil.last_disp is not None
        d_ppo, d_sil = on.sil.last_disp
        real = _delta(before, on)
        err = math.sqrt(sum(float((a + b - r).pow(2).sum()) for a, b, r in zip(d_ppo, d_sil, real, strict=True)))
        norm = math.sqrt(sum(float(r.pow(2).sum()) for r in real))
        assert norm > 0 and err / norm <= 1e-5
        assert s["steps_done"] == on.cfg.epochs * full.n_minibatches and 0.0 < s["disp_share"] < 1.0


def test_coefficient_nul_part_de_deplacement_nulle_et_aucun_calcul() -> None:
    on, off, batch = _learner(scfg(coef=0.0)), _learner(None), _one_batch()
    _filled(on, 1.0)
    s = on.update(batch).sil
    off.update(batch)
    assert (s["disp_share"], s["grad_share"], s["steps_done"], s["grad_norm_pi_ppo"]) == (0.0, 0.0, 0.0, 0.0)
    assert on.sil is not None and on.sil.ghost_ppo is None and _same(_params(on), _params(off))


def test_origine_stockee_sur_un_octet_et_jamais_lue_par_le_tirage_ni_la_perte() -> None:
    """ADR-009 D7-6 : l'origine est stockée (diagnostic `buffer_generic_frac`) ; permuter le champ origine ne change
    ni les indices tirés, ni les poids, ni la perte."""
    # Arrange : agent 0 marque après un état générique (t = 2), agent 1 après un kickoff (t = 3)
    sil, b = make_sil(), steps_arrays(4, 2)
    b["rew"][2, 0], b["term"][2, 0], b["rew"][3, 1], b["term"][3, 1] = 1.0, True, 1.0, True
    generic = np.zeros((4, 2), np.bool_)
    generic[2, 0] = True
    sil.ingest(b["obs"], b["act"], b["rew"], b["term"], b["trunc"], np.ones(2, np.bool_), 0, generic)
    buf = sil.buffer
    assert buf is not None and buf.origin.dtype == np.uint8 and buf.origin.itemsize == 1
    assert buf.origin[: buf.size].tolist() == [1, 1, 1, 0, 0, 0, 0] and buf.generic_frac() == pytest.approx(3 / 7)
    other = make_sil()
    other.load_state_dict(decode(encode(sil.state_dict())))
    assert other.buffer is not None
    other.buffer.origin[: other.buffer.size] = 1 - buf.origin[: buf.size]  # origine permutée
    policy, value = PolicyNet(3, 5, (8,)), ValueNet(3, (8,))
    out = []
    for s_ in (sil, other):
        assert s_.buffer is not None
        idx, w = s_.buffer.sample(64, s_.generator, 0.1)
        obs, act, ret = s_.buffer.batch(idx, CPU)
        pl, vl, _ = sil_losses(policy, value, obs, act, ret, w)
        out.append((idx, w, float(pl.detach()), float(vl.detach())))
    assert torch.equal(out[0][0], out[1][0]) and torch.equal(out[0][1], out[1][1]) and out[0][2:] == out[1][2:]


def test_origine_des_episodes_par_agent_depuis_info_et_dans_le_lot() -> None:
    from training.rollout import _agent_generic

    info = {"ended": np.array([True, True, False]), "end_generic": np.array([True, False, True])}
    assert _agent_generic(info, 6).tolist() == [True, True, False, False, False, False]
    assert _agent_generic({}, 6) is None
    with pytest.raises(ValueError):
        _agent_generic(info, 5)
    # le lot collecté porte le champ (faux sans les clés de l'env, comme Corridor)
    batch = _one_batch()
    assert batch.ep_generic is not None and batch.ep_generic.shape == batch.rew.shape and not batch.ep_generic.any()


def test_restauration_k4_recharge_l_etat_de_la_garde_sil(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Régression (revue de 414b83f, m2) : après une restauration K4 en cours de processus, l'état de la garde SIL
    restait celui de la trajectoire écartée. Le compteur d'alerte final doit compter les seules itérations de la
    trajectoire retenue."""
    # Arrange : coefficient constant sur tout le run, seuil d'alerte quasi nul, NaN au 5e appel (restauration vers 4)
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, grad_norm=float("nan")) if n == 5 else st)
    sil = {**SIL_RUN, "anneal_start_timesteps": 190, "anneal_end_timesteps": 191, "share_alert": 1e-9}
    out = tmp_path / "r"
    # Act
    s = go(cfg_of(sil=sil), out, env_factory=sil_factory(), memory_probe=lambda: (4096.0, 1.0))
    # Assert
    assert s.status == "completed" and s.restores == 1
    shares = [r["sil/disp_share"] for r in rows(out)]
    expected = 0
    for x in shares:
        expected = expected + 1 if x > 1e-9 else 0
    assert expected >= 2
    final = CheckpointManager(out / "checkpoints").load_latest()
    assert final is not None and final[1]["iteration"] == 6
    assert final[1]["extra"]["sil_guard"]["share_alert_streak"] == expected
