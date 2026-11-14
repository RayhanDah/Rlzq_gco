"""Pool d'adversaires (A3, ADR-005 D5) : répartition 80/20, scores de qualité Five, éviction, collecte à deux
politiques, exclusion du camp gelé de la perte, état dans le checkpoint et reprise exacte. Tout en CPU, sim factice."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_checkpoint import _same  # noqa: E402
from test_ppo import _collect_one  # noqa: E402
from test_run import cfg_of, events, go, inject, make_factory, rows  # noqa: E402

from models.policy import PolicyNet  # noqa: E402
from training import CheckpointManager, Collector, ConfigError  # noqa: E402
from training.selfplay import PastOpponents, QualityPool, make_layout, n_past_arenas  # noqa: E402

CPU = torch.device("cpu")
POOL = {"past_fraction": 0.5, "snapshot_every": 2, "max_size": 2, "quality_lr": 0.01}
# 4 arènes, 2 « passé » (0 : bleu gelé ; 2 : orange gelé) ; 24 transitions apprises par lot de 32 ; 192 -> 8 itérations


# ---------------------------------------------------------------------------------------------------- répartition


@pytest.mark.parametrize("n_envs", [5, 64, 1000, 4096])
def test_proportion_d_arenes_passe_exacte_et_camps_equilibres(n_envs: int) -> None:
    # Act
    lay = make_layout(n_envs, 2, 0.2)
    # Assert : partition fixe => l'écart à 20 % n'est que l'arrondi, borné par 1/(2·n_envs) (4 096 : 819, 19,995 %)
    assert abs(lay.n_past / n_envs - 0.2) <= 0.5 / n_envs
    assert np.unique(lay.arenas).size == lay.n_past and lay.arenas.min() >= 0 and lay.arenas.max() < n_envs
    assert abs(int((lay.frozen_team == 0).sum()) - int((lay.frozen_team == 1).sum())) <= 1  # bleu / orange
    assert lay.n_learned_agents == 2 * n_envs - lay.n_past
    assert not lay.learn_mask[lay.frozen_agents.reshape(-1)].any()


def test_arenes_passe_reparties_sur_les_indices() -> None:
    lay = make_layout(4096, 2, 0.2)
    assert lay.n_past == 819
    gaps = np.diff(lay.arenas)
    assert gaps.min() >= 4 and gaps.max() <= 6  # régulièrement espacées (4096 / 819 ≈ 5)


def test_2v2_gele_un_camp_entier() -> None:
    lay = make_layout(4, 4, 0.5)  # arènes 0 (bleu gelé : slots 0, 1) et 2 (orange gelé : slots 2, 3)
    assert lay.frozen_agents.tolist() == [[0, 1], [10, 11]]
    assert lay.learn_mask.sum() == 16 - 4


def test_fraction_sans_arene_passe_refusee() -> None:
    with pytest.raises(ValueError):
        make_layout(2, 2, 0.2)  # 0,4 -> 0 arène
    assert n_past_arenas(4096, 0.2) == 819


# ---------------------------------------------------------------------------------------------------- scores Five


def test_nouvel_instantane_entre_a_q_max_et_victoire_applique_la_regle_five() -> None:
    # Arrange
    pool = QualityPool(max_size=10, quality_lr=0.01, seed=0)
    a, _ = pool.add(0)
    b, _ = pool.add(50)
    assert pool.q == {a: 0.0, b: 0.0}
    # Act : victoire contre a ; p_a = 1/2, N = 2 => q_a = -0,01 / (2 · 0,5) = -0,01
    pool.record_win(a)
    c, _ = pool.add(100)
    # Assert
    assert pool.q[a] == pytest.approx(-0.01) and pool.q[b] == 0.0 and pool.q[c] == 0.0  # c entre à max(q) = 0
    p = pool.probs()
    expected = np.exp([-0.01, 0.0, 0.0]) / np.exp([-0.01, 0.0, 0.0]).sum()
    assert np.allclose(p, expected, rtol=0, atol=1e-15)
    # victoire contre b : q_b -= η / (N · p_b), p_b courant
    pool.record_win(b)
    assert pool.q[b] == pytest.approx(-0.01 / (3 * expected[1]))


def test_eviction_du_plus_mauvais_q_puis_du_plus_ancien_a_egalite() -> None:
    pool = QualityPool(max_size=3, quality_lr=0.5, seed=0)
    u0, _ = pool.add(0)
    u1, _ = pool.add(1)
    u2, _ = pool.add(2)
    pool.record_win(u1)  # u1 devient le plus mauvais
    u3, ev = pool.add(3)
    assert ev == u1 and pool.uids == [u0, u2, u3]
    _, ev = pool.add(4)  # q égaux (0) entre u0, u2, u3, et le nouveau : le plus ancien sort
    assert ev == u0 and len(pool) == 3
    assert pool.record_win(u1) is False  # évincé : sans effet


def test_tirage_suit_p_proportionnel_a_exp_q() -> None:
    # Arrange : q = (0, -1, -2) => p ≈ (0,665 ; 0,245 ; 0,090)
    pool = QualityPool(max_size=10, quality_lr=0.01, seed=3)
    for i in range(3):
        pool.add(i)
    pool.q = {0: 0.0, 1: -1.0, 2: -2.0}
    n = 20_000
    # Act
    draws = pool.sample(n)
    # Assert : tolérance = 5 écarts types binomiaux par catégorie (fausse alerte < 1e-6 ; tirage seedé de toute façon)
    p = pool.probs()
    for k in range(3):
        freq = float((draws == k).sum()) / n
        assert abs(freq - p[k]) <= 5 * math.sqrt(p[k] * (1 - p[k]) / n), (k, freq, p[k])


def test_etat_du_pool_aller_retour_donne_les_memes_tirages() -> None:
    a = QualityPool(5, 0.01, seed=1)
    for i in range(4):
        a.add(i)
    a.record_win(2)
    a.sample(7)
    b = QualityPool(5, 0.01, seed=99)
    b.load_state_dict(a.state_dict())
    assert np.array_equal(a.sample(50), b.sample(50)) and a.q == b.q and a.uids == b.uids


# ---------------------------------------------------------------------------------------------------- collecte


def _biased(action: int) -> PolicyNet:
    """Politique qui joue `action` avec probabilité ≈ 1 (biais de la tête)."""
    net = PolicyNet(7, 90, (8,))
    with torch.no_grad():
        net.net[-1].bias.fill_(-50.0)
        net.net[-1].bias[action] = 50.0
    return net


def _pool_collector(goals: dict[tuple[int, int], tuple[str, int]] | None = None) -> tuple[Collector, PastOpponents]:
    cfg = cfg_of(pool=POOL)
    env = make_factory(goals)(cfg)
    opp = PastOpponents(make_layout(4, 2, 0.5), QualityPool(2, 0.01, seed=0), lambda: PolicyNet(7, 90, (8,)), CPU)
    return Collector(env, _biased(3), 4, CPU, seed=0, opponents=opp), opp


def test_camp_gele_joue_l_instantane_et_masque_exact() -> None:
    # Arrange : courant joue 3, instantané joue 5
    col, opp = _pool_collector()
    opp.add_snapshot(_biased(5), iteration=0)
    # Act
    batch = col.collect(0)
    # Assert : agents gelés (0 et 5) = instantané ; les autres = courant ; masque = celui de la répartition
    frozen = opp.layout.frozen_agents.reshape(-1)
    assert frozen.tolist() == [0, 5]
    assert (batch.act[:, frozen] == 5).all()
    assert (batch.act[:, np.setdiff1d(np.arange(8), frozen)] == 3).all()
    assert batch.learn_mask.tolist() == opp.layout.learn_mask.tolist()
    assert col.learnable_steps == 4 * 6 and col.agent_steps == 4 * 8  # 75 % appris ici (2 arènes passé sur 4)


def test_sans_pool_la_collecte_est_inchangee() -> None:
    cfg = cfg_of()
    a = Collector(make_factory()(cfg), _biased(3), 4, CPU, seed=0).collect(0)
    assert a.learn_mask.all()
    with pytest.raises(ValueError):
        _, opp = _pool_collector()
        Collector(make_factory()(cfg_of(pool=POOL)), _biased(3), 4, CPU, 0, learn_mask=a.learn_mask, opponents=opp)


def test_pool_vide_refuse() -> None:
    col, _ = _pool_collector()
    with pytest.raises(RuntimeError, match="pool vide"):
        col.collect(0)


def test_victoire_contre_le_passe_met_q_a_jour_et_reaffecte() -> None:
    # Arrange : arène 0 (bleu gelé) : orange marque au step 1 => la politique courante gagne ; arène 2 (orange gelé) :
    # orange marque au step 2 => défaite
    col, opp = _pool_collector({(1, 0): ("orange", 1), (2, 2): ("orange", 1)})
    u0 = opp.add_snapshot(_biased(5), 0)
    u1 = opp.add_snapshot(_biased(6), 1)
    # Act
    col.collect(0)
    # Assert
    assert opp.wins == 1 and opp.losses == 1
    pool = opp.pool
    assert sorted(pool.q.values()) == pytest.approx([-0.01 / (2 * 0.5), 0.0])  # une seule victoire, p = 1/2, N = 2
    assert {u0, u1} == set(pool.uids)


def test_instantane_evince_en_jeu_finit_son_episode_puis_est_libere() -> None:
    # Arrange : pool de 1 ; les deux arènes jouent u0 ; u1 l'évince en cours d'épisode
    col, opp = _pool_collector()
    opp.pool.max_size = 1
    u0 = opp.add_snapshot(_biased(5), 0)
    col.collect(0)  # affectation : u0 partout ; 4 steps, épisodes de 8 steps (64 ticks >= 60) : pas finis
    # Act
    u1 = opp.add_snapshot(_biased(6), 1)
    evicted_but_kept = opp.pool.uids == [u1] and u0 in opp.nets
    batch = col.collect(1)  # steps 5 à 8 : l'épisode finit au 8e
    # Assert : u0 a joué jusqu'à la fin de son épisode (poids conservés), puis réaffectation à u1 et libération
    assert evicted_but_kept
    frozen = opp.layout.frozen_agents.reshape(-1)
    assert (batch.act[:, frozen] == 5).all()
    assert set(opp.assignment.tolist()) == {u1} and u0 not in opp.nets
    batch = col.collect(2)
    assert (batch.act[:, frozen] == 6).all()


# ---------------------------------------------------------------------------------------------------- perte


def test_aucune_transition_du_camp_gele_n_entre_dans_la_perte() -> None:
    # Arrange : un lot masqué (agents impairs gelés) et le même lot dont TOUTES les données des agents gelés
    # (obs, actions, log-probs, récompenses, obs finales) sont remplacées par du bruit
    mask = np.zeros(16, np.bool_)
    mask[::2] = True
    la, batch = _collect_one(mask=mask)
    lb, _ = _collect_one(mask=mask)
    rng = np.random.default_rng(0)
    frozen = np.flatnonzero(~mask)
    obs, act, logp, rew = batch.obs.copy(), batch.act.copy(), batch.logp.copy(), batch.rew.copy()
    obs[:, frozen] = rng.normal(size=obs[:, frozen].shape).astype(np.float32)
    act[:, frozen] = rng.integers(0, 3, size=act[:, frozen].shape)
    logp[:, frozen] = rng.normal(size=logp[:, frozen].shape).astype(np.float32)
    rew[:, frozen] = rng.normal(size=rew[:, frozen].shape).astype(np.float32)
    final_obs = batch.final_obs.copy()
    hit = np.isin(batch.final_agent, frozen)
    final_obs[hit] = 1e3
    last_obs = batch.last_obs.copy()
    last_obs[frozen] = -1e3
    noisy = dataclasses.replace(batch, obs=obs, act=act, logp=logp, rew=rew, final_obs=final_obs, last_obs=last_obs)
    # Act
    sa, sb = la.update(batch), lb.update(noisy)
    # Assert : mises à jour identiques bit à bit (gradient nul vis-à-vis des transitions gelées) ; n_samples = appris
    assert sa.n_samples == sb.n_samples == batch.n_steps * 8
    for pa, pb in zip(la.params, lb.params, strict=True):
        assert torch.equal(pa, pb)
    assert sa.policy_loss == sb.policy_loss and sa.value_loss == sb.value_loss


def test_le_bruit_change_la_mise_a_jour_sans_masque() -> None:
    # Témoin du test précédent : sans masque, le même bruit change bien les poids (le test n'est pas vide)
    la, batch = _collect_one()
    lb, _ = _collect_one()
    act = batch.act.copy()
    act[:, 1::2] = (act[:, 1::2] + 1) % 3
    lb.update(dataclasses.replace(batch, act=act))
    la.update(batch)
    assert any(not torch.equal(pa, pb) for pa, pb in zip(la.params, lb.params, strict=True))


# ---------------------------------------------------------------------------------------------------- config


def test_budget_en_timesteps_appris() -> None:
    cfg = cfg_of(pool=POOL)
    assert cfg.batch_agent_steps == 32 and cfg.batch_learned_steps == 24 and cfg.n_past_arenas == 2
    assert cfg.total_iterations == 8 and cfg.ppo.total_iterations == 8  # ⌈192 / 24⌉
    assert cfg.anneal_iterations == 8  # annealing de p_kick en timesteps appris (D4)
    big = cfg_of(pool={**POOL, "past_fraction": 0.2}, env={"n_envs": 4096})
    assert big.batch_learned_steps == (8192 - 819) * 4 and big.n_past_arenas == 819


def test_hash_inchange_sans_pool_et_sensible_au_pool() -> None:
    base = cfg_of()
    assert "pool" not in base.to_dict()
    assert cfg_of(pool=POOL).trajectory_hash() != base.trajectory_hash()
    assert cfg_of(pool={**POOL, "quality_lr": 0.02}).trajectory_hash() != cfg_of(pool=POOL).trajectory_hash()
    with pytest.raises(ConfigError):
        cfg_of(pool={**POOL, "past_fraction": 0.05})  # 0,2 arène sur 4 -> 0
    with pytest.raises(ConfigError):
        cfg_of(pool={**POOL, "inconnu": 1})


# ---------------------------------------------------------------------------------------------------- run


GOALS = {(1, 0): ("orange", 1), (9, 2): ("blue", 1), (17, 0): ("orange", 1), (30, 2): ("orange", 1)}


def _pool_state(out: Path) -> dict[str, Any]:
    found = CheckpointManager(out / "checkpoints").load_latest()
    assert found is not None
    st: dict[str, Any] = found[1]["trainer"]["collector"]["opponents"]
    return st


def test_run_avec_pool_instantanes_eviction_et_compteur_appris(tmp_path: Path) -> None:
    # Act
    out = tmp_path / "run"
    s = go(cfg_of(pool=POOL), out, env_factory=make_factory(GOALS))
    # Assert : θ₀ + instantanés aux itérations 2, 4, 6, 8 ; plafond 2
    st = _pool_state(out)
    assert s.status == "completed" and s.end_iteration == 8
    assert s.trained_timesteps == 8 * 24 and s.collected_timesteps == 9 * 32  # appris ≠ collecté
    assert st["pool"]["next_uid"] == 5 and len(st["pool"]["uids"]) == 2 and st["pool"]["iteration"][-1] == 8
    wins, losses, draws = st["counts"]
    assert wins >= 1 and losses >= 1 and draws >= 1  # victoires (q mis à jour), défaites et nuls contre le passé
    r = rows(out)
    assert r[-1]["train/trained_timesteps"] == 192 and r[-1]["train/collected_timesteps"] == 9 * 32
    assert r[-1]["pool/size"] == 2.0


def test_reprise_exacte_avec_pool(tmp_path: Path) -> None:
    # Arrange : référence continue
    cfg = cfg_of(pool=POOL)
    a, b = tmp_path / "a", tmp_path / "b"
    go(cfg, a, env_factory=make_factory(GOALS))
    # Act : arrêt à l'itération 3 (instantané à 2 déjà pris), reprise dans de nouveaux objets
    s1 = go(cfg, b, max_iterations=3, env_factory=make_factory(GOALS))
    s2 = go(cfg, b, resume=True, env_factory=make_factory(GOALS))
    # Assert : même journal, mêmes poids, même pool (poids des instantanés, q, tirage, affectation)
    assert s1.status == "paused" and s2.status == "completed" and s2.exact_env is True
    drop = ("t", "perf/agent_steps_per_s")
    assert [{k: v for k, v in x.items() if k not in drop} for x in rows(a)] == [
        {k: v for k, v in x.items() if k not in drop} for x in rows(b)
    ]
    la, lb = CheckpointManager(a / "checkpoints").load_latest(), CheckpointManager(b / "checkpoints").load_latest()
    assert la is not None and lb is not None
    assert _same(la[1]["trainer"]["learner"]["policy"], lb[1]["trainer"]["learner"]["policy"])
    assert _same(_pool_state(a), _pool_state(b))


def test_restauration_restaure_le_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : checkpoints toutes les 4 itérations, instantanés toutes les 2 ; NaN à la 7e mise à jour => retour à
    # l'itération 4 alors que l'instantané de l'itération 6 est déjà dans le pool
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, policy_loss=float("nan")) if n == 7 else st)
    out = tmp_path / "run"
    cfg = cfg_of(pool={**POOL, "max_size": 10}, checkpoint={"every_iterations": 4, "keep": 3})
    # Act
    s = go(cfg, out, env_factory=make_factory(GOALS))
    # Assert : le pool revient à son état de l'itération 4 (sans l'instantané 6 écarté), puis 6 et 8 sont rejoués ;
    # sans restauration du pool : instantanés 0, 2, 4, 6, 6, 8 et next_uid = 6
    assert s.restores == 1 and s.status == "completed"
    ev = [e for e in events(out) if e.get("event") == "restore"]
    assert ev and ev[0]["to_iteration"] == 4
    st = _pool_state(out)
    assert st["pool"]["iteration"] == [0, 2, 4, 6, 8] and st["pool"]["next_uid"] == 5


# ---------------------------------------------------------------------------------------------------- constats de revue


def test_victoire_contre_un_instantane_de_probabilite_nulle_ne_plante_pas() -> None:
    # Arrange : q si bas que exp(q − max q) vaut exactement 0,0 en float64
    pool = QualityPool(max_size=10, quality_lr=0.01, seed=0)
    a, _ = pool.add(0)
    pool.add(50)
    pool.q[a] = -1000.0
    assert pool.probs()[0] == 0.0
    # Act : sans plancher sur p, ZeroDivisionError (le run mourrait)
    assert pool.record_win(a) is True
    # Assert : q reste fini et ne remonte pas
    assert math.isfinite(pool.q[a]) and pool.q[a] < -1000.0


def test_pas_d_instantane_sur_une_iteration_non_saine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : KL au-dessus du seuil à la 2e mise à jour seulement : CONTINUE, non saine, sans restauration
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, approx_kl=1.0) if n == 2 else st)
    out = tmp_path / "run"
    # Act
    s = go(cfg_of(pool={**POOL, "max_size": 10}), out, env_factory=make_factory(GOALS))
    # Assert : pas d'instantané à l'itération 2 (il entrerait à q = max) ; 0, 4, 6, 8 seulement
    assert s.status == "completed" and s.restores == 0
    assert _pool_state(out)["pool"]["iteration"] == [0, 4, 6, 8]


@pytest.mark.parametrize("n_cars", [2, 4])
def test_inference_groupee_egale_un_passage_par_instantane(n_cars: int) -> None:
    # Arrange : 5 instantanés aléatoires distincts, 16 arènes « passé », groupes de tailles inégales (dont un vide)
    torch.manual_seed(0)
    layout = make_layout(32, n_cars, 0.5)
    opp = PastOpponents(layout, QualityPool(5, 0.01, seed=0), lambda: PolicyNet(7, 90, (8, 8)), CPU)
    uids = [opp.add_snapshot(PolicyNet(7, 90, (8, 8)), i) for i in range(5)]
    opp.assignment[:] = np.array([uids[i] for i in [4, 0, 0, 2, 4, 4, 1, 0, 2, 4, 4, 0, 1, 4, 0, 4]])  # uids[3] absent
    obs = torch.randn(32 * n_cars, 7)
    logits = torch.randn(32 * n_cars, 90)
    # Act
    out = opp.override_logits(obs, logits.clone())
    # Assert : chaque agent gelé reçoit les logits de SON instantané ; les autres lignes sont intactes
    frozen = layout.frozen_agents
    for j in range(layout.n_past):
        want = opp.policy_of(int(opp.assignment[j]))(obs[frozen[j]])
        # tolérance : baddbmm (groupé) et addmm (par réseau) n'additionnent pas dans le même ordre (float32)
        assert torch.allclose(out[frozen[j]], want, atol=1e-5, rtol=1e-5)
    others = np.setdiff1d(np.arange(32 * n_cars), frozen.reshape(-1))
    assert torch.equal(out[others], logits[others])


def test_empilement_suit_les_ajouts_evictions_et_rechargements() -> None:
    col, opp = _pool_collector()
    opp.pool.max_size = 1
    opp.add_snapshot(_biased(5), 0)
    col.collect(0)
    assert opp._stacked().uids == sorted(opp.nets)
    opp.add_snapshot(_biased(6), 1)  # u0 évincé mais encore en jeu : les deux réseaux sont empilés
    assert opp._stacked().uids == [0, 1]
    twin_col, twin = _pool_collector()
    twin.pool.max_size = 1
    twin.load_state_dict(opp.state_dict())
    assert twin._stacked().uids == [0, 1]
    a, b = col.collect(1), twin_col.collect(1)
    frozen = opp.layout.frozen_agents.reshape(-1)
    assert np.array_equal(a.act[:, frozen], b.act[:, frozen])


def test_rechargement_remplace_l_empilement_meme_a_uid_identiques() -> None:
    # Arrange : après une restauration, un uid peut être réattribué à d'autres poids (next_uid recule)
    col, opp = _pool_collector()
    opp.add_snapshot(_biased(5), 0)
    col.collect(0)  # empilement construit avec « joue 5 »
    _, other = _pool_collector()
    other.add_snapshot(_biased(6), 0)  # même uid 0, poids différents
    # Act
    opp.load_state_dict(other.state_dict())
    batch = col.collect(1)
    # Assert
    assert (batch.act[:, opp.layout.frozen_agents.reshape(-1)] == 6).all()
