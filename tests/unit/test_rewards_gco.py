"""Tests de la récompense GCO : `docs/GCO_SPEC.md` §7, tests 1 à 7 (le 8 est un processus : ADR + spec).

Non couvert ici : la moitié « observations miroir identiques » du §7.5 (lot obs, `src/obs/`) et l'écriture effective du
hash dans un checkpoint (§7.7, lot checkpoints) ; seule la production d'un hash stable est testée.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from dataclasses import asdict
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rewards import (
    PHYSICS_TICK_RATE_HZ,
    GoalNet,
    RewardConfig,
    RewardFunction,
    RewardVariant,
    Team,
    episode_end,
    gco_team_rewards,
    make_reward,
    rules_for,
    scorer_only_rewards,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261110
TICK_SKIP = 8  # §2.1 : 15 Hz de décision
GCO_VARIANTS = (RewardVariant.GCO_S, RewardVariant.GCO_STD, RewardVariant.GCO_EXT)
BLU, ORA = Team.BLUE, Team.ORANGE
LAYOUTS = {
    "1v1": [BLU, ORA],
    "2v2": [BLU, BLU, ORA, ORA],
    "3v3": [BLU, BLU, BLU, ORA, ORA, ORA],
    "2v2 entrelacé": [ORA, BLU, ORA, BLU],
}


def _expected(net: GoalNet, layout: list[Team]) -> list[float]:
    """Oracle §2.1 écrit indépendamment, élément par élément."""
    if net is GoalNet.NONE:
        return [0.0] * len(layout)
    return [-1.0 if int(team) == int(net) else 1.0 for team in layout]


# ---------------------------------------------------------------- §7.1 signe par équipe, somme nulle


@pytest.mark.parametrize("variant", GCO_VARIANTS)
@pytest.mark.parametrize("layout_name", list(LAYOUTS))
def test_7_1_but_par_equipe_et_somme_nulle(variant: RewardVariant, layout_name: str) -> None:
    # Arrange
    layout = LAYOUTS[layout_name]
    fn = make_reward({"variant": variant.value})
    # Act
    in_orange = fn.rewards_one(GoalNet.ORANGE, layout)  # bleu marque
    in_blue = fn.rewards_one(GoalNet.BLUE, layout)  # orange marque
    none = fn.rewards_one(GoalNet.NONE, layout)
    # Assert
    assert list(in_orange) == [1.0 if t is BLU else -1.0 for t in layout]
    assert list(in_blue) == [1.0 if t is ORA else -1.0 for t in layout]
    assert list(none) == [0.0] * len(layout)
    assert sum(in_orange) == 0.0 and sum(in_blue) == 0.0


def test_7_1_lot_vectorise_egale_oracle() -> None:
    # Arrange : 4096 arènes 2v2 aux compositions mélangées, filets aléatoires.
    rng = np.random.default_rng(SEED)
    n_envs = 4096
    teams = np.array([rng.permutation([0, 0, 1, 1]) for _ in range(n_envs)])
    nets = rng.integers(-1, 2, size=n_envs)
    # Act
    out = gco_team_rewards(nets, teams)
    # Assert
    assert out.shape == (n_envs, 4) and out.dtype == np.float32
    expected = np.array(
        [_expected(GoalNet(int(n)), [Team(int(t)) for t in row]) for n, row in zip(nets, teams, strict=True)]
    )
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(out.sum(axis=1), 0.0)


# ---------------------------------------------------------------- §7.2 csc : signe selon le filet touché


def test_7_2_csc_signe_selon_le_but_touche_pas_le_toucheur() -> None:
    # Arrange : un bleu pousse la balle dans son propre but. L'entrée GCO ne contient que le filet touché.
    fn = make_reward({"variant": "GCO-std"})
    # Act
    r = fn.rewards_one(GoalNet.BLUE, [BLU, ORA])
    # Assert
    assert r == (-1.0, 1.0)


@pytest.mark.parametrize("variant", GCO_VARIANTS)
def test_7_2_le_buteur_nest_pas_une_entree_gco(variant: RewardVariant) -> None:
    # Arrange
    fn = make_reward({"variant": variant.value})
    # Act / Assert : fournir un buteur (dernier toucheur) à une variante GCO est refusé (choix A10).
    with pytest.raises(ValueError, match="scorer doit être None"):
        fn.rewards_one(GoalNet.BLUE, [BLU, ORA], scorer=0)
    assert list(inspect.signature(gco_team_rewards).parameters) == ["goal_net", "agent_team"]


def test_7_2_scorer_only_csc_ne_credite_personne() -> None:
    # Arrange : ablation hors spec, csc du bleu 0 en 2v2 → aucun orange n'est buteur.
    fn = make_reward({"variant": "scorer-only"})
    # Act
    csc = fn.rewards_one(GoalNet.BLUE, [BLU, BLU, ORA, ORA], scorer=0)
    no_credit = fn.rewards_one(GoalNet.BLUE, [BLU, BLU, ORA, ORA], scorer=-1)
    normal = fn.rewards_one(GoalNet.ORANGE, [BLU, BLU, ORA, ORA], scorer=1)
    # Assert (choix A9) : −1 à toute l'équipe qui encaisse, +1 au seul buteur de l'équipe qui marque.
    assert csc == (-1.0, -1.0, 0.0, 0.0)
    assert no_credit == (-1.0, -1.0, 0.0, 0.0)
    assert normal == (0.0, 1.0, -1.0, -1.0)


def test_scorer_only_exige_un_buteur_valide() -> None:
    fn = make_reward({"variant": "scorer-only"})
    with pytest.raises(ValueError, match="scorer est requis"):
        fn.rewards_one(GoalNet.BLUE, [BLU, ORA])
    with pytest.raises(ValueError, match="scorer hors"):
        scorer_only_rewards([0], [BLU, ORA], [2])


# ---------------------------------------------------------------- §7.3 propriété : 0 hors but (troncature, reset)


@pytest.mark.parametrize("variant", GCO_VARIANTS)
def test_7_3_propriete_aucune_recompense_hors_but(variant: RewardVariant) -> None:
    # Arrange : 512 arènes 2v2 simulées en lot avec reset automatique, paramètres de touche et de but tirés par arène
    # pour couvrir les trois fins d'épisode. Seed fixée.
    rng = np.random.default_rng(SEED)
    cfg = RewardConfig(variant=variant)
    fn = RewardFunction(cfg)
    n_envs, n_steps = 512, 4000
    teams = np.array([rng.permutation([0, 0, 1, 1]) for _ in range(n_envs)])
    p_touch = rng.choice([0.002, 0.05, 0.5], size=n_envs)
    p_goal = rng.choice([1e-4, 2e-3], size=n_envs)
    since_touch = np.zeros(n_envs, dtype=np.int64)
    ep_ticks = np.zeros(n_envs, dtype=np.int64)
    first_step = np.ones(n_envs, dtype=bool)
    ends = {"but": 0, "no-touch": 0, "durée": 0}
    reset_steps = 0
    # Act / Assert à chaque step
    for _ in range(n_steps):
        ep_ticks += TICK_SKIP
        touched = rng.random(n_envs) < p_touch
        since_touch = np.where(touched, 0, since_touch + TICK_SKIP)
        goal = rng.random(n_envs) < p_goal
        nets = np.where(goal, rng.integers(0, 2, size=n_envs), GoalNet.NONE)

        out = fn.step(nets, teams, since_touch, ep_ticks)

        assert np.array_equal(out.rewards != 0, np.broadcast_to(goal[:, None], out.rewards.shape))
        assert np.all(out.rewards.sum(axis=1) == 0.0)
        assert np.all(np.abs(out.rewards[goal]) == 1.0)
        assert np.array_equal(out.terminated, goal)
        assert not np.any(out.terminated & out.truncated)
        assert np.all(out.rewards[out.truncated] == 0.0)
        assert np.all(out.rewards[first_step & ~goal] == 0.0)
        reset_steps += int(np.count_nonzero(first_step))

        ends["but"] += int(np.count_nonzero(out.terminated))
        ends["no-touch"] += int(np.count_nonzero(out.truncated & (since_touch >= cfg.no_touch_ticks)))
        ends["durée"] += int(np.count_nonzero(out.truncated & (since_touch < cfg.no_touch_ticks)))
        done = out.terminated | out.truncated
        first_step = done
        since_touch = np.where(done, 0, since_touch)
        ep_ticks = np.where(done, 0, ep_ticks)
    # Assert : couverture des trois fins et nombre d'épisodes
    assert sum(ends.values()) >= 1000, ends
    assert all(count >= 50 for count in ends.values()), ends
    assert reset_steps >= 1000


# ---------------------------------------------------------------- §7.4 terminated / truncated


def test_7_4_but_termine_sans_tronquer() -> None:
    cfg = RewardConfig()
    terminated, truncated = episode_end([GoalNet.BLUE, GoalNet.ORANGE], [0, 0], [8, 8], cfg)
    assert terminated.tolist() == [True, True]
    assert truncated.tolist() == [False, False]


def test_7_4_no_touch_et_duree_tronquent_au_seuil() -> None:
    # Arrange : seuils §2.2 en ticks à 120 Hz.
    cfg = RewardConfig()
    nt, mx = 20 * PHYSICS_TICK_RATE_HZ, 120 * PHYSICS_TICK_RATE_HZ
    assert (cfg.no_touch_ticks, cfg.max_episode_ticks) == (nt, mx)
    since = [nt - 1, nt, 0, 0, 0]
    ep = [nt, nt, mx - 1, mx, 0]
    nets = [GoalNet.NONE] * 5
    # Act
    terminated, truncated = episode_end(nets, since, ep, cfg)
    # Assert (choix A7 : déclenchement à >= seuil)
    assert terminated.tolist() == [False] * 5
    assert truncated.tolist() == [False, True, False, True, False]


def test_7_4_but_et_troncature_au_meme_step_donnent_terminated() -> None:
    cfg = RewardConfig()
    terminated, truncated = episode_end(
        [GoalNet.ORANGE, GoalNet.BLUE], [cfg.no_touch_ticks, 0], [8, cfg.max_episode_ticks], cfg
    )
    assert terminated.tolist() == [True, True]  # choix A6
    assert truncated.tolist() == [False, False]


def test_7_4_limites_lues_dans_la_config() -> None:
    cfg = RewardConfig(no_touch_timeout_s=45.0)  # ablation 20/45 s de la spec
    _, truncated = episode_end([GoalNet.NONE, GoalNet.NONE], [20 * 120, 45 * 120], [0, 0], cfg)
    assert truncated.tolist() == [False, True]


# ---------------------------------------------------------------- §7.5 miroir (partie récompense seulement)


@pytest.mark.parametrize("variant", GCO_VARIANTS)
def test_7_5_miroir_recompenses_opposees(variant: RewardVariant) -> None:
    # Arrange : état et son miroir bleu↔orange (filet échangé, NONE inchangé), sur 1v1, 2v2, 3v3.
    rng = np.random.default_rng(SEED)
    fn = make_reward({"variant": variant.value})
    for n in (1, 2, 3):
        teams = np.array([rng.permutation([0] * n + [1] * n) for _ in range(1000)])
        nets = rng.integers(-1, 2, size=1000)
        mirror_nets = np.where(nets == GoalNet.NONE, GoalNet.NONE, 1 - nets)
        # Act
        r = fn.rewards(nets, teams)
        r_mirror = fn.rewards(mirror_nets, teams)
        r_mirror_relabel = fn.rewards(mirror_nets, 1 - teams)
        # Assert : mêmes agents ⇒ récompenses opposées ; agents réétiquetés ⇒ récompenses identiques (équivariance).
        np.testing.assert_array_equal(r_mirror, -r)
        np.testing.assert_array_equal(r_mirror_relabel, r)


# ---------------------------------------------------------------- §7.6 pas de fuite


@pytest.mark.parametrize("variant", GCO_VARIANTS)
def test_7_6_aucune_autre_variable_ne_change_la_recompense(variant: RewardVariant) -> None:
    """La garantie §7.6 est structurelle : l'API n'accepte aucune variable d'état (balle, voitures, boost, toucheur...),
    donc un mock d'état modifié ne peut pas l'atteindre. Le test vérifie (a) que les signatures sont fermées et
    (b) que les seules autres entrées acceptées, les compteurs de ticks, ne changent jamais la récompense, y compris
    au-delà des seuils de troncature."""
    # Arrange
    rng = np.random.default_rng(SEED)
    fn = make_reward({"variant": variant.value})
    layout = [BLU, BLU, ORA, ORA]
    n = 2000
    since = rng.integers(0, 4 * fn.config.no_touch_ticks, size=n)
    ep = rng.integers(0, 2 * fn.config.max_episode_ticks, size=n)
    for net in GoalNet:
        reference = np.array(fn.rewards_one(net, layout), dtype=np.float32)
        # Act
        out = fn.step(np.full(n, int(net)), layout, since, ep)
        # Assert : compteurs sans effet sur r (ils changent bien les drapeaux : la boucle n'est pas vide de sens).
        np.testing.assert_array_equal(out.rewards, np.broadcast_to(reference, (n, len(layout))))
        if net is GoalNet.NONE:
            assert out.truncated.any() and not out.truncated.all()
    # Assert : signatures fermées, aucune variable d'état ne peut entrer dans le signal.
    assert list(inspect.signature(gco_team_rewards).parameters) == ["goal_net", "agent_team"]
    assert list(inspect.signature(RewardFunction.rewards).parameters) == ["self", "goal_net", "agent_team", "scorer"]
    assert list(inspect.signature(RewardFunction.step).parameters) == [
        "self",
        "goal_net",
        "agent_team",
        "ticks_since_touch",
        "episode_ticks",
        "scorer",
    ]


def test_7_6_le_signal_ne_depend_que_du_filet_et_des_equipes() -> None:
    # Arrange : toutes les combinaisons (filet, composition 2v2) en entrée
    fn = make_reward({"variant": "GCO-std"})
    layouts = sorted(set(permutations([0, 0, 1, 1])))
    assert len(layouts) == 6
    # Act / Assert
    for layout in layouts:
        for net in GoalNet:
            assert list(fn.rewards_one(net, layout)) == _expected(net, [Team(t) for t in layout])


# ---------------------------------------------------------------- §7.7 variantes et hash de config


def test_7_7_variantes_instanciables_par_config() -> None:
    for variant in RewardVariant:
        fn = make_reward({"variant": variant.value, "no_touch_timeout_s": 20, "max_episode_s": 120})
        assert fn.variant is variant
        assert fn.config.rules == rules_for(variant)
    assert make_reward({}).variant is RewardVariant.GCO_STD  # défaut = ligne principale (G1 H1)


def test_7_7_regles_des_variantes_transcrivent_le_tableau_3() -> None:
    s, std, ext, so = (rules_for(v) for v in RewardVariant)
    assert not any([s.generic_initial_states, s.learned_advantage_signals, s.intrinsic_bonuses, s.self_imitation])
    assert std.generic_initial_states and std.learned_advantage_signals and std.self_imitation and std.auxiliary_tasks
    assert not std.targeted_initial_states and not std.external_opponents and not std.demonstrations
    assert ext.targeted_initial_states and ext.external_opponents and ext.demonstrations and ext.ablation_only
    assert all(r.learned_signals_require_c1_c4 for r in (s, std, ext, so))  # choix A3
    assert not so.in_spec and so.ablation_only and not so.team_attribution
    # choix A2 : scorer-only = règles GCO-S hors attribution et étiquettes
    rest = {"in_spec", "ablation_only", "team_attribution"}
    assert {k: v for k, v in asdict(so).items() if k not in rest} == {
        k: v for k, v in asdict(s).items() if k not in rest
    }
    assert all(r.in_spec and r.team_attribution for r in (s, std, ext))


def test_7_7_config_refuse_cles_et_valeurs_invalides() -> None:
    with pytest.raises(ValueError, match="inconnues"):
        RewardConfig.from_dict({"variant": "GCO-std", "goal_scale": 150})
    with pytest.raises(ValueError):
        RewardConfig.from_dict({"variant": "GCO-strict"})
    with pytest.raises(ValueError, match="tick physique"):
        RewardConfig(no_touch_timeout_s=20.001)
    with pytest.raises(ValueError, match="> 0"):
        RewardConfig(max_episode_s=0)
    with pytest.raises(TypeError):
        RewardConfig(variant="GCO-std")  # type: ignore[arg-type]


GOLDEN_HASH_GCO_STD = "d531d5919a4d91811b47ca13fe0f372b5b09a45fcbcef56b767fcfc04cc8d9bd"


def test_7_7_hash_stable() -> None:
    # Valeur figée : si elle change, la sémantique ou la config par défaut a changé → ADR + spec (§7.8).
    assert RewardConfig().config_hash() == GOLDEN_HASH_GCO_STD
    assert RewardConfig.from_dict({"no_touch_timeout_s": 20, "max_episode_s": 120}).config_hash() == GOLDEN_HASH_GCO_STD
    assert make_reward(RewardConfig()).config_hash == GOLDEN_HASH_GCO_STD
    roundtrip = RewardConfig.from_dict(RewardConfig().to_dict())
    assert roundtrip == RewardConfig() and roundtrip.config_hash() == GOLDEN_HASH_GCO_STD


def test_7_7_hash_stable_entre_processus() -> None:
    # Un autre interpréteur, avec une graine de hachage Python différente, donne le même hash.
    env = {**os.environ, "PYTHONHASHSEED": "12345", "PYTHONPATH": str(ROOT / "src")}
    code = "from rewards import RewardConfig; print(RewardConfig().config_hash())"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == GOLDEN_HASH_GCO_STD


def test_7_7_hash_sensible_a_chaque_champ() -> None:
    base = RewardConfig()
    variants = [RewardConfig(variant=v) for v in RewardVariant]
    changed = [RewardConfig(no_touch_timeout_s=45.0), RewardConfig(max_episode_s=300.0)]
    hashes = [c.config_hash() for c in [*variants, *changed]]
    assert len(set(hashes)) == len(hashes)
    assert base.config_hash() in hashes  # GCO-std par défaut
    assert all(len(h) == 64 for h in hashes)


# ---------------------------------------------------------------- validation des entrées


@pytest.mark.parametrize(
    ("nets", "teams", "match"),
    [
        ([2], [BLU, ORA], "goal_net hors"),
        ([0], [BLU, BLU], "inégales"),
        ([0], [BLU, ORA, ORA], "inégales"),
        ([0], [0, 2], "agent_team hors"),
        ([0.0], [BLU, ORA], "entiers"),
        ([True], [BLU, ORA], "entiers"),
        ([[0]], [BLU, ORA], "forme"),
        ([0, 1], np.zeros((3, 2), dtype=int), "forme"),
    ],
)
def test_entrees_invalides_refusees(nets: Any, teams: Any, match: str) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        gco_team_rewards(nets, teams)


def test_compteurs_negatifs_refuses() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        episode_end([GoalNet.NONE], [-1], [0], RewardConfig())


@pytest.mark.parametrize(
    ("attr", "value"),
    [("SPEC_VERSION", "0.3"), ("HASH_SCHEMA_VERSION", 2), ("PHYSICS_TICK_RATE_HZ", 60)],
)
def test_7_7_hash_sensible_aux_versions_et_constantes(monkeypatch: pytest.MonkeyPatch, attr: str, value: Any) -> None:
    # Arrange
    from rewards import config as config_module

    before = RewardConfig().config_hash()
    # Act
    monkeypatch.setattr(config_module, attr, value)
    after = RewardConfig().config_hash()
    # Assert
    assert after != before


def test_7_7_hash_sensible_aux_regles_derivees(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : une règle de GCO-std modifiée sans toucher aux champs de config.
    from dataclasses import replace

    from rewards import config as config_module

    before = RewardConfig().config_hash()
    rules = dict(config_module._RULES)
    rules[RewardVariant.GCO_STD] = replace(rules[RewardVariant.GCO_STD], demonstrations=True)
    # Act
    monkeypatch.setattr(config_module, "_RULES", rules)
    after = RewardConfig().config_hash()
    # Assert
    assert after != before
