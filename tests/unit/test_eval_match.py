"""Matchs d'évaluation (`eval.match`) sur un simulateur factice : paires, côtés, arrêt, déterminisme."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from actions import ACTION_TABLE_SHA256, N_ACTIONS
from env.initial_states import InitialStates
from eval.match import (
    DRAW,
    LOSS,
    TICKS_PER_MINUTE,
    WIN,
    MatchResult,
    NetPolicy,
    RandomPolicy,
    load_checkpoint_policy,
    load_policy,
    pair_seeds,
    play_match,
    sprt_on_match,
    summarize,
)
from eval.sprt import Decision, SPRTConfig
from models import PolicyNet
from obs import OBS_SCHEMA_VERSION
from obs.state import BatchState
from rewards import RewardConfig
from training.checkpoint import (
    CheckpointCorruptError,
    CheckpointIdentity,
    CheckpointIncompatibleError,
    CheckpointManager,
)

F32 = npt.NDArray[np.float32]
I64 = npt.NDArray[np.int64]
TICK_SKIP = 8
OBS_D = 3
NO_TOUCH_S = 1.0  # 120 ticks = 15 steps sans touche : troncature (nul) ; le simulateur factice ne produit aucune touche
CFG = RewardConfig(no_touch_timeout_s=NO_TOUCH_S)


class DuelSim:
    """Simulateur factice 1v1. Par step et par arène, la voiture bleue marque avec la probabilité
    `rate[action] + blue_bonus`,
    l'orange avec `rate[action]` (un seul but par step, bleu prioritaire). Le mode `scripted` remplace ce hasard par un
    script fonction de la graine de kickoff : graine paire => but bleu au step 3, graine impaire => aucun but (nul)."""

    def __init__(
        self,
        n_envs: int,
        rates: dict[int, float],
        blue_bonus: float = 0.0,
        seed: int = 0,
        scripted: bool = False,
    ) -> None:
        self._e = n_envs
        self._rates = np.array([rates.get(a, 0.0) for a in range(N_ACTIONS)])
        self._bonus = blue_bonus
        self._rng = np.random.default_rng(seed)
        self._scripted = scripted
        self._state = BatchState.zeros(n_envs, 2)
        self._blue = np.zeros(n_envs, dtype=np.int64)
        self._orange = np.zeros(n_envs, dtype=np.int64)
        self._tick = np.zeros(n_envs, dtype=np.int64)
        self._arena_seed = np.ones(n_envs, dtype=np.uint64)
        self._since_reset = np.zeros(n_envs, dtype=np.int64)
        self.reset_log: list[tuple[int, int]] = []  # (arène, graine de kickoff) dans l'ordre des resets

    n_envs = property(lambda self: self._e)
    n_cars = property(lambda self: 2)
    tick_skip = property(lambda self: TICK_SKIP)

    def state(self) -> BatchState:
        return self._state

    def step(self, actions: I64) -> None:
        self._tick += TICK_SKIP
        self._since_reset += 1
        act = actions.reshape(self._e, 2)
        if self._scripted:
            blue_goal = (self._arena_seed % 2 == 0) & (self._since_reset == 3)
            self._blue += blue_goal
            return
        pb = np.clip(self._rates[act[:, 0]] + self._bonus, 0.0, 1.0)
        po = self._rates[act[:, 1]]
        u = self._rng.random(self._e)
        self._blue += u < pb
        self._orange += (u >= pb) & (u < pb + po)

    def scores(self) -> tuple[I64, I64]:
        return self._blue.copy(), self._orange.copy()

    def env_ticks(self) -> I64:
        return self._tick.copy()

    def touch_ticks(self) -> I64:
        return np.full((self._e, 2), -1, dtype=np.int64)

    def reset_to_kickoff(self, arena_idx: I64, kickoff_seeds: Any) -> None:
        for a, s in zip(arena_idx.tolist(), np.asarray(kickoff_seeds).tolist(), strict=True):
            self.reset_log.append((a, int(s)))
            self._arena_seed[a] = s
            self._since_reset[a] = 0

    def set_states(self, arena_idx: I64, states: InitialStates) -> None:
        raise AssertionError("l'évaluation ne joue que des kickoffs (p_kick = 1)")


def duel_obs(state: BatchState, prev: I64) -> F32:
    out = np.zeros((prev.shape[0], OBS_D), dtype=np.float32)
    out[:, 0] = prev
    return out


class ConstPolicy:
    def __init__(self, action: int) -> None:
        self.action = action
        self.calls = 0

    def __call__(self, obs: F32) -> I64:
        self.calls += 1
        return np.full(obs.shape[0], self.action, dtype=np.int64)


def play(
    a: Any,
    b: Any,
    sim: DuelSim,
    n: int,
    seed: int = 0,
) -> MatchResult:
    return play_match(a, b, sim, n, seed=seed, reward_config=CFG, build_obs=duel_obs, obs_dim=OBS_D)


# ------------------------------------------------------------------------------------------------ structure


def test_chaque_paire_est_jouee_deux_fois_avec_la_meme_graine_et_cotes_echanges() -> None:
    sim = DuelSim(8, {1: 0.05})
    n_episodes = 40
    res = play(ConstPolicy(1), ConstPolicy(1), sim, n_episodes, seed=3)
    assert res.n_pairs == 20 and res.outcome.shape == (20, 2) and res.n_episodes == 40
    seeds = pair_seeds(3, res.n_pairs)
    per_arena: dict[int, list[int]] = {}
    for arena, s in sim.reset_log:
        per_arena.setdefault(arena, []).append(s)
    for p in range(res.n_pairs):
        a, k = p % sim.n_envs, p // sim.n_envs
        # la paire p est jouée dans l'arène p mod E : épisodes 2k (A bleu) et 2k+1 (A orange), même graine
        assert per_arena[a][2 * k] == per_arena[a][2 * k + 1] == int(seeds[p])
    assert ((seeds >= 1) & (seeds <= 2**32 - 1)).all()
    assert (res.ticks > 0).all()  # tous les épisodes planifiés sont terminés et mesurés


def test_nombre_impair_d_episodes_arrondi_a_la_paire() -> None:
    res = play(ConstPolicy(1), ConstPolicy(1), DuelSim(4, {1: 0.05}), 7)
    assert res.n_pairs == 4 and res.n_episodes == 8


def test_cote_de_a_victoire_defaite_nul() -> None:
    """A (action 1, marque) contre B (action 0, ne marque jamais) : A gagne dans les deux côtés, jamais de défaite."""
    res = play(ConstPolicy(1), ConstPolicy(0), DuelSim(8, {1: 0.2}), 80)
    assert (res.outcome == LOSS).sum() == 0
    assert (res.outcome[:, 0] == WIN).sum() > 0 and (res.outcome[:, 1] == WIN).sum() > 0
    assert (res.outcome == DRAW).sum() > 0  # troncature no-touch : nul
    lost = play(ConstPolicy(0), ConstPolicy(1), DuelSim(8, {1: 0.2}), 80)
    assert (lost.outcome == WIN).sum() == 0 and (lost.outcome == LOSS).sum() > 0  # rôles inversés : A perd


def test_duree_en_ticks_des_nuls_est_le_no_touch() -> None:
    res = play(ConstPolicy(0), ConstPolicy(0), DuelSim(4, {}), 16)
    assert (res.outcome == DRAW).all()
    assert (res.ticks == 120).all()  # 1 s sans touche à 120 Hz


# ------------------------------------------------------------------------------------------------ côtés et force


def test_avantage_de_cote_n_apporte_pas_d_elo_une_fois_les_cotes_echanges() -> None:
    """Le bleu marque 3x plus vite (bonus de côté) ; A = B = même politique : score ≈ 50 %, Elo ≈ 0 dans l'IC."""
    sim = DuelSim(16, {1: 0.02}, blue_bonus=0.04, seed=1)
    res = play(ConstPolicy(1), ConstPolicy(1), sim, 300, seed=11)
    s = summarize(res, n_boot=200, seed=0)
    assert s.win_rate_blue > 0.55 and s.win_rate_orange < 0.45  # l'avantage de côté existe bien dans le simulateur
    assert s.score.low is not None and s.score.high is not None
    assert s.score.low <= 0.5 <= s.score.high
    assert s.elo_decisive.low is not None and s.elo_decisive.high is not None
    assert s.elo_decisive.low <= 0.0 <= s.elo_decisive.high
    # contre-épreuve : sans échange de côtés (A toujours bleu), le même avantage donnerait un Elo très positif
    only_blue = res.outcome[:, :1]
    w, lo = int((only_blue == WIN).sum()), int((only_blue == LOSS).sum())
    assert w > 1.5 * lo


def test_politique_contre_elle_meme_donne_50_pour_cent_dans_l_ic() -> None:
    sim = DuelSim(16, {1: 0.05}, seed=2)
    res = play(ConstPolicy(1), ConstPolicy(1), sim, 300, seed=5)
    s = summarize(res, n_boot=200, seed=1)
    assert s.score.low is not None and s.score.high is not None
    assert s.score.low <= 0.5 <= s.score.high
    assert s.elo_decisive.low is not None and s.elo_decisive.low <= 0.0 <= (s.elo_decisive.high or 0.0)
    assert not s.elo_regularized


def test_politique_plus_forte_gagne_de_l_elo_significatif() -> None:
    sim = DuelSim(16, {1: 0.06, 2: 0.02}, seed=3)  # décisif : 75 % de victoires de A
    res = play(ConstPolicy(1), ConstPolicy(2), sim, 300, seed=7)
    s = summarize(res, n_boot=200, seed=2)
    assert s.elo_decisive.low is not None and s.elo_decisive.low > 100.0
    assert s.goals_for_per_min.value is not None and s.goals_against_per_min.value is not None
    assert s.goals_for_per_min.value > 2.0 * s.goals_against_per_min.value


# ------------------------------------------------------------------------------------------------ règle d'arrêt


def test_regle_d_arret_sans_biais_vers_les_episodes_courts() -> None:
    """Graine paire : but bleu au step 3 (épisode court de 24 ticks) ; graine impaire : nul de 120 ticks (épisode
    long). Les paires de graine impaire finissent bien après les autres, mais l'échantillon planifié doit être joué en
    entier : A gagne exactement une fois par paire paire, 2 nuls exactement par paire impaire. Une règle « s'arrêter
    aux N premiers épisodes terminés » aurait sous-représenté les nuls longs."""
    n_pairs = 60
    seeds = pair_seeds(9, n_pairs)
    n_even = int((seeds % 2 == 0).sum())
    assert 10 < n_even < n_pairs - 10  # les deux types sont présents
    res = play(ConstPolicy(0), ConstPolicy(0), DuelSim(8, {}, scripted=True), 2 * n_pairs, seed=9)
    assert (res.outcome == WIN).sum() == n_even and (res.outcome == LOSS).sum() == n_even
    assert (res.outcome == DRAW).sum() == 2 * (n_pairs - n_even)
    even = seeds % 2 == 0
    assert (res.ticks[even] == 24).all() and (res.ticks[~even] == 120).all()
    s = summarize(res, n_boot=200, seed=0)
    assert s.draw_rate.value == pytest.approx(1 - n_even / n_pairs)  # part exacte de l'échantillon planifié


def test_peu_de_paires_pour_beaucoup_d_arenes_termine_et_ignore_les_arenes_a_vide() -> None:
    res = play(ConstPolicy(1), ConstPolicy(1), DuelSim(16, {1: 0.1}), 4)  # 2 paires pour 8 couples d'arènes
    assert res.n_pairs == 2 and (res.ticks > 0).all()


# ------------------------------------------------------------------------------------------------ déterminisme


def test_determinisme_a_seed_fixe_et_sensibilite_a_la_seed() -> None:
    def run(seed: int) -> MatchResult:
        sim = DuelSim(8, {1: 0.05}, seed=4)
        return play(RandomPolicy(1), ConstPolicy(1), sim, 120, seed=seed)

    a, b = run(5), run(5)
    assert a.equals(b)
    assert not np.array_equal(pair_seeds(5, 10), pair_seeds(6, 10))
    sa, sb = summarize(a, seed=3, n_boot=100), summarize(b, seed=3, n_boot=100)
    assert sa == sb  # bootstrap à graine fixe


# ------------------------------------------------------------------------------------------------ politiques


def test_politique_reseau_sans_gradient_et_sans_modifier_les_poids() -> None:
    torch.manual_seed(0)
    net_a, net_b = PolicyNet(OBS_D, hidden=(16, 16)), PolicyNet(OBS_D, hidden=(16, 16))
    before = {k: v.clone() for k, v in net_a.state_dict().items()}
    pol_a, pol_b = NetPolicy(net_a, "sample", seed=1), NetPolicy(net_b, "argmax", seed=2)
    res = play(pol_a, pol_b, DuelSim(8, {1: 0.05}), 40)
    assert res.n_pairs == 20
    after = net_a.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(p.grad is None for p in net_a.parameters())
    assert not net_a.training  # eval()
    assert torch.is_grad_enabled()  # état global restauré
    out = pol_a(np.zeros((5, OBS_D), dtype=np.float32))
    assert out.dtype == np.int64 and out.shape == (5,) and ((out >= 0) & (out < N_ACTIONS)).all()


def test_modes_sample_et_argmax() -> None:
    torch.manual_seed(1)
    net = PolicyNet(OBS_D, hidden=(8,))
    with torch.no_grad():
        net.net[-1].bias.zero_()
        net.net[-1].bias[17] = 50.0  # action 17 quasi certaine
    obs = np.ones((64, OBS_D), dtype=np.float32)
    assert (NetPolicy(net, "argmax", seed=0)(obs) == 17).all()
    assert (NetPolicy(net, "sample", seed=0)(obs) == 17).all()
    flat = PolicyNet(OBS_D, hidden=(8,))
    with torch.no_grad():
        flat.net[-1].weight.zero_()
        flat.net[-1].bias.zero_()
    assert (NetPolicy(flat, "argmax", seed=0)(obs) == 0).all()  # égalité : premier indice
    drawn = NetPolicy(flat, "sample", seed=0)(np.ones((4000, OBS_D), dtype=np.float32))
    assert len(set(drawn.tolist())) > 80  # uniforme sur 90 actions
    same = NetPolicy(flat, "sample", seed=0)(np.ones((4000, OBS_D), dtype=np.float32))
    assert np.array_equal(drawn, same)  # graine fixe
    with pytest.raises(ValueError):
        NetPolicy(net, "greedy", seed=0)  # type: ignore[arg-type]


def test_agent_aleatoire_uniforme_sur_les_90_actions() -> None:
    draws = RandomPolicy(0)(np.zeros((90_000, OBS_D), dtype=np.float32))
    counts = np.bincount(draws, minlength=N_ACTIONS)
    assert counts.shape == (N_ACTIONS,) and counts.min() > 800 and counts.max() < 1200  # attendu 1000 ± 3 sigma (~95)
    assert np.array_equal(draws[:10], RandomPolicy(0)(np.zeros((10, OBS_D), dtype=np.float32)))


def test_validation_des_entrees() -> None:
    odd = play(ConstPolicy(1), ConstPolicy(1), DuelSim(3, {1: 0.1}), 10)  # nombre impair d'arènes : accepté
    assert odd.n_pairs == 5 and (odd.ticks > 0).all()
    with pytest.raises(ValueError, match="paire"):
        play(ConstPolicy(1), ConstPolicy(1), DuelSim(4, {}), 1)

    def bad(obs: F32) -> I64:
        return np.zeros(obs.shape[0] + 1, dtype=np.int64)

    with pytest.raises(ValueError, match="politique"):
        play(bad, ConstPolicy(1), DuelSim(4, {}), 4)


# ------------------------------------------------------------------------------------------------ agrégats


def _result(outcome: list[list[int]], ticks: list[list[int]]) -> MatchResult:
    o = np.array(outcome, dtype=np.int8)
    return MatchResult(
        outcome=o,
        ticks=np.array(ticks, dtype=np.int64),
        seed=0,
        n_pairs=o.shape[0],
        n_envs=4,
        tick_skip=TICK_SKIP,
        env_steps=0,
        wall_s=0.0,
    )


def test_buts_par_minute_et_comptes() -> None:
    # 10 paires : A marque dans 4 épisodes, encaisse dans 2, 14 nuls ; chaque épisode dure 1 minute (7200 ticks)
    outcome = [[WIN, DRAW]] * 2 + [[WIN, LOSS]] * 2 + [[DRAW, DRAW]] * 6
    res = _result(outcome, [[TICKS_PER_MINUTE] * 2] * 10)
    s = summarize(res, n_boot=200, seed=0)
    assert (s.wins, s.losses, s.draws) == (4, 2, 14)
    assert s.minutes_played == pytest.approx(20.0)
    assert s.goals_for_per_min.value == pytest.approx(4 / 20) and s.goals_against_per_min.value == pytest.approx(2 / 20)
    assert s.score.value == pytest.approx((4 + 7) / 20) and s.draw_rate.value == pytest.approx(0.7)
    assert s.mean_episode_s == pytest.approx(60.0)
    lo, hi = s.goals_for_per_min.low, s.goals_for_per_min.high
    assert lo is not None and hi is not None and lo < 0.2 < hi
    assert s.elo_decisive.value == pytest.approx(400 * np.log10(2.0))  # 4 victoires contre 2 défaites
    d = s.as_dict()
    assert d["bootstrap"]["unit"] == "pair" and d["wins"] == 4


def test_invaincu_donne_elo_regularise_signale_et_aucun_decisif_donne_none() -> None:
    res = _result([[WIN, WIN]] * 10, [[100, 100]] * 10)
    s = summarize(res, n_boot=200, seed=0)
    assert s.elo_regularized and s.elo_decisive.value is not None and np.isfinite(s.elo_decisive.value)
    none = summarize(_result([[DRAW, DRAW]] * 5, [[100, 100]] * 5), n_boot=100, seed=0)
    assert none.elo_decisive.value is None and not none.elo_regularized
    assert none.goals_for_per_min.value == 0.0 and none.draw_rate.value == 1.0


def test_bootstrap_par_paires_ic_plus_large_si_les_paires_sont_correlees() -> None:
    """Deux épisodes d'une même paire identiques (corrélation parfaite) : l'IC par paires est ≈ √2 fois plus large que
    celui d'épisodes indépendants de même taille ; on vérifie simplement qu'il n'est pas plus étroit."""
    rng = np.random.default_rng(0)
    pair_w = rng.random(200) < 0.5
    corr = _result([[WIN, WIN] if w else [LOSS, LOSS] for w in pair_w], [[100, 100]] * 200)
    indep = _result(rng.choice([WIN, LOSS], size=(200, 2)).tolist(), [[100, 100]] * 200)
    wc = summarize(corr, n_boot=300, seed=0).score
    wi = summarize(indep, n_boot=300, seed=0).score
    assert wc.low is not None and wc.high is not None and wi.low is not None and wi.high is not None
    assert (wc.high - wc.low) > 1.2 * (wi.high - wi.low)


def test_sprt_sur_un_match_ordre_par_paires() -> None:
    cfg = SPRTConfig(max_episodes=1200)
    res = _result([[WIN, WIN]] * 40, [[100, 100]] * 40)
    r = sprt_on_match(res, cfg)
    assert r.decision is Decision.ACCEPT_H1 and r.n_episodes % 2 == 0 and r.n_episodes < 80
    flat = sprt_on_match(_result([[WIN, LOSS]] * 40, [[100, 100]] * 40), SPRTConfig(max_episodes=60))
    assert flat.decision is Decision.CAP_REACHED and flat.n_episodes == 60


# ------------------------------------------------------------------------------------------------ checkpoints


def _write_ckpt(tmp_path: Path, net: PolicyNet, **identity_override: Any) -> Path:
    identity = dataclasses.asdict(
        CheckpointIdentity.current(reward_config_hash="h", obs_dim=OBS_D, tick_skip=TICK_SKIP)
    )
    identity.update(identity_override)
    state = {
        "format_version": 1,
        "identity": identity,
        "train_config_hash": "c",
        "iteration": 7,
        "versions": {"git_commit": "abc"},
        "trainer": {"learner": {"policy": net.state_dict()}},
    }
    return CheckpointManager(tmp_path).save(state)


def test_chargement_checkpoint_poids_identite_et_provenance(tmp_path: Path) -> None:
    torch.manual_seed(2)
    net = PolicyNet(OBS_D, hidden=(12, 12))
    path = _write_ckpt(tmp_path, net)
    pol, info = load_checkpoint_policy(path, mode="argmax", seed=0, obs_dim=OBS_D, tick_skip=TICK_SKIP)
    assert all(torch.equal(v, pol.net.state_dict()[k]) for k, v in net.state_dict().items())
    assert info.kind == "checkpoint" and info.iteration == 7 and info.git_commit == "abc" and info.mode == "argmax"
    assert info.sha256 is not None and len(info.sha256) == 64
    assert info.identity["action_table_sha256"] == ACTION_TABLE_SHA256
    pol2, info2 = load_policy(str(path), mode="sample", seed=1, obs_dim=OBS_D, tick_skip=TICK_SKIP)
    assert isinstance(pol2, NetPolicy) and info2.sha256 == info.sha256
    rnd, rinfo = load_policy("random", mode="sample", seed=1, obs_dim=OBS_D, tick_skip=TICK_SKIP)
    assert isinstance(rnd, RandomPolicy) and rinfo.kind == "random"


@pytest.mark.parametrize(
    "override",
    [
        {"obs_schema_version": OBS_SCHEMA_VERSION + 1},
        {"action_table_sha256": "0" * 64},
        {"action_schema_version": 99},
        {"tick_skip": 4},
        {"obs_dim": OBS_D + 1},
    ],
)
def test_checkpoint_d_identite_differente_refuse(tmp_path: Path, override: dict[str, Any]) -> None:
    path = _write_ckpt(tmp_path, PolicyNet(OBS_D, hidden=(8,)), **override)
    with pytest.raises(CheckpointIncompatibleError):
        load_checkpoint_policy(path, mode="argmax", seed=0, obs_dim=OBS_D, tick_skip=TICK_SKIP)


def test_checkpoint_dont_les_poids_ne_correspondent_pas_a_l_obs_refuse(tmp_path: Path) -> None:
    path = _write_ckpt(tmp_path, PolicyNet(OBS_D + 5, hidden=(8,)))
    with pytest.raises(CheckpointIncompatibleError):
        load_checkpoint_policy(path, mode="argmax", seed=0, obs_dim=OBS_D, tick_skip=TICK_SKIP)


def test_checkpoint_corrompu_refuse(tmp_path: Path) -> None:
    path = _write_ckpt(tmp_path, PolicyNet(OBS_D, hidden=(8,)))
    data = bytearray(path.read_bytes())
    data[-5] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(CheckpointCorruptError):
        load_checkpoint_policy(path, mode="argmax", seed=0, obs_dim=OBS_D, tick_skip=TICK_SKIP)


# ------------------------------------------------------------------------------------------------ scripts/eval.py


def _load_script() -> Any:
    path = Path(__file__).resolve().parents[2] / "scripts" / "eval.py"
    spec = importlib.util.spec_from_file_location("eval_script", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_rapport_json_avec_provenance_et_identite_des_checkpoints(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt = _write_ckpt(tmp_path / "ck", PolicyNet(OBS_D, hidden=(8,)))
    rep = mod.evaluate(
        str(ckpt),
        "random",
        40,
        seed=1,
        sim=DuelSim(8, {1: 0.06, 2: 0.02}),
        reward_config=CFG,
        build_obs=duel_obs,
        obs_dim=OBS_D,
        sprt_elo=50.0,
        n_boot=100,
        command=["python", "scripts/eval.py", "--a", str(ckpt)],
    )
    again = json.loads(json.dumps(rep))  # entièrement sérialisable
    prov = again["provenance"]
    assert prov["seed"] == 1 and prov["command"][1] == "scripts/eval.py" and len(prov["git_commit"]) >= 7
    assert prov["git_dirty"] in (True, False, None) and "date_utc" in prov and "thermal_state" in prov
    assert {"platform", "processor", "device", "torch"} <= set(prov["hardware"])
    assert again["policies"]["a"]["kind"] == "checkpoint"
    assert again["policies"]["a"]["sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert again["policies"]["b"]["kind"] == "random"
    assert again["config"]["episodes_played"] == 40 and again["config"]["p_kick"] == 1.0
    assert again["summary"]["n_episodes"] == 40 and len(again["per_pair"]["outcome"]) == 20
    assert again["throughput"]["episodes_per_s"] > 0 and again["sprt"]["h1_elo"] == 50.0
    assert again["sprt"]["decision"] in ("continue", "accept_h1", "accept_h0", "cap_reached")


def test_script_refuse_d_ecraser_la_sortie(tmp_path: Path) -> None:
    mod = _load_script()
    out = tmp_path / "res.json"
    out.write_text("{}", encoding="utf-8")
    code = mod.main(["--a", "random", "--b", "random", "--episodes", "4", "--out", str(out)])
    assert code == 2 and out.read_text(encoding="utf-8") == "{}"


# ------------------------------------------------------------------------------------------------ attribution A / côté


class ActionDuelSim(DuelSim):
    """Déterministe : au 2e step de chaque épisode, une équipe marque si son action est STRICTEMENT supérieure à
    celle de l'autre (égalité : aucun but). Aucun hasard : le résultat ne dépend que des actions et de leur slot."""

    def step(self, actions: I64) -> None:
        self._tick += TICK_SKIP
        self._since_reset += 1
        act = actions.reshape(self._e, 2)
        at_goal = self._since_reset == 2
        self._blue += at_goal & (act[:, 0] > act[:, 1])
        self._orange += at_goal & (act[:, 1] > act[:, 0])


def slot_obs(state: BatchState, prev: I64) -> F32:
    """Obs encodant l'agent : [slot (indice d'agent % 2), arène (indice d'agent // 2), action précédente]."""
    n = prev.shape[0]
    out = np.zeros((n, OBS_D), dtype=np.float32)
    out[:, 0] = np.arange(n) % 2
    out[:, 1] = np.arange(n) // 2
    out[:, 2] = prev
    return out


class SlotPolicy:
    """Action `a_slot0` si l'obs indique le slot 0, `a_slot1` sinon ; enregistre les obs reçues à chaque appel."""

    def __init__(self, a_slot0: int, a_slot1: int) -> None:
        self.a = (a_slot0, a_slot1)
        self.calls: list[F32] = []

    def __call__(self, obs: F32) -> I64:
        self.calls.append(obs.copy())
        return np.where(obs[:, 0] == 0, self.a[0], self.a[1]).astype(np.int64)


def play_slots(a: Any, b: Any, n_envs: int = 4, n: int = 16) -> MatchResult:
    return play_match(a, b, ActionDuelSim(n_envs, {}), n, seed=0, reward_config=CFG, build_obs=slot_obs, obs_dim=OBS_D)


def test_attribution_de_a_a_son_slot_et_son_cote_resultats_exacts() -> None:
    """A = « 5 si slot 0, sinon 0 » contre B = 3 : A gagne exactement quand il est bleu (slot 0), perd quand il est
    orange ; rôles inversés (A = « 5 si slot 1, sinon 0 »), c'est le contraire."""
    res = play_slots(SlotPolicy(5, 0), SlotPolicy(3, 3))
    assert (res.outcome[:, 0] == WIN).all() and (res.outcome[:, 1] == LOSS).all()
    inv = play_slots(SlotPolicy(0, 5), SlotPolicy(3, 3))
    assert (inv.outcome[:, 0] == LOSS).all() and (inv.outcome[:, 1] == WIN).all()
    assert (res.ticks == 16).all()  # 2 steps de 8 ticks : le but tombe bien au 2e step


def test_acheminement_a_ne_recoit_que_les_lignes_de_ses_slots() -> None:
    """Chaque appel de A (resp. B) reçoit exactement une ligne par arène, dans l'ordre des arènes, du slot de A
    (resp. B) : A est au slot 0 (bleu) aux épisodes pairs de l'arène et au slot 1 (orange) aux
    impairs, B l'inverse. Un épisode dure 2 steps, donc le step t joue l'épisode t // 2. Une attribution « A toujours
    bleu » échoue ici (elle laisse les résultats ci-dessus inchangés, le signe étant compensé)."""
    n_envs = 4
    a, b = SlotPolicy(5, 0), SlotPolicy(3, 3)
    play_slots(a, b, n_envs=n_envs, n=16)
    assert len(a.calls) == len(b.calls) >= 8
    arenas = np.arange(n_envs)
    for t, (obs_a, obs_b) in enumerate(zip(a.calls, b.calls, strict=True)):
        a_slot = (t // 2) % 2
        assert obs_a.shape == (n_envs, OBS_D) and obs_b.shape == (n_envs, OBS_D)
        assert (obs_a[:, 1] == arenas).all() and (obs_b[:, 1] == arenas).all()  # une ligne par arène, dans l'ordre
        assert (obs_a[:, 0] == a_slot).all() and (obs_b[:, 0] == 1 - a_slot).all()
        assert (obs_a[:, 0] != obs_b[:, 0]).all()  # jamais la même ligne pour A et B


def test_attribution_avec_un_nombre_d_arenes_impair_et_un_seul_episode_par_arene() -> None:
    res = play_slots(SlotPolicy(5, 0), SlotPolicy(3, 3), n_envs=3, n=6)  # 3 paires, 3 arènes : 1 paire par arène
    assert (res.outcome[:, 0] == WIN).all() and (res.outcome[:, 1] == LOSS).all()
