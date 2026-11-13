"""Matchs d'évaluation entre deux politiques A et B sur `BatchEnv` (1v1, kickoffs seuls), `docs/01_papers.md` §1.9.

Unité : épisode kickoff -> premier but ; troncature (no-touch ou durée max) = nul. Jeu par PAIRES : la paire p est
formée de deux épisodes de MÊME graine de kickoff, joués l'un après l'autre DANS LA MÊME ARÈNE, A bleu dans le premier
et A orange dans le second. Mesuré (RocketGPU @ build local, voir tests/integration/test_eval_rocketgpu.py) : le
kickoff tiré par une graine dépend aussi de l'indice d'arène (même graine dans deux arènes = positions différentes,
parfois miroir), donc l'appariement exige la même arène. La graine de la paire est imposée via un simulateur
enveloppe (`_PairedKickoffSim`) qui remplace la graine tirée par l'env : l'env d'entraînement n'est pas modifié, et
celui d'évaluation est une instance distincte construite sur un simulateur dédié.

Règle d'arrêt (pas de biais vers les épisodes courts). Les épisodes à jouer sont fixés AVANT de jouer : P = ceil(N/2)
paires, la paire p est jouée par l'arène p mod E, en position p div E. Aucun épisode n'est choisi, tronqué ou écarté
d'après son issue ou sa durée ; on joue jusqu'à ce que TOUS les épisodes planifiés soient terminés (les arènes qui ont
fini leur quota tournent à vide, leurs résultats sont ignorés). Une règle « s'arrêter au N-ième épisode terminé »
sur-représenterait les épisodes courts (les longs en cours seraient exclus) ; ici l'échantillon est exactement
l'ensemble planifié. Coût : on attend l'épisode le plus long (<= durée max).

Une « politique » est un appelable `(obs (n, D) f32) -> indices d'actions (n,) i64`, appelé sur les seuls agents qu'elle
contrôle. Sans gradient (`torch.inference_mode`), poids jamais modifiés.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
import torch

from actions import N_ACTIONS
from env import BatchEnv, InitialStates, InitialStateSampler, Simulator
from eval.sprt import SPRTConfig, SPRTResult, sprt_scan
from models import PolicyNet, sample_actions
from obs.state import BatchState
from rewards import GoalNet, RewardConfig, RewardFunction
from rewards.config import PHYSICS_TICK_RATE_HZ
from training.checkpoint import (
    CheckpointIdentity,
    CheckpointIncompatibleError,
    CheckpointManager,
    check_identity,
)

F32Array = npt.NDArray[np.float32]
I64Array = npt.NDArray[np.int64]
U64Array = npt.NDArray[np.uint64]
I8Array = npt.NDArray[np.int8]
ObsBuilder = Callable[[BatchState, I64Array], F32Array]

TICKS_PER_MINUTE = PHYSICS_TICK_RATE_HZ * 60
KICKOFF_SEED_MOD = 2**32 - 2  # graines de kickoff dans [1, 2^32 - 1] (`env.initial_states.KICKOFF_SEED_RANGE`)

WIN, DRAW, LOSS = 1, 0, -1
"""Résultat d'un épisode du point de vue de A."""


# ------------------------------------------------------------------------------------------------ politiques


class Policy(Protocol):
    def __call__(self, obs: F32Array) -> I64Array:
        """`obs` (n, D) f32 -> indices d'actions (n,) i64 dans [0, 90)."""


class RandomPolicy:
    """Agent aléatoire : uniforme sur les `n_actions` (90) actions, générateur numpy seedé."""

    def __init__(self, seed: int, n_actions: int = N_ACTIONS) -> None:
        self._rng = np.random.default_rng(seed)
        self._n = n_actions

    def __call__(self, obs: F32Array) -> I64Array:
        return np.asarray(self._rng.integers(0, self._n, size=obs.shape[0], dtype=np.int64), dtype=np.int64)


class NetPolicy:
    """Politique d'un `PolicyNet` : `mode` explicite, "sample" (échantillonne π(.|s), comme à l'entraînement) ou
    "argmax" (action la plus probable). Le réseau est passé en `eval()` ; aucun gradient n'est calculé."""

    def __init__(
        self, net: PolicyNet, mode: Literal["sample", "argmax"], seed: int, device: str | torch.device = "cpu"
    ) -> None:
        if mode not in ("sample", "argmax"):
            raise ValueError(f"mode '{mode}' : 'sample' ou 'argmax' attendu")
        self.net = net.to(device).eval()
        self.mode = mode
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def __call__(self, obs: F32Array) -> I64Array:
        with torch.inference_mode():
            logits = self.net(torch.from_numpy(obs).to(self.device))
            act = sample_actions(logits, self.generator)[0] if self.mode == "sample" else logits.argmax(dim=-1)
            out: I64Array = act.cpu().numpy().astype(np.int64, copy=False)
        return out


@dataclass(frozen=True)
class PolicyInfo:
    """Identité d'une politique évaluée (provenance du résultat)."""

    kind: str
    """"random" ou "checkpoint"."""
    path: str | None = None
    sha256: str | None = None
    iteration: int | None = None
    git_commit: str | None = None
    identity: dict[str, Any] = field(default_factory=dict)
    mode: str | None = None


def load_checkpoint_policy(
    path: str | Path,
    *,
    mode: Literal["sample", "argmax"],
    seed: int,
    obs_dim: int,
    tick_skip: int,
    device: str | torch.device = "cpu",
) -> tuple[NetPolicy, PolicyInfo]:
    """Charge la politique d'un checkpoint (somme de contrôle vérifiée avant désérialisation, `weights_only`).

    Identité vérifiée : version du schéma d'obs, dimension d'obs, version du schéma d'actions, SHA-256 de la table
    d'actions et tick-skip doivent coïncider avec le code courant (`CheckpointIncompatibleError` sinon). La récompense
    (hash, version) n'est pas comparée : elle ne change pas ce que la politique fait. Le réseau est reconstruit d'après
    les formes des poids (largeurs cachées lues dans le fichier)."""
    p = Path(path)
    raw = p.read_bytes()
    state = CheckpointManager.load(p)
    saved = state["identity"]
    expected = replace(
        CheckpointIdentity.current(
            reward_config_hash=saved["reward_config_hash"], obs_dim=obs_dim, tick_skip=tick_skip
        ),
        reward_spec_version=saved["reward_spec_version"],
        reward_hash_schema_version=saved["reward_hash_schema_version"],
    )
    check_identity(state, expected)  # obs/actions/tick_skip ; les champs récompense sont recopiés du fichier
    weights = state["trainer"]["learner"]["policy"]
    layers = sorted(int(k.split(".")[1]) for k in weights if k.startswith("net.") and k.endswith(".weight"))
    if not layers:
        raise CheckpointIncompatibleError("poids de politique introuvables dans le checkpoint")
    hidden = tuple(int(weights[f"net.{i}.weight"].shape[0]) for i in layers[:-1])
    n_actions = int(weights[f"net.{layers[-1]}.weight"].shape[0])
    if int(weights[f"net.{layers[0]}.weight"].shape[1]) != obs_dim or n_actions != N_ACTIONS:
        raise CheckpointIncompatibleError(f"politique {obs_dim}->{n_actions} attendue, poids incompatibles")
    net = PolicyNet(obs_dim, n_actions, hidden)
    net.load_state_dict(weights)
    info = PolicyInfo(
        kind="checkpoint",
        path=str(p),
        sha256=hashlib.sha256(raw).hexdigest(),
        iteration=int(state["iteration"]),
        git_commit=str(state["versions"].get("git_commit")),
        identity=dict(saved),
        mode=mode,
    )
    return NetPolicy(net, mode, seed, device), info


def load_policy(
    spec: str,
    *,
    mode: Literal["sample", "argmax"],
    seed: int,
    obs_dim: int,
    tick_skip: int,
    device: str | torch.device = "cpu",
) -> tuple[Policy, PolicyInfo]:
    """`"random"` -> agent aléatoire ; sinon chemin d'un checkpoint."""
    if spec == "random":
        return RandomPolicy(seed), PolicyInfo(kind="random")
    return load_checkpoint_policy(spec, mode=mode, seed=seed, obs_dim=obs_dim, tick_skip=tick_skip, device=device)


# ------------------------------------------------------------------------------------------------ kickoffs appariés


def pair_seeds(seed: int, n_pairs: int) -> U64Array:
    """Graines de kickoff des paires (uint32 non nul, `KICKOFF_SEED_RANGE`), fonction de `seed` seul."""
    raw = np.random.SeedSequence(seed).generate_state(n_pairs, dtype=np.uint32).astype(np.uint64)
    out: U64Array = raw % np.uint64(KICKOFF_SEED_MOD) + np.uint64(1)
    return out


class _PairedKickoffSim:
    """Enveloppe de `Simulator` : au n-ième reset de l'arène a (n = 0, 1, ...), impose la graine de la paire
    p = a + (n // 2)·E ; les épisodes n = 2k (A bleu) et 2k+1 (A orange) de l'arène partagent donc leur graine."""

    def __init__(self, sim: Simulator, seeds: U64Array) -> None:
        self._sim = sim
        self._seeds = seeds
        self.reset_count = np.zeros(sim.n_envs, dtype=np.int64)
        """Resets déjà effectués par arène : l'épisode en cours est le n° `reset_count - 1`."""

    @property
    def n_envs(self) -> int:
        return self._sim.n_envs

    @property
    def n_cars(self) -> int:
        return self._sim.n_cars

    @property
    def tick_skip(self) -> int:
        return self._sim.tick_skip

    def state(self) -> BatchState:
        return self._sim.state()

    def step(self, actions: I64Array) -> None:
        self._sim.step(actions)

    def scores(self) -> tuple[I64Array, I64Array]:
        return self._sim.scores()

    def env_ticks(self) -> I64Array:
        return self._sim.env_ticks()

    def touch_ticks(self) -> I64Array:
        return self._sim.touch_ticks()

    def reset_to_kickoff(self, arena_idx: I64Array, kickoff_seeds: U64Array) -> None:
        seeds = kickoff_seeds.copy()
        for i, a in enumerate(arena_idx.tolist()):
            p = a + (int(self.reset_count[a]) // 2) * self._sim.n_envs
            if p < self._seeds.shape[0]:
                seeds[i] = self._seeds[p]
            self.reset_count[a] += 1
        self._sim.reset_to_kickoff(arena_idx, seeds)

    def set_states(self, arena_idx: I64Array, states: InitialStates) -> None:
        self._sim.set_states(arena_idx, states)


# ------------------------------------------------------------------------------------------------ jeu


@dataclass(frozen=True)
class MatchResult:
    """Épisodes planifiés, tous terminés. Tableaux `(n_pairs, 2)` : colonne 0 = A bleu, colonne 1 = A orange."""

    outcome: I8Array
    """+1 A gagne (A marque), 0 nul (troncature no-touch ou durée max), -1 B gagne."""
    ticks: I64Array
    """Durée de l'épisode en ticks physiques (120 Hz)."""
    seed: int
    n_pairs: int
    n_envs: int
    tick_skip: int
    env_steps: int
    """Steps d'arène (n_envs par appel de `step`), arènes à vide comprises."""
    wall_s: float

    @property
    def n_episodes(self) -> int:
        return 2 * self.n_pairs

    def outcomes_in_play_order(self) -> I8Array:
        """Résultats aplatis dans l'ordre (paire, côté) : ordre de décision du SPRT, indépendant de l'ordre
        d'achèvement des épisodes."""
        out: I8Array = self.outcome.reshape(-1)
        return out

    def equals(self, other: MatchResult) -> bool:
        """Égalité des résultats (hors durée d'horloge)."""
        return (
            np.array_equal(self.outcome, other.outcome)
            and np.array_equal(self.ticks, other.ticks)
            and self.seed == other.seed
        )


def play_match(
    policy_a: Policy,
    policy_b: Policy,
    sim: Simulator,
    n_episodes: int,
    *,
    seed: int,
    reward_config: RewardConfig | None = None,
    build_obs: ObsBuilder | None = None,
    obs_dim: int | None = None,
) -> MatchResult:
    """Joue ceil(n_episodes / 2) paires. `sim` doit être un simulateur DÉDIÉ à l'évaluation (1v1) : l'env
    d'entraînement n'est ni utilisé ni modifié. `p_kick = 1` : kickoffs seuls."""
    if n_episodes < 2:
        raise ValueError("n_episodes >= 2 (au moins une paire)")
    if sim.n_cars != 2 or sim.n_envs < 1:
        raise ValueError(f"1v1 attendu, reçu {sim.n_envs} arènes x {sim.n_cars} voitures")
    n_pairs = (n_episodes + 1) // 2
    e = sim.n_envs
    paired = _PairedKickoffSim(sim, pair_seeds(seed, n_pairs))
    cfg = reward_config or RewardConfig()
    env = BatchEnv(paired, RewardFunction(cfg), InitialStateSampler(2, 1.0, seed), build_obs=build_obs, obs_dim=obs_dim)
    team = np.asarray(sim.state().car_team)
    blue_slot = int(np.flatnonzero(team == 0)[0])
    arenas = np.arange(e, dtype=np.int64)

    outcome = np.zeros((n_pairs, 2), dtype=np.int8)
    ticks = np.zeros((n_pairs, 2), dtype=np.int64)
    recorded = np.zeros((n_pairs, 2), dtype=bool)
    finished = np.zeros(e, dtype=np.int64)  # épisodes terminés par arène
    max_steps = -(-cfg.max_episode_ticks // sim.tick_skip) + 1
    waves = -(-n_pairs // e)
    step_cap = 2 * waves * max_steps + 2

    t0 = time.perf_counter()
    obs = env.reset(seed)
    n_steps = 0
    while not recorded.all():
        if n_steps >= step_cap:
            raise RuntimeError(f"{n_steps} steps sans achever les épisodes planifiés (plafond théorique {step_cap})")
        # épisode en cours de chaque arène : n° = finished[a] ; A est bleu aux n° pairs, orange aux impairs
        a_slot = np.where(finished % 2 == 0, blue_slot, 1 - blue_slot)
        a_idx = arenas * 2 + a_slot
        b_idx = arenas * 2 + (1 - a_slot)
        act = np.empty(env.n_agents, dtype=np.int64)
        act[a_idx] = _call(policy_a, obs[a_idx])
        act[b_idx] = _call(policy_b, obs[b_idx])
        obs, _, term, trunc, info = env.step(act)
        n_steps += 1
        done = np.flatnonzero((term | trunc)[::2])
        for a in done.tolist():
            n = int(finished[a])
            p = a + (n // 2) * e
            side = n % 2  # 0 : A bleu, 1 : A orange
            finished[a] += 1
            if p >= n_pairs:
                continue  # quota de l'arène atteint : épisode hors plan, ignoré
            if bool(term[2 * a]):
                blue_scored = int(info["goal_net"][a]) == int(GoalNet.ORANGE)
                outcome[p, side] = WIN if blue_scored == (side == 0) else LOSS
            else:
                outcome[p, side] = DRAW
            ticks[p, side] = int(info["episode_ticks"][a])
            recorded[p, side] = True
    return MatchResult(
        outcome=outcome,
        ticks=ticks,
        seed=seed,
        n_pairs=n_pairs,
        n_envs=e,
        tick_skip=sim.tick_skip,
        env_steps=n_steps * e,
        wall_s=time.perf_counter() - t0,
    )


def _call(policy: Policy, obs: F32Array) -> I64Array:
    act = np.asarray(policy(obs))
    if act.shape != (obs.shape[0],) or not np.issubdtype(act.dtype, np.integer):
        raise ValueError(f"politique : ({obs.shape[0]},) entiers attendu, reçu {act.dtype} {act.shape}")
    return act.astype(np.int64, copy=False)


# ------------------------------------------------------------------------------------------------ agrégats


@dataclass(frozen=True)
class Interval:
    value: float | None
    low: float | None
    high: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {"value": self.value, "low": self.low, "high": self.high}


@dataclass(frozen=True)
class MatchSummary:
    n_episodes: int
    n_pairs: int
    wins: int
    draws: int
    losses: int
    score: Interval
    """(victoires + nuls/2) / épisodes."""
    win_rate_blue: float
    """Score de A quand A joue en bleu (pour repérer un avantage de côté ; les paires l'annulent dans `score`)."""
    win_rate_orange: float
    draw_rate: Interval
    elo_decisive: Interval
    """Elo de A sur B sur les parties décisives (nuls exclus, cohérent avec le SPRT) ; None si 0 partie décisive."""
    elo_regularized: bool
    """Vrai si 0 victoire ou 0 défaite : Elo avec 0,5 victoire et 0,5 défaite fictives (borne, pas estimation)."""
    goals_for_per_min: Interval
    goals_against_per_min: Interval
    goal_freq_for: Interval
    """Fréquence de but de A par épisode (victoires / épisodes, nuls compris)."""
    goal_freq_against: Interval
    """Fréquence de but de B par épisode (défaites / épisodes)."""
    n_decisive: int
    """Parties décisives (victoires + défaites)."""
    minutes_played: float
    mean_episode_s: float
    n_boot: int
    alpha: float
    boot_seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_episodes": self.n_episodes,
            "n_pairs": self.n_pairs,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "score": self.score.as_dict(),
            "win_rate_blue": self.win_rate_blue,
            "win_rate_orange": self.win_rate_orange,
            "draw_rate": self.draw_rate.as_dict(),
            "elo_decisive": self.elo_decisive.as_dict(),
            "elo_regularized": self.elo_regularized,
            "goals_for_per_min": self.goals_for_per_min.as_dict(),
            "goals_against_per_min": self.goals_against_per_min.as_dict(),
            "goal_freq_for": self.goal_freq_for.as_dict(),
            "goal_freq_against": self.goal_freq_against.as_dict(),
            "n_decisive": self.n_decisive,
            "minutes_played": self.minutes_played,
            "mean_episode_s": self.mean_episode_s,
            "bootstrap": {"n_boot": self.n_boot, "alpha": self.alpha, "seed": self.boot_seed, "unit": "pair"},
        }


def _ratio(num: npt.NDArray[np.float64], den: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out: npt.NDArray[np.float64] = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
    return out


def _elo_vec(w: npt.NDArray[np.float64], lo: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Elo décisif 400 log10(w / lo) ; NaN sans partie décisive ; si w ou lo vaut 0 : 0,5 fictive de chaque côté."""
    out = np.full(w.shape, np.nan)
    ok = (w + lo) > 0
    pad = np.where((w == 0) | (lo == 0), 0.5, 0.0)
    with np.errstate(all="ignore"):
        out[ok] = (400.0 * np.log10((w + pad) / (lo + pad)))[ok]
    return out


def summarize(res: MatchResult, *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> MatchSummary:
    """Agrégats avec IC percentile (1 - alpha) par bootstrap PAR PAIRES (les deux épisodes d'une paire, corrélés par le
    kickoff commun, restent ensemble), graine fixe. Buts par minute : buts de A (resp. encaissés) / minutes jouées,
    toutes durées d'épisode comprises (nuls inclus). `goal_freq_*` : buts par épisode de chaque camp ; `n_decisive`.

    Lecture de l'Elo « décisif » (`elo_decisive`, `elo_regularized`) :
    - les réplicats bootstrap mélangent, près du bord (peu de victoires ou de défaites), des valeurs régularisées
      (0,5 partie fictive par côté quand un réplicat a 0 victoire ou 0 défaite) et non régularisées ; l'IC près du
      bord est donc approximatif et le drapeau `elo_regularized` ne qualifie que l'estimation ponctuelle ;
    - l'Elo décisif ne distingue pas un agent qui marque rarement d'un agent qui domine : 3 victoires contre 1 défaite
      sur 1 000 épisodes (99,6 % de nuls) donnent le même Elo que 750 contre 250. Il se lit toujours avec la fréquence
      de but (`goal_freq_*`), le taux de nuls et `n_decisive` (cf. `eval.sprt.decisive_win_test`)."""
    out = res.outcome
    n_pairs = res.n_pairs
    w_p = (out == WIN).sum(axis=1).astype(np.float64)
    l_p = (out == LOSS).sum(axis=1).astype(np.float64)
    d_p = (out == DRAW).sum(axis=1).astype(np.float64)
    m_p = res.ticks.sum(axis=1).astype(np.float64) / TICKS_PER_MINUTE
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_pairs, size=(n_boot, n_pairs))

    def stat(w: Any, lo: Any, d: Any, m: Any) -> dict[str, npt.NDArray[np.float64]]:
        n = w + lo + d
        return {
            "score": (w + 0.5 * d) / n,
            "draw": d / n,
            "elo": _elo_vec(w, lo),
            "gf": _ratio(w, m),
            "ga": _ratio(lo, m),
            "gfe": w / n,
            "gae": lo / n,
        }

    point = stat(w_p.sum(keepdims=True), l_p.sum(keepdims=True), d_p.sum(keepdims=True), m_p.sum(keepdims=True))
    boot = stat(w_p[idx].sum(1), l_p[idx].sum(1), d_p[idx].sum(1), m_p[idx].sum(1))

    def interval(key: str) -> Interval:
        v = float(point[key][0])
        b = boot[key][np.isfinite(boot[key])]
        if not np.isfinite(v) or b.size == 0:
            return Interval(None, None, None)
        lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return Interval(v, float(lo), float(hi))

    w, lo_n, d = int(w_p.sum()), int(l_p.sum()), int(d_p.sum())
    n = res.n_episodes
    return MatchSummary(
        n_episodes=n,
        n_pairs=n_pairs,
        wins=w,
        draws=d,
        losses=lo_n,
        score=interval("score"),
        win_rate_blue=float(((out[:, 0] == WIN) + 0.5 * (out[:, 0] == DRAW)).mean()),
        win_rate_orange=float(((out[:, 1] == WIN) + 0.5 * (out[:, 1] == DRAW)).mean()),
        draw_rate=interval("draw"),
        elo_decisive=interval("elo"),
        elo_regularized=(w == 0 or lo_n == 0) and (w + lo_n) > 0,
        goals_for_per_min=interval("gf"),
        goals_against_per_min=interval("ga"),
        goal_freq_for=interval("gfe"),
        goal_freq_against=interval("gae"),
        n_decisive=w + lo_n,
        minutes_played=float(m_p.sum()),
        mean_episode_s=float(res.ticks.mean() / PHYSICS_TICK_RATE_HZ),
        n_boot=n_boot,
        alpha=alpha,
        boot_seed=seed,
    )


def sprt_on_match(res: MatchResult, cfg: SPRTConfig) -> SPRTResult:
    """SPRT rejoué (rétrospectivement) sur les résultats dans l'ordre (paire, côté), évalué aux frontières de paires.
    Le plafond `cfg.max_episodes` s'applique au nombre total d'épisodes. Équivalent à un test en ligne : l'ordre des
    paires ne dépend pas de l'ordre d'achèvement."""
    return sprt_scan(res.outcomes_in_play_order(), cfg, check_every=2)


__all__ = [
    "DRAW",
    "LOSS",
    "TICKS_PER_MINUTE",
    "WIN",
    "Interval",
    "MatchResult",
    "MatchSummary",
    "NetPolicy",
    "Policy",
    "PolicyInfo",
    "RandomPolicy",
    "load_checkpoint_policy",
    "load_policy",
    "pair_seeds",
    "play_match",
    "sprt_on_match",
    "summarize",
]
