"""Self-imitation learning (SIL, Oh et al. 2018, « Self-Imitation Learning », ICML, arXiv:1806.05635) pour la boucle
PPO : intervention (3) de l'ADR-005 D6, réglages de l'ADR-009 (proposé), autorisée en GCO-std (`docs/GCO_SPEC.md` §3),
désactivée par défaut.

Perte (papier §3, éq. 1–3), sur des transitions (s, a, R) rejouées, avec poids d'importance w :
    L_sil = c_sil(t) · moyenne[ w · ( −log π(a|s)·(R − V(s))₊ + β·½·((R − V(s))₊)² ) ],
(R − V)₊ sans gradient dans le terme de politique (poids d'échantillon). Priorités stockées p = ((R − V)₊ + ε)^α,
transition neuve à la priorité maximale courante, mises à jour après chaque pas avec le V du pas (ADR-009 D2 ; Schaul
et al. 2016). Poids w_i = (N·P(i))^(−β_is) / max_j w_j = (p_i / p_min)^(−β_is).

Conditions C1–C4 (spec §3, ADR-009 D6) :
- C1 : R vient uniquement de la récompense ±1 de l'env (`RolloutBatch.rew`), du drapeau `term`, de `learn_mask` et de
  γ = `ppo.gamma`. Une récompense non nulle hors terminal lève une erreur (hypothèse §2.1 violée).
- C2 : c_sil = `coef` jusqu'à `anneal_start_timesteps` appris, linéaire jusqu'à 0 à `anneal_end_timesteps`, puis 0 :
  plus aucun pas ni collecte, buffer et anneau libérés (`released`).
- C3 : `enabled` (désactivé : le learner n'instancie rien, comportement bit à bit identique au PPO seul).
- C4 : journal par itération (`SILStats`, `sil/*`) dont rho = |g_π,sil| / (|g_π,sil| + |g_π,ppo|) (normes avant clip
  sur les paramètres de la politique) ; alerte et arrêts par `sil_guard` (rho, RAM disponible).

Données (ADR-009 D1) :
- épisodes terminés par un but (`term`) de l'agent appris (`learn_mask`), récompense +1 seulement par défaut
  (`keep_conceded=False`) ; les troncatures (no-touch 20 s, 120 s) ne sont JAMAIS admises : leur retour Monte-Carlo
  n'existe pas, le compléter par V(s′) rendrait la cible circulaire, le traiter comme 0 contredirait la spec §2.2 ;
- R_t = γ^(T−1−t)·r_T (la transition du but porte R = r_T), exact pour chaque transition gardée ;
- fenêtre circulaire des `horizon_steps` derniers pas par agent : un épisode plus long perd sa tête, sans biais sur R ;
- obs stockées en float16 par défaut (`store_float16`, parité testée), actions en int16.

Mises à jour (`mode`) :
- `separate` (ADR-009 D3, défaut) : après les epochs PPO, `updates_per_iteration` pas SIL de `batch_size` transitions,
  si le buffer en contient au moins `batch_size`, avec un optimiseur Adam DISTINCT (état PPO jamais modifié), même LR
  que PPO à l'itération et clip global `max_grad_norm` propre ;
- `joint` (option) : un minibatch SIL de `batch_size` par minibatch PPO, gradient ajouté à celui de PPO avant le clip,
  même pas d'optimiseur PPO (aucun optimiseur SIL). Avec un Adam séparé, c_sil n'agit sur le déplacement des poids
  qu'à travers ε (rapport m̂/(√v̂ + ε)) : effet sous-linéaire. Mesuré le 2026-11-19 (CPU, réseaux v0 initiaux, obs
  aléatoires, 20 pas de 8 192, script hors dépôt) : |Δθ| = 0,59 / 0,11 / 0,017 pour c_sil = 0,1 / 0,01 / 0,001.

État (reprise exacte, y compris après perte du GPU : tout est en numpy CPU, l'optimiseur SIL passe par son
`state_dict`) : anneau, buffer (obs, actions, R, priorités, itération d'entrée), générateur, compteurs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch

from models.policy import PolicyNet, log_prob_entropy
from models.value import ValueNet

F32 = npt.NDArray[np.float32]
BOOL = npt.NDArray[np.bool_]
SIL_SEED_OFFSET = 104_729  # flux du tirage SIL distinct de celui des permutations PPO (même seed de learner)
ACTION_DTYPE = np.int16
MODES = ("separate", "joint")


class _Optimizer(Protocol):
    param_groups: list[dict[str, Any]]

    def zero_grad(self, set_to_none: bool = True) -> None: ...

    def step(self) -> object: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


@dataclass(frozen=True, kw_only=True)
class SILConfig:
    """Hyperparamètres SIL. Défauts = ADR-009 (proposé) ; sources : Oh et al. 2018, annexe A, tableau 4 (PPO+SIL) et
    tableau 3 (A2C+SIL)."""

    enabled: bool = False  # C3 : interrupteur ; False = PPO seul, bit à bit
    anneal_start_timesteps: int  # C2 (ADR-009 D4 : 0,5B) : c_sil = coef jusqu'ici (timesteps APPRIS)
    anneal_end_timesteps: int  # C2 (ADR-009 D4 : 0,9B) : c_sil = 0 à partir d'ici
    coef: float = 0.1  # tableau 4 : « SIL loss weight » 0,1 (tableau 3, A2C : 1)
    value_coef: float = 0.01  # β ; tableau 3 : 0,01 ; tableau 4 : {0,01 ; 0,05}
    mode: str = "separate"  # ADR-009 D3 ; "joint" : perte ajoutée aux minibatchs PPO (voir docstring du module)
    updates_per_iteration: int = 2  # ADR-009 D3 (mode separate)
    batch_size: int = 8_192  # ADR-009 D3 (tableaux 3–4 : 512 pour des lots PPO de 2 048)
    max_grad_norm: float = 0.5  # ADR-009 D3 : clip global propre au pas SIL (mode separate)
    capacity: int = 250_000  # ADR-009 D3 : plafond RAM (tableau 4 : 50 000) ; en transitions, éviction FIFO
    horizon_steps: int = 300  # ADR-009 D1 : 20 s à 15 Hz avant le but (écart au papier, qui garde l'épisode)
    priority_alpha: float = 0.6  # tableau 3 : 0,6 ; tableau 4 : {0,6 ; 1,0}
    priority_eps: float = 1.0e-3  # ADR-009 D2
    is_beta: float = 0.1  # tableau 4 : « Bias correction for prioritized replay » 0,1 (convention de Schaul)
    keep_conceded: bool = False  # ADR-009 D1 : épisodes perdants exclus en v1
    store_float16: bool = True  # ADR-009 D1 : sous réserve du test de parité D7-5
    rho_alert: float = 0.5  # ADR-009 D5 (C4) : alerte si rho > 0,5 ...
    rho_alert_patience: int = 50  # ... pendant 50 itérations consécutives
    rho_stop: float = 0.8  # ADR-009 KS2 : STOP si rho > 0,8 ...
    rho_stop_patience: int = 500  # ... pendant 500 itérations consécutives, tant que c_sil > 0
    ram_min_available_mb: float = 1_000.0  # ADR-009 KS1 : STOP si RAM disponible < 1 Go ...
    ram_low_max_s: float = 600.0  # ... pendant plus de 10 min (durée des itérations cumulée)

    def __post_init__(self) -> None:
        checks = {
            "0 <= anneal_start_timesteps <= anneal_end_timesteps": 0
            <= self.anneal_start_timesteps
            <= self.anneal_end_timesteps,
            "coef >= 0": self.coef >= 0.0,
            "value_coef >= 0": self.value_coef >= 0.0,
            f"mode dans {MODES}": self.mode in MODES,
            "updates_per_iteration >= 1": self.updates_per_iteration >= 1,
            "batch_size >= 1": self.batch_size >= 1,
            "max_grad_norm > 0": self.max_grad_norm > 0.0,
            "capacity >= batch_size": self.capacity >= self.batch_size,
            "horizon_steps >= 1": self.horizon_steps >= 1,
            "priority_alpha > 0": self.priority_alpha > 0.0,
            "priority_eps > 0": self.priority_eps > 0.0,
            "0 <= is_beta <= 1": 0.0 <= self.is_beta <= 1.0,
            "0 < rho_alert <= 1 et 0 < rho_stop <= 1": 0.0 < self.rho_alert <= 1.0 and 0.0 < self.rho_stop <= 1.0,
            "patiences >= 1": min(self.rho_alert_patience, self.rho_stop_patience) >= 1,
            "ram_min_available_mb >= 0 et ram_low_max_s >= 0": self.ram_min_available_mb >= 0.0
            and self.ram_low_max_s >= 0.0,
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"SILConfig invalide : {', '.join(bad)}")

    def coef_at(self, learned_timesteps: int) -> float:
        """c_sil pour une mise à jour faite après `learned_timesteps` timesteps appris (ADR-009 D4) : `coef` avant
        `anneal_start_timesteps`, linéaire jusqu'à 0 en `anneal_end_timesteps`, 0 ensuite."""
        t, a, b = learned_timesteps, self.anneal_start_timesteps, self.anneal_end_timesteps
        if t >= b:
            return 0.0
        if t <= a:
            return self.coef
        return self.coef * (b - t) / (b - a)

    @property
    def obs_dtype(self) -> type[np.floating[Any]]:
        return np.float16 if self.store_float16 else np.float32


# ---------------------------------------------------------------------------------------------------- retours


def goal_returns(reward: float, length: int, gamma: float) -> F32:
    """Retours Monte-Carlo des `length` dernières transitions d'un épisode terminé par la récompense terminale `reward`
    (toutes les autres sont nulles, spec §2.1) : R_j = γ^(length−1−j)·reward, de la plus ancienne à la plus récente."""
    if length < 1:
        raise ValueError(f"length={length}")
    k = np.arange(length - 1, -1, -1, dtype=np.float64)
    return (reward * np.power(gamma, k)).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- stockage


class ReplayBuffer:
    """Buffer FIFO priorisé de transitions (obs, action, R, priorité, itération d'entrée), tableaux préalloués."""

    def __init__(self, capacity: int, obs_dim: int, obs_dtype: type[np.floating[Any]] = np.float16) -> None:
        if capacity < 1 or obs_dim < 1:
            raise ValueError(f"capacity={capacity}, obs_dim={obs_dim}")
        self.capacity, self.obs_dim = capacity, obs_dim
        self.obs = np.zeros((capacity, obs_dim), dtype=obs_dtype)
        self.act = np.zeros(capacity, dtype=ACTION_DTYPE)
        self.ret = np.zeros(capacity, dtype=np.float32)
        self.prio = np.zeros(capacity, dtype=np.float32)
        self.born = np.zeros(capacity, dtype=np.int32)
        self.size = 0
        self.pos = 0  # prochaine case écrite

    @property
    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in (self.obs, self.act, self.ret, self.prio, self.born)))

    def max_priority(self) -> float:
        return float(np.max(self.prio[: self.size])) if self.size else 1.0

    def add(self, obs: npt.NDArray[Any], act: npt.NDArray[Any], ret: F32, iteration: int) -> None:
        """Ajoute L transitions dans l'ordre, à la priorité maximale courante (1 si vide) ; au-delà de la capacité,
        les plus anciennes sont écrasées."""
        n = int(ret.shape[0])
        if obs.shape != (n, self.obs_dim) or act.shape != (n,):
            raise ValueError(f"formes incohérentes : obs {obs.shape}, act {act.shape}, ret {ret.shape}")
        if n > self.capacity:
            obs, act, ret, n = obs[-self.capacity :], act[-self.capacity :], ret[-self.capacity :], self.capacity
        p = self.max_priority()
        idx = (self.pos + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.act[idx] = act
        self.ret[idx] = ret
        self.prio[idx] = p
        self.born[idx] = iteration
        self.pos = int((self.pos + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int, generator: torch.Generator, is_beta: float) -> tuple[torch.Tensor, torch.Tensor]:
        """`n` indices tirés avec remise, P(i) = p_i / Σp, et poids w_i = (p_i / p_min)^(−β_is) ≤ 1 (CPU)."""
        if self.size == 0:
            raise ValueError("buffer SIL vide")
        prio = torch.from_numpy(self.prio[: self.size])
        idx = torch.multinomial(prio, n, replacement=True, generator=generator)
        weight = (prio.index_select(0, idx) / prio.min()).pow(-is_beta)
        return idx, weight

    def update_priorities(self, idx: torch.Tensor, gap: torch.Tensor, alpha: float, eps: float) -> None:
        """p_i ← ((R − V)₊ + ε)^α pour les indices tirés (doublons : la dernière valeur l'emporte)."""
        self.prio[idx.numpy()] = (gap.detach().cpu().double().numpy() + eps) ** alpha

    def batch(self, idx: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(obs float32, actions i64, R) des indices `idx` (CPU) sur `device` ; indexation numpy côté CPU."""
        i = idx.numpy()
        obs = torch.from_numpy(np.ascontiguousarray(self.obs[i])).to(device).float()
        act = torch.from_numpy(self.act[i].astype(np.int64)).to(device)
        ret = torch.from_numpy(np.ascontiguousarray(self.ret[i])).to(device)
        return obs, act, ret

    def state_dict(self) -> dict[str, Any]:
        """Vues des cases remplies (à sérialiser aussitôt : `checkpoint.encode` copie)."""
        n = self.size
        return {
            "obs": self.obs[:n],
            "act": self.act[:n],
            "ret": self.ret[:n],
            "prio": self.prio[:n],
            "born": self.born[:n],
            "pos": self.pos,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        n = int(np.asarray(state["ret"]).shape[0])
        shape = np.asarray(state["obs"]).shape
        if n > self.capacity or shape != (n, self.obs_dim):
            raise ValueError(f"buffer SIL : {n} transitions de forme {shape} incompatibles")
        for name in ("obs", "act", "ret", "prio", "born"):
            getattr(self, name)[:n] = state[name]
        self.size, self.pos = n, int(state["pos"])


class EpisodeTracker:
    """Anneau des `horizon` derniers pas (obs, action) de chaque agent appris et longueur de l'épisode en cours vue
    dans l'anneau (≤ horizon). Un curseur commun : tous les agents avancent ensemble."""

    def __init__(self, n_agents: int, obs_dim: int, horizon: int, obs_dtype: type[np.floating[Any]]) -> None:
        if n_agents < 1 or obs_dim < 1 or horizon < 1:
            raise ValueError(f"n_agents={n_agents}, obs_dim={obs_dim}, horizon={horizon}")
        self.n_agents, self.obs_dim, self.horizon = n_agents, obs_dim, horizon
        self.obs = np.zeros((horizon, n_agents, obs_dim), dtype=obs_dtype)
        self.act = np.zeros((horizon, n_agents), dtype=ACTION_DTYPE)
        self.length = np.zeros(n_agents, dtype=np.int64)
        self.cursor = 0

    @property
    def nbytes(self) -> int:
        return int(self.obs.nbytes + self.act.nbytes + self.length.nbytes)

    def step(
        self, obs: npt.NDArray[Any], act: npt.NDArray[Any], rew: F32, term: BOOL, keep_conceded: bool
    ) -> list[tuple[int, float, int]]:
        """Enregistre un pas (formes (n_agents, …)) ; rend les épisodes à stocker, (agent, récompense, longueur), à lire
        par `segment` avant `end_step`."""
        if np.any((rew != 0.0) & ~term):
            raise ValueError("récompense non nulle hors terminal : la SIL suppose la récompense GCO (spec §2.1)")
        c = self.cursor
        self.obs[c] = obs
        self.act[c] = act
        np.minimum(self.length + 1, self.horizon, out=self.length)
        keep = term & ((rew > 0.0) | ((rew < 0.0) & keep_conceded))
        return [(int(a), float(rew[a]), int(self.length[a])) for a in np.flatnonzero(keep)]

    def segment(self, agent: int, length: int) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Les `length` derniers pas de `agent`, du plus ancien au plus récent (curseur courant)."""
        idx = (self.cursor - np.arange(length - 1, -1, -1)) % self.horizon
        return self.obs[idx, agent], self.act[idx, agent]

    def end_step(self, done: BOOL) -> None:
        """Fin d'épisode (term ou trunc) : longueur remise à 0 (le pas suivant ouvre un nouvel épisode)."""
        self.length[done] = 0
        self.cursor = (self.cursor + 1) % self.horizon

    def state_dict(self) -> dict[str, Any]:
        return {"obs": self.obs, "act": self.act, "length": self.length, "cursor": self.cursor}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if np.asarray(state["obs"]).shape != self.obs.shape:
            raise ValueError(f"anneau SIL {np.asarray(state['obs']).shape} != {self.obs.shape}")
        self.obs[...] = state["obs"]
        self.act[...] = state["act"]
        self.length[...] = state["length"]
        self.cursor = int(state["cursor"])


# ---------------------------------------------------------------------------------------------------- journal C4


@dataclass(frozen=True)
class SILStats:
    """Journal C4 d'une itération (`sil/*`, ADR-009 D5). Jamais de NaN : une SIL sans pas (buffer trop petit, c_sil
    nul, libérée) écrit des zéros, pour ne pas déclencher la garde des valeurs non finies."""

    coef: float
    steps_done: float  # pas SIL appliqués
    steps_inactive: float  # minibatchs tirés sans aucun (R − V)₊ > 0 : pas sauté (mode separate)
    released: float  # 1.0 après la fin de l'annealing (buffer et anneau libérés)
    buffer_size: float
    buffer_new: float  # transitions ajoutées par ce lot
    buffer_age_mean: float  # âge moyen des transitions du buffer, en itérations
    buffer_bytes: float
    tracker_bytes: float
    episodes_added: float
    episodes_total: float
    loss_policy: float  # moyenne sur les pas SIL, avant coefficient
    loss_value: float
    frac_pos: float  # part des transitions tirées avec (R − V)₊ > 0
    adv_pos_mean: float  # moyenne de (R − V)₊ sur les transitions tirées
    adv_pos_max: float
    ppo_adv_abs_raw: float  # |A| PPO brut moyen (première epoch), même unité que (R − V)₊
    grad_norm_pi_sil: float  # norme du gradient SIL (c_sil compris) sur π, avant clip, moyenne des pas SIL
    grad_norm_pi_ppo: float  # norme du gradient PPO sur π, avant clip, moyenne des minibatchs PPO
    rho: float  # grad_norm_pi_sil / (grad_norm_pi_sil + grad_norm_pi_ppo) ; 0 sans pas SIL

    def as_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}


# ---------------------------------------------------------------------------------------------------- perte


def sil_losses(
    policy: PolicyNet,
    value: ValueNet,
    obs: torch.Tensor,
    act: torch.Tensor,
    ret: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(L_policy, L_value, (R − V)₊ par échantillon, détaché), papier éq. 2–3 avec poids d'importance `weight`. Le
    poids (R − V)₊ du terme de politique est détaché : seul le terme de valeur entraîne V. Gradient nul si R ≤ V."""
    v = value(obs)
    gap = (ret - v).clamp(min=0.0)
    logp, _ = log_prob_entropy(policy(obs), act)
    policy_loss = -(logp * weight * gap.detach()).mean()
    value_loss = (weight * 0.5 * gap.pow(2)).mean()
    return policy_loss, value_loss, gap.detach()


def _grad_norm(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    norms = [p.grad.detach().norm() for p in params if p.grad is not None]
    return torch.stack(norms).norm() if norms else torch.zeros(())


# ---------------------------------------------------------------------------------------------------- SIL


class SelfImitation:
    """État et calculs SIL d'un learner : suivi des épisodes, buffer priorisé, tirage, optimiseur SIL, compteurs."""

    def __init__(
        self,
        cfg: SILConfig,
        gamma: float,
        obs_dim: int,
        n_actions: int,
        seed: int,
        optimizer: _Optimizer | None = None,
    ) -> None:
        if n_actions > np.iinfo(ACTION_DTYPE).max:
            raise ValueError(f"n_actions={n_actions} > {np.iinfo(ACTION_DTYPE).max} (actions stockées en int16)")
        if (cfg.mode == "separate") != (optimizer is not None):
            raise ValueError("mode separate : un optimiseur SIL est requis ; mode joint : aucun")
        self.cfg, self.gamma, self.obs_dim = cfg, gamma, obs_dim
        self.optimizer = optimizer
        self.buffer: ReplayBuffer | None = ReplayBuffer(cfg.capacity, obs_dim, cfg.obs_dtype)
        self.tracker: EpisodeTracker | None = None  # créé au premier lot (nombre d'agents appris)
        self.cols: npt.NDArray[np.int64] | None = None
        self.generator = torch.Generator()
        self.generator.manual_seed(seed + SIL_SEED_OFFSET)
        self.episodes_total = 0
        self.released = False

    def reseed(self, seed: int) -> None:
        self.generator.manual_seed(seed + SIL_SEED_OFFSET)

    @property
    def nbytes(self) -> int:
        """Octets des tableaux SIL (buffer + anneau), hors tenseurs d'une mise à jour et état de l'optimiseur."""
        b = 0 if self.buffer is None else self.buffer.nbytes
        return b + (0 if self.tracker is None else self.tracker.nbytes)

    def release(self) -> None:
        """Fin de l'annealing (C2) : plus de collecte ni de pas, mémoire rendue."""
        self.buffer, self.tracker, self.released = None, None, True

    def ingest(
        self,
        obs: F32,
        act: npt.NDArray[np.int64],
        rew: F32,
        term: BOOL,
        trunc: BOOL,
        learn_mask: BOOL,
        iteration: int,
    ) -> tuple[int, int]:
        """Lot (T, N, …) dans l'ordre de collecte : met à jour l'anneau et stocke les épisodes admis (agents de
        `learn_mask`). Rend (épisodes ajoutés, transitions ajoutées)."""
        if self.buffer is None:
            return 0, 0
        cols = np.flatnonzero(learn_mask)
        if self.tracker is None:
            self.tracker = EpisodeTracker(cols.size, self.obs_dim, self.cfg.horizon_steps, self.cfg.obs_dtype)
            self.cols = cols
        elif self.cols is None or not np.array_equal(cols, self.cols):
            raise ValueError("learn_mask a changé en cours de run : suivi des épisodes SIL incohérent")
        tr = self.tracker
        n_ep = n_tr = 0
        for t in range(obs.shape[0]):
            term_t = term[t, cols]
            ends = tr.step(obs[t, cols], act[t, cols], rew[t, cols], term_t, self.cfg.keep_conceded)
            for agent, reward, length in ends:
                o, a = tr.segment(agent, length)
                self.buffer.add(o, a, goal_returns(reward, length, self.gamma), iteration)
                n_ep, n_tr = n_ep + 1, n_tr + length
            tr.end_step(term_t | trunc[t, cols])
        self.episodes_total += n_ep
        return n_ep, n_tr

    def begin(
        self,
        obs: F32,
        act: npt.NDArray[np.int64],
        rew: F32,
        term: BOOL,
        trunc: BOOL,
        learn_mask: BOOL,
        *,
        iteration: int,
        learned_per_iteration: int,
        device: torch.device,
        n_ppo_steps: int,
    ) -> SILIteration:
        """Début d'une mise à jour : coefficient (timesteps appris AVANT cette mise à jour), libération en fin
        d'annealing, puis ingestion du lot AVANT tout pas (algorithme 1 du papier)."""
        coef = self.cfg.coef_at(iteration * learned_per_iteration)
        if coef == 0.0 and iteration * learned_per_iteration >= self.cfg.anneal_end_timesteps and not self.released:
            self.release()
        n_ep, n_tr = self.ingest(obs, act, rew, term, trunc, learn_mask, iteration)
        return SILIteration(self, coef, iteration, n_ep, n_tr, device, n_ppo_steps)

    def state_dict(self) -> dict[str, Any]:
        opt = self.optimizer
        return {
            "released": self.released,
            "buffer": None if self.buffer is None else self.buffer.state_dict(),
            "tracker": None if self.tracker is None else self.tracker.state_dict(),
            "cols": self.cols,
            "generator": self.generator.get_state(),
            "episodes_total": self.episodes_total,
            "optimizer": None if opt is None else opt.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if bool(state["released"]):
            self.release()
        else:
            self.released = False
            self.buffer = ReplayBuffer(self.cfg.capacity, self.obs_dim, self.cfg.obs_dtype)
            self.buffer.load_state_dict(state["buffer"])
        if state["tracker"] is None:
            self.tracker, self.cols = None, None
        else:
            cols = np.asarray(state["cols"], dtype=np.int64)
            self.tracker = EpisodeTracker(cols.size, self.obs_dim, self.cfg.horizon_steps, self.cfg.obs_dtype)
            self.tracker.load_state_dict(state["tracker"])
            self.cols = cols
        if (self.optimizer is None) != (state["optimizer"] is None):
            raise ValueError("état d'optimiseur SIL présent XOR mode separate : incohérent")
        if self.optimizer is not None:
            self.optimizer.load_state_dict(dict(state["optimizer"]))
        self.generator.set_state(state["generator"].cpu())
        self.episodes_total = int(state["episodes_total"])


class SILIteration:
    """SIL d'une mise à jour PPO. Appels de `Learner.update`, dans l'ordre : `note_raw_advantages` (première epoch),
    pour chaque minibatch PPO `note_ppo_grad` puis (mode joint) `joint_step` entre le `backward` PPO et le clip,
    enfin (mode separate) `separate_steps` après les epochs, puis `stats`."""

    def __init__(
        self,
        sil: SelfImitation,
        coef: float,
        iteration: int,
        n_ep: int,
        n_tr: int,
        device: torch.device,
        n_ppo_steps: int,
    ) -> None:
        self.sil, self.cfg, self.coef, self.iteration = sil, sil.cfg, coef, iteration
        self.n_ep, self.n_tr, self.device, self.n_ppo_steps = n_ep, n_tr, device, n_ppo_steps
        self.steps = 0  # minibatchs SIL tirés (pas appliqués + pas sautés faute d'échantillon actif)
        self.inactive = 0
        self._ppo_grad = torch.zeros((), device=device)
        self._adv_raw = torch.zeros((), device=device)
        self._acc = np.zeros(6)  # loss_policy, loss_value, frac_pos, adv_pos_mean, |g_π,sil|, Σ (pour la moyenne)
        self._adv_max = 0.0

    @property
    def can_step(self) -> bool:
        b = self.sil.buffer
        return self.coef > 0.0 and b is not None and b.size >= self.cfg.batch_size

    def note_raw_advantages(self, adv: torch.Tensor) -> None:
        self._adv_raw = adv.detach().abs().mean()

    def note_ppo_grad(self, policy_params: Sequence[torch.nn.Parameter]) -> None:
        """Norme du gradient PPO sur π (avant toute contribution SIL et avant le clip)."""
        self._ppo_grad = self._ppo_grad + _grad_norm(policy_params).detach()

    def _loss(
        self, policy: PolicyNet, value: ValueNet
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        buf = self.sil.buffer
        assert buf is not None
        idx, weight = buf.sample(self.cfg.batch_size, self.sil.generator, self.cfg.is_beta)
        obs, act, ret = buf.batch(idx, self.device)
        pl, vl, gap = sil_losses(policy, value, obs, act, ret, weight.to(self.device))
        return idx, pl, vl, gap, self.coef * (pl + self.cfg.value_coef * vl)

    def _record(
        self, idx: torch.Tensor, pl: torch.Tensor, vl: torch.Tensor, gap: torch.Tensor, g: float, applied: bool = True
    ) -> None:
        gap_cpu = gap.cpu()  # une lecture hôte par pas SIL (mise à jour des priorités)
        buf = self.sil.buffer
        assert buf is not None
        buf.update_priorities(idx, gap_cpu, self.cfg.priority_alpha, self.cfg.priority_eps)
        self._acc += np.array(
            [float(pl), float(vl), float((gap_cpu > 0).float().mean()), float(gap_cpu.mean()), g, 1.0]
        )
        self._adv_max = max(self._adv_max, float(gap_cpu.max()))
        self.steps += 1
        self.inactive += 0 if applied else 1

    def joint_step(self, policy: PolicyNet, value: ValueNet, policy_params: Sequence[torch.nn.Parameter]) -> None:
        """Mode joint : ajoute le gradient de c_sil·L_sil aux gradients PPO du minibatch (rien si `can_step` faux)."""
        if self.cfg.mode != "joint" or not self.can_step:
            return
        with torch.no_grad():
            saved = [None if p.grad is None else p.grad.detach().clone() for p in policy_params]
        idx, pl, vl, gap, loss = self._loss(policy, value)
        torch.autograd.backward([loss])
        with torch.no_grad():
            diffs = [
                (p.grad - g if g is not None else p.grad).norm()
                for p, g in zip(policy_params, saved, strict=True)
                if p.grad is not None
            ]
            g_sil = float(torch.stack(diffs).norm()) if diffs else 0.0
        self._record(idx, pl.detach(), vl.detach(), gap, g_sil)

    def separate_steps(
        self,
        policy: PolicyNet,
        value: ValueNet,
        params: Sequence[torch.nn.Parameter],
        policy_params: Sequence[torch.nn.Parameter],
        lr: float,
    ) -> None:
        """Mode separate : `updates_per_iteration` pas de l'optimiseur SIL (rien si `can_step` faux)."""
        opt = self.sil.optimizer
        if self.cfg.mode != "separate" or opt is None or not self.can_step:
            return
        for group in opt.param_groups:
            group["lr"] = lr
        for _ in range(self.cfg.updates_per_iteration):
            idx, pl, vl, gap, loss = self._loss(policy, value)
            if not bool((gap > 0.0).any()):
                # R ≤ V partout : gradient nul ; pas d'Adam (son élan déplacerait quand même les poids)
                self._record(idx, pl.detach(), vl.detach(), gap, 0.0, applied=False)
                continue
            opt.zero_grad(set_to_none=True)
            torch.autograd.backward([loss])
            g_sil = float(_grad_norm(policy_params))
            torch.nn.utils.clip_grad_norm_(list(params), self.cfg.max_grad_norm)
            opt.step()
            self._record(idx, pl.detach(), vl.detach(), gap, g_sil)

    def stats(self) -> SILStats:
        sil, buf, k = self.sil, self.sil.buffer, max(self.steps, 1)
        pl, vl, frac_pos, adv_mean, g_sil = (float(x) / k for x in self._acc[:5])
        g_ppo = float(self._ppo_grad) / max(self.n_ppo_steps, 1)
        rho = g_sil / (g_sil + g_ppo) if self.steps and g_sil + g_ppo > 0.0 else 0.0
        size = 0 if buf is None else buf.size
        age = float(self.iteration - buf.born[:size].mean()) if buf is not None and size else 0.0
        return SILStats(
            coef=self.coef,
            steps_done=float(self.steps - self.inactive),
            steps_inactive=float(self.inactive),
            released=float(sil.released),
            buffer_size=float(size),
            buffer_new=float(self.n_tr),
            buffer_age_mean=age,
            buffer_bytes=float(0 if buf is None else buf.nbytes),
            tracker_bytes=float(0 if sil.tracker is None else sil.tracker.nbytes),
            episodes_added=float(self.n_ep),
            episodes_total=float(sil.episodes_total),
            loss_policy=pl,
            loss_value=vl,
            frac_pos=frac_pos,
            adv_pos_mean=adv_mean,
            adv_pos_max=self._adv_max,
            ppo_adv_abs_raw=float(self._adv_raw),
            grad_norm_pi_sil=g_sil,
            grad_norm_pi_ppo=g_ppo,
            rho=rho,
        )


# ---------------------------------------------------------------------------------------------------- garde C4


@dataclass(frozen=True)
class SILGuardState:
    """Compteurs de la garde SIL (sauvegardés dans `extra["sil_guard"]` des checkpoints)."""

    rho_alert_streak: int = 0
    rho_stop_streak: int = 0
    ram_low_s: float = 0.0


def sil_guard(
    cfg: SILConfig,
    state: SILGuardState,
    *,
    rho: float,
    coef: float,
    ram_available_mb: float | None,
    iteration_s: float,
) -> tuple[SILGuardState, tuple[str, ...], tuple[str, str] | None]:
    """Fonction pure (ADR-009 D5, KS1, KS2) : (nouvel état, alertes « règle : détail », arrêt (règle, raison) ou None).
    - alerte `sil_rho` : rho > `rho_alert` pendant `rho_alert_patience` itérations consécutives ;
    - STOP `sil_rho` (KS2) : rho > `rho_stop` pendant `rho_stop_patience` itérations consécutives avec c_sil > 0 ;
    - STOP `sil_ram` (KS1) : RAM disponible < `ram_min_available_mb` pendant plus de `ram_low_max_s` secondes
      (durées d'itération cumulées, remises à 0 dès une itération au-dessus du seuil ; mesure absente : inchangé)."""
    a = state.rho_alert_streak + 1 if rho > cfg.rho_alert else 0
    s = state.rho_stop_streak + 1 if rho > cfg.rho_stop and coef > 0.0 else 0
    low = state.ram_low_s
    if ram_available_mb is not None:
        low = low + iteration_s if ram_available_mb < cfg.ram_min_available_mb else 0.0
    new = SILGuardState(rho_alert_streak=a, rho_stop_streak=s, ram_low_s=low)
    alerts: tuple[str, ...] = ()
    if a >= cfg.rho_alert_patience:
        alerts = (f"sil_rho : rho {rho:.3g} > {cfg.rho_alert:g} depuis {a} itérations (C4, ADR-009 D5)",)
    stop: tuple[str, str] | None = None
    if s >= cfg.rho_stop_patience:
        stop = ("sil_rho", f"rho {rho:.3g} > {cfg.rho_stop:g} depuis {s} itérations avec c_sil > 0 (ADR-009 KS2)")
    elif low > cfg.ram_low_max_s:
        stop = (
            "sil_ram",
            f"RAM disponible {ram_available_mb} Mo < {cfg.ram_min_available_mb:g} Mo depuis {low:.0f} s (ADR-009 KS1)",
        )
    return new, alerts, stop
