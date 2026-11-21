"""Tâche auxiliaire « prochain but » (ADR-010, proposé) : intervention (4) de `docs/GCO_SPEC.md` §8, autorisée en
GCO-std (spec §3, ligne « Tâches auxiliaires » : perte de représentation, pas récompense), désactivée par défaut.

Cible (D1, C1) : pour l'agent appris i au pas t, y_t = +1 si l'équipe de i marque, −1 si elle encaisse (csc compris :
c'est le signe de r(i) au pas du but, spec §2.1, aucune autre lecture), 0 si aucun but dans les `horizon_steps` (H)
pas [t, t + H). Un pas sort de la fenêtre avec y = 0 dès que H pas sont passés sans fin d'épisode ; une fin par but
(`term`) étiquette tous les pas encore dans la fenêtre avec le signe de r(i) (0 si r(i) = 0) ; une TRONCATURE
(`trunc` sans `term` : no-touch 20 s, 120 s) les masque (futur inconnu, spec §2.2) : jamais étiquetés, comptés dans
`frac_masked`. Agents de `learn_mask` seulement ; l'origine de l'épisode n'est ni stockée ni lue. Une récompense non
nulle hors terminal lève une erreur (hypothèse §2.1 violée : la cible n'aurait plus de sens).

Tête (D3) : linéaire `hidden` → 3 sur la sortie de la dernière couche cachée de π (`policy.net[:-1]`, après ReLU ;
`nn.Sequential.__getitem__` avec une tranche rend un `Sequential` qui PARTAGE les modules, vérifié dans
`torch/nn/modules/container.py`, torch 2.15.0). Module séparé : hors du `state_dict` de π (export, `eval.match` et
RLBot inchangés), invisible du critique. Init orthogonale gain 0,01 (convention de la tête de π) tirée d'un
générateur propre (`torch.nn.init.orthogonal_(generator=...)`, torch 2.15.0) : l'état du générateur global n'est pas
touché, θ₀ de π reste celui de std.

Perte et optimisation (D3, D4) : entropie croisée à 3 classes (UNREAL §3.2), à CHAQUE minibatch PPO, gradient de
λ·L_aux ajouté à celui de PPO AVANT le clip global (la tête y est incluse), même Adam, même LR ; aucun pas
d'optimiseur ajouté. La perte n'atteint ni la couche de sortie de π, ni V, ni la récompense, ni les retours, ni les
avantages (C2, ADR-010 D6 ; test). Classes : indice 0 = −1 (encaissé), 1 = 0 (aucun but), 2 = +1 (marqué).

Rejeu (D3) : deux compartiments FIFO de `capacity_per_class` transitions (y = 0 ; y ≠ 0), obs en float16 par défaut.
Lot de `batch_size` = moitié tirée uniformément AVEC remise dans chaque compartiment (rééquilibrage 50/50, UNREAL
§3.2–3.3) ; aucun lot tant qu'un compartiment a moins de `batch_size / 2` transitions (journalisé). Les indices des
lots de l'itération sont tirés au début de la mise à jour (générateur CPU propre) et les obs copiées vers l'appareil en
une fois ; les statistiques restent sur l'appareil et sont lues une fois par itération (D4, ADR-001 M4).

λ (D5, D6) : `coef`, constant par défaut. Calendrier optionnel (lecture littérale de C2, repli de D6) :
`anneal_start_timesteps` et `anneal_end_timesteps` (timesteps APPRIS) : λ plein jusqu'au premier, linéaire jusqu'à 0
au second, puis 0 ; à 0 après la fin du calendrier, fenêtre et rejeu sont libérés (`released`). Les deux absents
(None) : λ constant (ce que propose l'ADR).

Parts (D5, C4), sur les paramètres du TRONC de π seulement : `grad_share` = ‖g_aux‖ / (‖g_aux‖ + ‖g_PPO‖), normes avant
clip moyennées sur les minibatchs avec lot (indicateur de l'alerte C4) ; `disp_share` = part de déplacement (méthode
d'ADR-009 D5, moments fantômes d'Adam lus par `training.sil.AdamView`) : seul m^aux est tenu (m^aux_k = β₁·m^aux_(k−1)
+ (1 − β₁)·f_k·g^aux_k, f_k facteur du clip global), m^ppo = `exp_avg` − m^aux ; Δθ^x_k = −(lr / (1 − β₁^t))·m^x_k /
(√v_k / √(1 − β₂^t) + ε) ; D^x = ‖Σ_k Δθ^x_k‖ et disp_share = D^aux / (D^aux + D^ppo). Diagnostic, jamais critère.

État (reprise exacte) : fenêtre, rejeu, tête, générateur, colonnes apprises, moment fantôme, compteurs ; l'état Adam
de la tête est dans celui de l'optimiseur PPO.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from models.policy import POLICY_HEAD_GAIN, PolicyNet
from training.sil import CLIP_EPS, AdamView

F32 = npt.NDArray[np.float32]
BOOL = npt.NDArray[np.bool_]
N_CLASSES = 3
NEG, ZERO, POS = 0, 1, 2  # indices de classe : y + 1
AUX_SEED_OFFSET = 130_363  # flux du tirage du rejeu, distinct de PPO, SIL (104 729) et pool (7 919)
HEAD_SEED_OFFSET = 15_485_863  # flux de l'init de la tête (jamais le générateur global)
LABEL_DTYPE = np.int8
BORN_DTYPE = np.int32


@dataclass(frozen=True, kw_only=True)
class AuxConfig:
    """Hyperparamètres de la tâche auxiliaire. Défauts = ADR-010 (proposé) ; sources en commentaire."""

    enabled: bool = False  # C3 : False = PPO seul, bit à bit ; refusé en GCO-S (spec §3)
    coef: float = 0.1  # λ_aux (D5) : provisoire, fixé par les bancs D5 (descente 0,1 → 0,0316 → 0,01)
    horizon_steps: int = 150  # D2 : H = 150 pas (10 s à 15 Hz), règle RAM d'ADR-009 D7
    batch_size: int = 8_192  # D3 : 4 096 + 4 096 par minibatch PPO
    capacity_per_class: int = 125_000  # D3 : deux compartiments FIFO
    store_float16: bool = True  # D3 : sous réserve de la parité D8-5 (sinon float32 et H = 75)
    share_alert: float = 0.5  # D6 (C4) : alerte si grad_share > 0,5 ...
    share_alert_patience: int = 50  # ... pendant 50 itérations consécutives
    ram_min_available_mb: float = 1_000.0  # KS1 : STOP si RAM disponible < 1 Go ...
    ram_low_max_s: float = 600.0  # ... pendant plus de 10 min (durée des itérations cumulée)
    anneal_start_timesteps: int | None = None  # D6, repli : λ plein jusqu'ici (timesteps APPRIS) ; None : constant
    anneal_end_timesteps: int | None = None  # ... linéaire jusqu'à 0 ici, puis 0 (fenêtre et rejeu libérés)

    def __post_init__(self) -> None:
        a, b = self.anneal_start_timesteps, self.anneal_end_timesteps
        checks = {
            "coef >= 0": self.coef >= 0.0,
            "horizon_steps >= 1": self.horizon_steps >= 1,
            "batch_size pair et >= 2 (moitié par compartiment)": self.batch_size >= 2 and self.batch_size % 2 == 0,
            "capacity_per_class >= batch_size / 2": self.capacity_per_class >= self.batch_size // 2,
            "0 < share_alert <= 1": 0.0 < self.share_alert <= 1.0,
            "share_alert_patience >= 1": self.share_alert_patience >= 1,
            "ram_min_available_mb >= 0 et ram_low_max_s >= 0": self.ram_min_available_mb >= 0.0
            and self.ram_low_max_s >= 0.0,
            "anneal_start_timesteps et anneal_end_timesteps : tous deux absents, ou 0 <= début <= fin": (
                a is None and b is None
            )
            or (a is not None and b is not None and 0 <= a <= b),
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"AuxConfig invalide : {', '.join(bad)}")

    @property
    def half(self) -> int:
        return self.batch_size // 2

    @property
    def annealed(self) -> bool:
        return self.anneal_end_timesteps is not None

    def coef_at(self, learned_timesteps: int) -> float:
        """λ pour une mise à jour faite après `learned_timesteps` timesteps appris : `coef` sans calendrier ; sinon
        `coef` jusqu'à `anneal_start_timesteps`, linéaire jusqu'à 0 en `anneal_end_timesteps`, 0 ensuite."""
        a, b = self.anneal_start_timesteps, self.anneal_end_timesteps
        if a is None or b is None:
            return self.coef
        t = learned_timesteps
        if t >= b:
            return 0.0
        if t <= a:
            return self.coef
        return self.coef * (b - t) / (b - a)

    @property
    def obs_dtype(self) -> type[np.floating[Any]]:
        return np.float16 if self.store_float16 else np.float32


# ---------------------------------------------------------------------------------------------------- tête


class AuxHead(nn.Module):
    """Tête linéaire `n_in` → 3. Paramètres créés vides (aucun tirage) puis initialisés par `init_from`."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        if n_in < 1:
            raise ValueError(f"n_in={n_in}")
        self.weight = nn.Parameter(torch.empty(N_CLASSES, n_in))
        self.bias = nn.Parameter(torch.zeros(N_CLASSES))

    def init_from(self, generator: torch.Generator, gain: float = POLICY_HEAD_GAIN) -> None:
        with torch.no_grad():
            nn.init.orthogonal_(self.weight, gain=gain, generator=generator)
            self.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(features, self.weight, self.bias)


def policy_trunk(policy: PolicyNet) -> nn.Sequential:
    """Tronc de π : toutes les couches sauf la sortie (`policy.net[:-1]`, modules partagés avec π)."""
    trunk = policy.net[:-1]
    if not isinstance(trunk, nn.Sequential) or len(trunk) == 0:
        raise TypeError("politique sans couche cachée : pas de tronc pour la tâche auxiliaire")
    return trunk


def make_aux_head(policy: PolicyNet, seed: int) -> AuxHead:
    """Tête de la largeur de la dernière couche cachée de π, sur le device de π, init par un générateur propre (le
    générateur global n'est pas consommé : θ₀ et la trajectoire du reste de l'assemblage restent ceux de std)."""
    out = policy.net[-1]
    if not isinstance(out, nn.Linear):
        raise TypeError(f"sortie de π {type(out).__name__} : nn.Linear attendu")
    head = AuxHead(out.in_features)
    g = torch.Generator()
    g.manual_seed(seed + HEAD_SEED_OFFSET)
    head.init_from(g)
    return head.to(out.weight.device)


def aux_logits(trunk: nn.Module, head: AuxHead, obs: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = head(trunk(obs))
    return out


# ---------------------------------------------------------------------------------------------------- étiquetage


@dataclass(frozen=True)
class Exits:
    """Pas sortis de la fenêtre lors d'un pas de collecte : obs étiquetées 0, obs ± et leurs étiquettes, effectifs."""

    zero_obs: npt.NDArray[Any]
    pos_obs: npt.NDArray[Any]
    pos_label: npt.NDArray[np.int8]
    n_masked: int

    @property
    def n_pos(self) -> int:
        return int((self.pos_label > 0).sum())

    @property
    def n_neg(self) -> int:
        return int((self.pos_label < 0).sum())

    @property
    def n_zero(self) -> int:
        return int(self.zero_obs.shape[0])


class GoalWindow:
    """Fenêtre glissante des `horizon` derniers pas (obs) de chaque agent appris, non encore étiquetés, et leur
    nombre par agent (≤ horizon). Curseur commun : tous les agents avancent ensemble. Étiquetage vectorisé (numpy),
    aucun appel par agent en Python (D4)."""

    def __init__(self, n_agents: int, obs_dim: int, horizon: int, obs_dtype: type[np.floating[Any]]) -> None:
        if n_agents < 1 or obs_dim < 1 or horizon < 1:
            raise ValueError(f"n_agents={n_agents}, obs_dim={obs_dim}, horizon={horizon}")
        self.n_agents, self.obs_dim, self.horizon = n_agents, obs_dim, horizon
        self.obs = np.zeros((horizon, n_agents, obs_dim), dtype=obs_dtype)
        self.length = np.zeros(n_agents, dtype=np.int64)
        self.cursor = 0

    @property
    def nbytes(self) -> int:
        return int(self.obs.nbytes + self.length.nbytes)

    def step(self, obs: npt.NDArray[Any], rew: F32, term: BOOL, trunc: BOOL) -> Exits:
        """Un pas de collecte (formes (n_agents, …)) : (1) le pas le plus ancien des agents à fenêtre pleine sort
        avec y = 0 (H pas sans fin) ; (2) le pas courant entre ; (3) à une fin d'épisode, tous les pas de la fenêtre
        sortent : signe de r(i) si `term`, masqués si troncature seule."""
        if np.any((rew != 0.0) & ~term):
            raise ValueError("récompense non nulle hors terminal : la tâche auxiliaire suppose la récompense GCO §2.1")
        h, c = self.horizon, self.cursor
        full = np.flatnonzero(self.length == h)
        zero_old = self.obs[c, full]  # copie (indexation avancée) avant l'écrasement de la case
        self.obs[c] = obs
        np.minimum(self.length + 1, h, out=self.length)
        ends = np.flatnonzero(term | trunc)
        lengths = self.length[ends]
        total = int(lengths.sum())
        agents = np.repeat(ends, lengths)
        starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
        # rang depuis le pas courant (0) : du plus ancien au plus récent pour chaque agent qui finit
        back = np.repeat(lengths, lengths) - 1 - (np.arange(total, dtype=np.int64) - starts)
        seg_obs = self.obs[(c - back) % h, agents]
        is_term = np.repeat(term[ends], lengths)
        sign = np.repeat(np.sign(rew[ends]).astype(LABEL_DTYPE), lengths)
        nonzero = is_term & (sign != 0)
        zero_end = is_term & (sign == 0)
        self.length[ends] = 0
        self.cursor = (c + 1) % h
        zero_obs = np.concatenate([zero_old, seg_obs[zero_end]]) if zero_end.any() else zero_old
        return Exits(zero_obs, seg_obs[nonzero], sign[nonzero], int((~is_term).sum()))

    def state_dict(self) -> dict[str, Any]:
        return {"obs": self.obs, "length": self.length, "cursor": self.cursor}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if np.asarray(state["obs"]).shape != self.obs.shape:
            raise ValueError(f"fenêtre auxiliaire {np.asarray(state['obs']).shape} != {self.obs.shape}")
        self.obs[...] = state["obs"]
        self.length[...] = state["length"]
        self.cursor = int(state["cursor"])


# ---------------------------------------------------------------------------------------------------- rejeu


class ClassReplay:
    """Compartiment FIFO préalloué : obs, étiquette (int8, −1/0/+1), itération d'entrée (int32)."""

    _FIELDS = ("obs", "label", "born")

    def __init__(self, capacity: int, obs_dim: int, obs_dtype: type[np.floating[Any]] = np.float16) -> None:
        if capacity < 1 or obs_dim < 1:
            raise ValueError(f"capacity={capacity}, obs_dim={obs_dim}")
        self.capacity, self.obs_dim = capacity, obs_dim
        self.obs = np.zeros((capacity, obs_dim), dtype=obs_dtype)
        self.label = np.zeros(capacity, dtype=LABEL_DTYPE)
        self.born = np.zeros(capacity, dtype=BORN_DTYPE)
        self.size = 0
        self.pos = 0  # prochaine case écrite

    @property
    def nbytes(self) -> int:
        return int(sum(getattr(self, a).nbytes for a in self._FIELDS))

    def add(self, obs: npt.NDArray[Any], label: npt.NDArray[Any], iteration: int) -> None:
        """Ajoute n transitions dans l'ordre ; au-delà de la capacité, les plus anciennes sont écrasées."""
        n = int(label.shape[0])
        if obs.shape != (n, self.obs_dim):
            raise ValueError(f"formes incohérentes : obs {obs.shape}, label {label.shape}")
        if n == 0:
            return
        if n > self.capacity:
            obs, label, n = obs[-self.capacity :], label[-self.capacity :], self.capacity
        idx = (self.pos + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.label[idx] = label
        self.born[idx] = iteration
        self.pos = int((self.pos + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, n: int, generator: torch.Generator) -> npt.NDArray[np.int64]:
        """`n` indices uniformes avec remise parmi les cases remplies (CPU)."""
        if self.size == 0:
            raise ValueError("compartiment vide")
        return torch.randint(self.size, (n,), generator=generator).numpy()

    def age_mean(self, iteration: int) -> float:
        return float(iteration - self.born[: self.size].mean()) if self.size else 0.0

    def state_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {name: getattr(self, name)[: self.size] for name in self._FIELDS}
        out["pos"] = self.pos
        return out

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        n = int(np.asarray(state["label"]).shape[0])
        shape = np.asarray(state["obs"]).shape
        if n > self.capacity or shape != (n, self.obs_dim):
            raise ValueError(f"rejeu auxiliaire : {n} transitions de forme {shape} incompatibles")
        for name in self._FIELDS:
            getattr(self, name)[:n] = state[name]
        self.size, self.pos = n, int(state["pos"])


# ---------------------------------------------------------------------------------------------------- journal C4


@dataclass(frozen=True)
class AuxStats:
    """Journal C4 d'une itération (`aux/*`, ADR-010 D9 ; la RAM est ajoutée par la boucle). Jamais de NaN : une
    itération sans lot (compartiment ± trop petit, λ = 0, libérée) écrit des zéros."""

    coef: float
    loss: float  # entropie croisée moyenne sur les lots, AVANT λ
    batch_present: float  # minibatchs PPO avec un lot auxiliaire
    recall_pos: float  # sur le lot rééquilibré (le rappel ne dépend pas du rééquilibrage, la précision si)
    recall_neg: float
    recall_zero: float
    precision_pos: float
    precision_neg: float
    precision_zero: float
    frac_pos: float  # parmi les pas sortis de la fenêtre pendant l'ingestion de ce lot
    frac_neg: float
    frac_zero: float
    frac_masked: float
    exits: float  # pas sortis de la fenêtre (étiquetés ou masqués)
    replay_pos_size: float
    replay_zero_size: float
    replay_pos_new: float  # transitions ajoutées par ce lot
    replay_zero_new: float
    replay_pos_age_mean: float  # en itérations
    replay_bytes: float
    window_bytes: float
    released: float  # 1.0 après la fin du calendrier optionnel de λ (fenêtre et rejeu libérés)
    grad_norm_trunk_aux: float  # ‖g de λ·L_aux‖ sur le tronc, avant clip, moyenne des minibatchs avec lot
    grad_norm_trunk_ppo: float  # ‖g PPO‖ sur le tronc, avant clip, mêmes minibatchs
    grad_share: float  # indicateur C4 (alerte) : aux / (aux + ppo) ; 0 sans lot
    disp_norm_aux: float
    disp_norm_ppo: float
    disp_share: float  # diagnostic (ADR-009 D5) ; 0 sans lot

    def as_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}


def _ratio(a: float, b: float) -> float:
    return a / b if b > 0.0 else 0.0


# ---------------------------------------------------------------------------------------------------- tâche


class AuxTask:
    """État et calculs de la tâche auxiliaire d'un learner : fenêtre, rejeu à deux compartiments, tête, tirage,
    moment fantôme, compteurs."""

    def __init__(self, cfg: AuxConfig, policy: PolicyNet, head: AuxHead, seed: int) -> None:
        self.cfg, self.obs_dim = cfg, policy.obs_dim
        self.trunk = policy_trunk(policy)
        self.trunk_params = list(self.trunk.parameters())
        self.head = head
        self.pos: ClassReplay | None = ClassReplay(cfg.capacity_per_class, self.obs_dim, cfg.obs_dtype)
        self.zero: ClassReplay | None = ClassReplay(cfg.capacity_per_class, self.obs_dim, cfg.obs_dtype)
        self.window: GoalWindow | None = None  # créée au premier lot (nombre d'agents appris)
        self.cols: npt.NDArray[np.int64] | None = None
        self.generator = torch.Generator()
        self.generator.manual_seed(seed + AUX_SEED_OFFSET)
        self.released = False
        self.ghost: list[torch.Tensor] | None = None  # m^aux sur le tronc (None : jamais de lot, ou λ = 0)
        self.last_disp: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None
        """(Σ_k Δθ^ppo_k, Σ_k Δθ^aux_k) sur le tronc, dernière itération avec lot (diagnostic, test D8-8)."""

    def reseed(self, seed: int) -> None:
        self.generator.manual_seed(seed + AUX_SEED_OFFSET)

    @property
    def replay_nbytes(self) -> int:
        return sum(0 if r is None else r.nbytes for r in (self.pos, self.zero))

    @property
    def nbytes(self) -> int:
        """Octets de la fenêtre et du rejeu (hors tenseurs d'une mise à jour, tête et état de l'optimiseur)."""
        return self.replay_nbytes + (0 if self.window is None else self.window.nbytes)

    def release(self) -> None:
        """Fin du calendrier optionnel de λ : plus de collecte ni de calcul, mémoire rendue."""
        self.pos = self.zero = None
        self.window, self.released, self.ghost = None, True, None

    def ingest(self, obs: F32, rew: F32, term: BOOL, trunc: BOOL, learn_mask: BOOL, iteration: int) -> dict[str, int]:
        """Lot (T, N, …) dans l'ordre de collecte : met à jour la fenêtre des agents de `learn_mask` et range les pas
        sortis dans les compartiments. Rend les effectifs (pos, neg, zero, masked)."""
        counts = {"pos": 0, "neg": 0, "zero": 0, "masked": 0}
        if self.pos is None or self.zero is None:
            return counts
        cols = np.flatnonzero(learn_mask)
        if self.window is None:
            self.window = GoalWindow(cols.size, self.obs_dim, self.cfg.horizon_steps, self.cfg.obs_dtype)
            self.cols = cols
        elif self.cols is None or not np.array_equal(cols, self.cols):
            raise ValueError("learn_mask a changé en cours de run : fenêtre de la tâche auxiliaire incohérente")
        for t in range(obs.shape[0]):
            ex = self.window.step(obs[t, cols], rew[t, cols], term[t, cols], trunc[t, cols])
            self.zero.add(ex.zero_obs, np.zeros(ex.n_zero, LABEL_DTYPE), iteration)
            self.pos.add(ex.pos_obs, ex.pos_label, iteration)
            counts["pos"] += ex.n_pos
            counts["neg"] += ex.n_neg
            counts["zero"] += ex.n_zero
            counts["masked"] += ex.n_masked
        return counts

    def begin(
        self,
        obs: F32,
        rew: F32,
        term: BOOL,
        trunc: BOOL,
        learn_mask: BOOL,
        *,
        iteration: int,
        learned_per_iteration: int,
        device: torch.device,
        n_ppo_steps: int,
        optimizer: Any,
    ) -> AuxIteration:
        """Début d'une mise à jour : λ (timesteps appris AVANT cette mise à jour), libération en fin de calendrier,
        ingestion du lot AVANT tout pas, puis tirage des lots de l'itération."""
        learned = iteration * learned_per_iteration
        coef = self.cfg.coef_at(learned)
        end = self.cfg.anneal_end_timesteps
        if coef == 0.0 and end is not None and learned >= end and not self.released:
            self.release()
        sizes = (0 if self.pos is None else self.pos.size, 0 if self.zero is None else self.zero.size)
        counts = self.ingest(obs, rew, term, trunc, learn_mask, iteration)
        new = (
            (0 if self.pos is None else self.pos.size) - sizes[0],
            (0 if self.zero is None else self.zero.size) - sizes[1],
        )
        return AuxIteration(self, coef, iteration, counts, new, device, n_ppo_steps, optimizer)

    def state_dict(self) -> dict[str, Any]:
        return {
            "released": self.released,
            "pos": None if self.pos is None else self.pos.state_dict(),
            "zero": None if self.zero is None else self.zero.state_dict(),
            "window": None if self.window is None else self.window.state_dict(),
            "cols": self.cols,
            "generator": self.generator.get_state(),
            "head": self.head.state_dict(),
            "ghost": self.ghost,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if bool(state["released"]):
            self.release()
        else:
            self.released = False
            self.pos = ClassReplay(self.cfg.capacity_per_class, self.obs_dim, self.cfg.obs_dtype)
            self.zero = ClassReplay(self.cfg.capacity_per_class, self.obs_dim, self.cfg.obs_dtype)
            self.pos.load_state_dict(state["pos"])
            self.zero.load_state_dict(state["zero"])
        if state["window"] is None:
            self.window, self.cols = None, None
        else:
            cols = np.asarray(state["cols"], dtype=np.int64)
            self.window = GoalWindow(cols.size, self.obs_dim, self.cfg.horizon_steps, self.cfg.obs_dtype)
            self.window.load_state_dict(state["window"])
            self.cols = cols
        self.head.load_state_dict(dict(state["head"]))
        self.generator.set_state(state["generator"].cpu())
        ghost = state.get("ghost")
        # sur CPU au chargement ; ramené sur le device du tronc au premier usage (`AuxIteration`)
        self.ghost = None if ghost is None else [t.clone() for t in ghost]


class AuxIteration:
    """Tâche auxiliaire d'une mise à jour PPO. Appels de `Learner.update`, dans l'ordre, pour chaque minibatch PPO :
    `joint_step` après le `backward` PPO, puis `after_step` après le clip global et le pas d'Adam ; enfin `stats`.
    Avec λ = 0 ou sans lot, `joint_step` ne calcule rien (et, λ = 0, `after_step` non plus)."""

    def __init__(
        self,
        aux: AuxTask,
        coef: float,
        iteration: int,
        counts: Mapping[str, int],
        new: tuple[int, int],
        device: torch.device,
        n_ppo_steps: int,
        optimizer: Any,
    ) -> None:
        self.aux, self.cfg, self.coef, self.iteration = aux, aux.cfg, coef, iteration
        self.counts, self.new, self.n_ppo_steps = dict(counts), new, n_ppo_steps
        self.active = coef > 0.0 and not aux.released
        if not self.active:
            aux.ghost = None  # décomposition à reprendre de zéro si λ redevient > 0
        pos, zero, half = aux.pos, aux.zero, self.cfg.half
        self.has_batch = bool(
            self.active and pos is not None and zero is not None and pos.size >= half and zero.size >= half
        )
        self.k = 0
        self._obs: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._g_aux: list[torch.Tensor] | None = None
        self._adam: AdamView | None = None
        self._disp_aux: list[torch.Tensor] | None = None
        self._disp_ppo: list[torch.Tensor] | None = None
        self._acc: torch.Tensor | None = None
        params = aux.trunk_params
        if self.has_batch:
            assert pos is not None and zero is not None
            n = n_ppo_steps * half
            ip, iz = pos.sample(n, aux.generator), zero.sample(n, aux.generator)
            obs = np.empty((n_ppo_steps, self.cfg.batch_size, aux.obs_dim), dtype=pos.obs.dtype)
            obs[:, :half] = pos.obs[ip].reshape(n_ppo_steps, half, aux.obs_dim)
            obs[:, half:] = zero.obs[iz].reshape(n_ppo_steps, half, aux.obs_dim)
            y = np.empty((n_ppo_steps, self.cfg.batch_size), dtype=np.int64)
            y[:, :half] = pos.label[ip].reshape(n_ppo_steps, half).astype(np.int64) + 1
            y[:, half:] = zero.label[iz].reshape(n_ppo_steps, half).astype(np.int64) + 1
            self._obs = torch.from_numpy(obs).to(device)  # une copie hôte → appareil par itération (D4)
            self._y = torch.from_numpy(y).to(device)
            self._acc = torch.zeros(4 + N_CLASSES * N_CLASSES, device=device)
            self._adam = AdamView(optimizer)
            self._disp_aux = [torch.zeros_like(p) for p in params]
            self._disp_ppo = [torch.zeros_like(p) for p in params]
            if aux.ghost is None:
                aux.ghost = [torch.zeros_like(p) for p in params]  # m^aux = 0, m^ppo = exp_avg
        if aux.ghost is not None:
            aux.ghost = [t.to(p.device) for t, p in zip(aux.ghost, params, strict=True)]
            if self._adam is None:
                self._adam = AdamView(optimizer)

    def joint_step(self) -> None:
        """Ajoute le gradient de λ·L_aux du lot k aux gradients PPO du minibatch (rien sans lot)."""
        if not self.has_batch:
            return
        assert self._obs is not None and self._y is not None and self._acc is not None
        params = self.aux.trunk_params
        with torch.no_grad():
            g_ppo = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
        x = self._obs[self.k].float()
        y = self._y[self.k]
        logits = aux_logits(self.aux.trunk, self.aux.head, x)
        ce = nn.functional.cross_entropy(logits, y)
        torch.autograd.backward([self.coef * ce])
        with torch.no_grad():
            self._g_aux = [
                torch.zeros_like(p) if p.grad is None else p.grad.detach() - g
                for p, g in zip(params, g_ppo, strict=True)
            ]
            classes = torch.arange(N_CLASSES, device=y.device)
            true = (y.unsqueeze(1) == classes).float()  # (B, 3)
            pred = (logits.argmax(-1).unsqueeze(1) == classes).float()
            conf = (true.T @ pred).reshape(-1)  # conf[i, j] = effectif (classe i, prédiction j)
            self._acc += torch.cat(
                [
                    torch.stack([ce.detach(), torch.ones((), device=y.device), _norm(self._g_aux), _norm(g_ppo)]),
                    conf,
                ]
            )

    def after_step(self, total_grad_norm: torch.Tensor, max_norm: float) -> None:
        """Après le clip global et le pas d'Adam : moment fantôme m^aux et déplacements Δθ^aux, Δθ^ppo (tronc)."""
        ghost, adam = self.aux.ghost, self._adam
        if not self.active or ghost is None or adam is None:
            self.k += 1
            return
        b1, b2, eps, lr = adam.beta1, adam.beta2, adam.eps, adam.lr
        with torch.no_grad():
            f = torch.clamp(max_norm / (total_grad_norm.detach() + CLIP_EPS), max=1.0)
            for i, p in enumerate(self.aux.trunk_params):
                ghost[i].mul_(b1)
                if self._g_aux is not None:
                    ghost[i].add_(f * self._g_aux[i], alpha=1.0 - b1)
                if self._disp_aux is None or self._disp_ppo is None:
                    continue
                m = adam.exp_avg(p)
                v, t = adam.exp_avg_sq_and_step(p)
                step_size = lr / (1.0 - b1**t)
                denom = v.sqrt() / (1.0 - b2**t) ** 0.5 + eps
                m_ppo = (torch.zeros_like(p) if m is None else m) - ghost[i]
                self._disp_aux[i].add_(ghost[i] / denom, alpha=-step_size)
                self._disp_ppo[i].add_(m_ppo / denom, alpha=-step_size)
        self._g_aux = None
        self.k += 1

    def stats(self) -> AuxStats:
        aux, c = self.aux, self.counts
        if self._acc is not None and self._disp_aux is not None and self._disp_ppo is not None:
            aux.last_disp = (self._disp_ppo, self._disp_aux)
            vals = torch.cat([self._acc, torch.stack([_norm(self._disp_aux), _norm(self._disp_ppo)])]).tolist()
        else:
            vals = [0.0] * (4 + N_CLASSES * N_CLASSES + 2)  # une seule lecture hôte par itération (D4)
        ce_sum, n_b, g_aux_sum, g_ppo_sum = vals[:4]
        conf = np.asarray(vals[4 : 4 + N_CLASSES * N_CLASSES]).reshape(N_CLASSES, N_CLASSES)
        d_aux, d_ppo = vals[-2:]
        g_aux, g_ppo = _ratio(g_aux_sum, n_b), _ratio(g_ppo_sum, n_b)
        recall = [_ratio(float(conf[i, i]), float(conf[i].sum())) for i in range(N_CLASSES)]
        precision = [_ratio(float(conf[i, i]), float(conf[:, i].sum())) for i in range(N_CLASSES)]
        exits = c["pos"] + c["neg"] + c["zero"] + c["masked"]
        pos, zero = aux.pos, aux.zero
        return AuxStats(
            coef=self.coef,
            loss=_ratio(ce_sum, n_b),
            batch_present=float(n_b),
            recall_pos=recall[POS],
            recall_neg=recall[NEG],
            recall_zero=recall[ZERO],
            precision_pos=precision[POS],
            precision_neg=precision[NEG],
            precision_zero=precision[ZERO],
            frac_pos=_ratio(c["pos"], exits),
            frac_neg=_ratio(c["neg"], exits),
            frac_zero=_ratio(c["zero"], exits),
            frac_masked=_ratio(c["masked"], exits),
            exits=float(exits),
            replay_pos_size=float(0 if pos is None else pos.size),
            replay_zero_size=float(0 if zero is None else zero.size),
            replay_pos_new=float(self.new[0]),
            replay_zero_new=float(self.new[1]),
            replay_pos_age_mean=0.0 if pos is None else pos.age_mean(self.iteration),
            replay_bytes=float(aux.replay_nbytes),
            window_bytes=float(0 if aux.window is None else aux.window.nbytes),
            released=float(aux.released),
            grad_norm_trunk_aux=g_aux,
            grad_norm_trunk_ppo=g_ppo,
            grad_share=_ratio(g_aux, g_aux + g_ppo),
            disp_norm_aux=d_aux,
            disp_norm_ppo=d_ppo,
            disp_share=_ratio(d_aux, d_aux + d_ppo),
        )


def _norm(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Norme L2 globale, sur l'appareil (aucune lecture hôte)."""
    out: torch.Tensor = torch.stack([t.norm() for t in tensors]).norm()
    return out


# ---------------------------------------------------------------------------------------------------- garde C4


@dataclass(frozen=True)
class AuxGuardState:
    """Compteurs de la garde auxiliaire (sauvegardés dans `extra["aux_guard"]` des checkpoints)."""

    share_alert_streak: int = 0
    ram_low_s: float = 0.0


def aux_guard(
    cfg: AuxConfig,
    state: AuxGuardState,
    *,
    share: float,
    ram_available_mb: float | None,
    iteration_s: float,
) -> tuple[AuxGuardState, tuple[str, ...], tuple[str, str] | None]:
    """Fonction pure (ADR-010 D6 C4, KS1) : (nouvel état, alertes « règle : détail », arrêt (règle, raison) ou None).
    `share` = `grad_share` de l'itération.
    - alerte `aux_share` : share > `share_alert` pendant `share_alert_patience` itérations consécutives ;
    - STOP `aux_ram` (KS1) : RAM disponible < `ram_min_available_mb` pendant plus de `ram_low_max_s` secondes
      (durées d'itération cumulées, remises à 0 dès une itération au-dessus du seuil ; mesure absente : inchangé)."""
    a = state.share_alert_streak + 1 if share > cfg.share_alert else 0
    low = state.ram_low_s
    if ram_available_mb is not None:
        low = low + iteration_s if ram_available_mb < cfg.ram_min_available_mb else 0.0
    new = AuxGuardState(share_alert_streak=a, ram_low_s=low)
    alerts: tuple[str, ...] = ()
    if a >= cfg.share_alert_patience:
        alerts = (f"aux_share : grad_share {share:.3g} > {cfg.share_alert:g} depuis {a} itérations (C4, ADR-010 D6)",)
    stop: tuple[str, str] | None = None
    if low > cfg.ram_low_max_s:
        stop = (
            "aux_ram",
            f"RAM disponible {ram_available_mb} Mo < {cfg.ram_min_available_mb:g} Mo depuis {low:.0f} s (ADR-010 KS1)",
        )
    return new, alerts, stop


def expected_nbytes(n_agents: int, obs_dim: int, cfg: AuxConfig) -> int:
    """Octets de la fenêtre et du rejeu aux dimensions données (formule de l'ADR-010, Mesures)."""
    item = np.dtype(cfg.obs_dtype).itemsize
    window = n_agents * cfg.horizon_steps * obs_dim * item + n_agents * 8
    replay = 2 * cfg.capacity_per_class * (obs_dim * item + 1 + 4)
    return window + replay
