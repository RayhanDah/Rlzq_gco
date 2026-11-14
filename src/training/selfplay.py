"""Pool d'adversaires du self-play (ADR-005 D5, `docs/01_papers.md` §1.8, OpenAI Five App. N).

Répartition (fixe pour tout le run) : une fraction `past_fraction` des arènes (20 %) joue contre un instantané passé,
les autres contre la version courante (deux camps appris). Dans une arène « passé », un camp entier est gelé : ses
voitures jouent l'instantané tiré pour l'épisode et sont exclues de la perte (`learn_mask` faux) ; l'autre camp joue
la politique courante et apprend. Les arènes « passé » sont réparties régulièrement sur les indices et le camp gelé
alterne bleu / orange (aucun côté favorisé). La partition étant fixe, la proportion d'arènes et le nombre de
transitions apprises par lot sont exacts et constants : le budget en timesteps appris (ADR-005 D3) se convertit en
itérations sans approximation.

Scores de qualité (Five) : tirage p_i ∝ exp(q_i) à chaque début d'épisode d'une arène « passé » ; quand la politique
courante bat l'instantané i (elle marque), q_i ← q_i − η / (N · p_i), N = taille du pool, p_i = probabilité courante ;
nuls et défaites ne changent rien. Un nouvel instantané entre avec q = max(q) (0 pour le premier) ; au-delà de
`max_size`, le plus mauvais q sort (à égalité, le plus ancien). Un instantané évincé qui joue encore dans une arène y
finit son épisode (il ne change pas d'adversaire en cours de partie), puis ses poids sont libérés.

Aucune récompense n'est ajoutée (GCO strict) : seul l'adversaire change. Le tirage utilise un générateur CPU propre
(état sauvegardé dans le checkpoint avec les poids, les scores et l'affectation des arènes : reprise exacte).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from models.policy import PolicyNet

F32 = npt.NDArray[np.float32]
I64 = npt.NDArray[np.int64]
BOOL = npt.NDArray[np.bool_]

GOAL_NONE = -1  # `rewards.GoalNet.NONE` ; GoalNet.t = but dans le filet de l'équipe t


# ---------------------------------------------------------------------------------------------------- répartition


@dataclass(frozen=True)
class PoolLayout:
    """Partition fixe des arènes et des agents (agents à plat `a = e·C + k`, ADR-003 §5)."""

    n_envs: int
    n_cars: int
    arenas: I64  # (P,) arènes « passé », croissantes
    frozen_team: I64  # (P,) équipe gelée (0 bleu, 1 orange) de chaque arène « passé »
    frozen_agents: I64  # (P, C/2) agents gelés de chaque arène « passé »
    learn_mask: BOOL  # (N,) faux pour les agents gelés

    @property
    def n_past(self) -> int:
        return int(self.arenas.size)

    @property
    def n_learned_agents(self) -> int:
        return int(self.learn_mask.sum())


def n_past_arenas(n_envs: int, past_fraction: float) -> int:
    """Nombre d'arènes « passé » : arrondi au plus proche de `past_fraction · n_envs` (écart ≤ 1/(2·n_envs))."""
    return math.floor(past_fraction * n_envs + 0.5)


def make_layout(n_envs: int, n_cars: int, past_fraction: float) -> PoolLayout:
    """Arènes « passé » régulièrement espacées ; camp gelé alterné ; équipe par slot : `k < C/2` bleu (ADR-003 §1)."""
    if n_cars < 2 or n_cars % 2:
        raise ValueError(f"n_cars={n_cars} : pair et >= 2 attendu")
    p = n_past_arenas(n_envs, past_fraction)
    if not 1 <= p < n_envs:
        raise ValueError(f"past_fraction={past_fraction} sur {n_envs} arènes donne {p} arène(s) « passé »")
    arenas = (np.arange(p, dtype=np.int64) * n_envs) // p
    frozen_team = np.arange(p, dtype=np.int64) % 2
    half = n_cars // 2
    slots = frozen_team[:, None] * half + np.arange(half, dtype=np.int64)[None, :]
    frozen_agents = arenas[:, None] * n_cars + slots
    mask = np.ones(n_envs * n_cars, dtype=np.bool_)
    mask[frozen_agents.reshape(-1)] = False
    return PoolLayout(n_envs, n_cars, arenas, frozen_team, frozen_agents, mask)


# ---------------------------------------------------------------------------------------------------- scores


_MIN_PROB = 1e-12  # plancher de p dans la règle Five : évite la division par zéro, borne le pas sur q


class QualityPool:
    """Identifiants d'instantanés, scores de qualité q, tirage p ∝ exp(q) et éviction. Sans poids (voir
    `PastOpponents`) : testable seul."""

    def __init__(self, max_size: int, quality_lr: float, seed: int) -> None:
        if max_size < 1 or not quality_lr > 0.0:
            raise ValueError(f"max_size={max_size} >= 1 et quality_lr={quality_lr} > 0 attendus")
        self.max_size, self.quality_lr = max_size, quality_lr
        self.uids: list[int] = []  # membres, ordre d'entrée
        self.q: dict[int, float] = {}
        self.iteration: dict[int, int] = {}  # itération d'entraînement de chaque instantané
        self.next_uid = 0
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __len__(self) -> int:
        return len(self.uids)

    def add(self, iteration: int) -> tuple[int, int | None]:
        """Ajoute un instantané (q = max des q, 0 si vide) ; rend (uid ajouté, uid évincé ou None)."""
        uid = self.next_uid
        self.next_uid += 1
        self.q[uid] = max(self.q.values()) if self.q else 0.0
        self.iteration[uid] = int(iteration)
        self.uids.append(uid)
        evicted = None
        if len(self.uids) > self.max_size:
            evicted = min(self.uids, key=lambda u: (self.q[u], u))  # plus mauvais q ; à égalité, le plus ancien
            self.uids.remove(evicted)
            del self.q[evicted], self.iteration[evicted]
        return uid, evicted

    def probs(self) -> npt.NDArray[np.float64]:
        """p_i ∝ exp(q_i), dans l'ordre de `uids` (softmax stabilisé)."""
        if not self.uids:
            raise RuntimeError("pool vide")
        q = np.array([self.q[u] for u in self.uids], dtype=np.float64)
        e = np.exp(q - np.max(q))
        out: npt.NDArray[np.float64] = e / e.sum()
        return out

    def sample(self, n: int) -> I64:
        """`n` tirages indépendants (avec remise) selon `probs()`."""
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        idx = torch.multinomial(torch.from_numpy(self.probs()), n, replacement=True, generator=self.generator)
        members = np.array(self.uids, dtype=np.int64)
        out: I64 = members[idx.numpy()]
        return out

    def record_win(self, uid: int) -> bool:
        """La politique courante a battu `uid` : q_uid ← q_uid − η/(N·p_uid). Sans effet (False) si `uid` évincé."""
        if uid not in self.q:
            return False
        p = max(float(self.probs()[self.uids.index(uid)]), _MIN_PROB)  # p peut valoir exactement 0,0 (q très bas)
        self.q[uid] -= self.quality_lr / (len(self.uids) * p)
        return True

    def reseed(self, seed: int) -> None:
        self.generator.manual_seed(seed)

    def state_dict(self) -> dict[str, Any]:
        return {
            "uids": list(self.uids),
            "q": [self.q[u] for u in self.uids],
            "iteration": [self.iteration[u] for u in self.uids],
            "next_uid": self.next_uid,
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        uids = [int(u) for u in state["uids"]]
        self.uids = uids
        self.q = {u: float(v) for u, v in zip(uids, state["q"], strict=True)}
        self.iteration = {u: int(v) for u, v in zip(uids, state["iteration"], strict=True)}
        self.next_uid = int(state["next_uid"])
        self.generator.set_state(state["generator"].cpu())


# ---------------------------------------------------------------------------------------------------- collecte


@dataclass
class _Stack:
    """Poids de U `PolicyNet` empilés : couche i = (W (U, in, out), b (U, 1, out)) ; ReLU entre les couches."""

    uids: list[int]
    layers: list[tuple[torch.Tensor, torch.Tensor]]

    @staticmethod
    def build(uids: list[int], nets: list[PolicyNet]) -> _Stack:
        per_net: list[list[nn.Linear]] = []
        for net in nets:
            mods = list(net.net)
            lin = [m for m in mods[0::2] if isinstance(m, nn.Linear)]
            if len(lin) != len(mods[0::2]) or not all(isinstance(m, nn.ReLU) for m in mods[1::2]):
                raise TypeError("empilement : PolicyNet attendu en Linear/ReLU alternés (models.policy.build_mlp)")
            per_net.append(lin)
        layers = [
            (
                torch.stack([lin[i].weight.detach().t() for lin in per_net]).contiguous(),
                torch.stack([lin[i].bias.detach() for lin in per_net]).unsqueeze(1).contiguous(),
            )
            for i in range(len(per_net[0]))
        ]
        return _Stack(list(uids), layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` (U, B, D) → logits (U, B, A) : la tranche u passe dans le réseau u."""
        for i, (w, b) in enumerate(self.layers):
            x = torch.baddbmm(b, x, w)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class PastOpponents:
    """Adversaires passés pendant la collecte : poids des instantanés (sur le device de l'acteur), affectation
    instantané ↔ arène « passé », mises à jour de q en fin d'épisode. Crochet de `training.rollout.Collector`."""

    def __init__(
        self,
        layout: PoolLayout,
        pool: QualityPool,
        make_policy: Callable[[], PolicyNet],
        device: torch.device,
    ) -> None:
        self.layout, self.pool, self.make_policy, self.device = layout, pool, make_policy, device
        self.learn_mask = layout.learn_mask
        self.nets: dict[int, PolicyNet] = {}  # membres du pool + évincés encore en jeu
        self.assignment = np.full(layout.n_past, -1, dtype=np.int64)  # uid joué par chaque arène « passé »
        self.wins = 0  # épisodes « passé » gagnés par la politique courante (elle marque)
        self.losses = 0  # l'instantané marque
        self.draws = 0  # troncature sans but
        self._rows = torch.from_numpy(layout.frozen_agents.reshape(-1)).to(device)
        self._stack: _Stack | None = None  # poids empilés de `nets` ; reconstruit quand l'ensemble change

    # -- instantanés

    def _stacked(self) -> _Stack:
        """Poids de tous les réseaux vivants empilés par couche, dans l'ordre croissant des uid (ordre indépendant de
        l'historique : même empilement après une reprise). Un seul passage groupé remplace un passage par instantané
        (mesuré sur XPU à 100 instantanés : ≈ 0,7 ms par réseau et par step, ADR-001 M7)."""
        uids = sorted(self.nets)
        if self._stack is None or self._stack.uids != uids:
            self._stack = _Stack.build(uids, [self.nets[u] for u in uids])
        return self._stack

    def add_snapshot(self, policy: PolicyNet, iteration: int) -> int:
        """Copie gelée de `policy` (poids clonés) ; rend son uid. L'évincé reste en mémoire s'il joue encore."""
        net = self.make_policy().to(self.device)
        net.load_state_dict(policy.state_dict())
        net.requires_grad_(False)
        net.eval()
        uid, _ = self.pool.add(iteration)
        self.nets[uid] = net
        self._stack = None
        self._release()
        return uid

    def snapshot_iterations(self) -> list[int]:
        """Itérations des instantanés du pool (ordre d'entrée)."""
        return [self.pool.iteration[u] for u in self.pool.uids]

    def policy_of(self, uid: int) -> PolicyNet:
        return self.nets[uid]

    def _release(self) -> None:
        live = set(self.pool.uids) | {int(u) for u in self.assignment if u >= 0}
        for uid in [u for u in self.nets if u not in live]:
            del self.nets[uid]

    def _assign(self, which: I64) -> None:
        self.assignment[which] = self.pool.sample(int(which.size))

    # -- crochet du collecteur

    def override_logits(self, obs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Remplace (en place) les logits des agents gelés par ceux de leur instantané ; affecte d'abord un adversaire
        aux arènes qui n'en ont pas (début de run)."""
        todo = np.flatnonzero(self.assignment < 0)
        if todo.size:
            self._assign(todo)
        stack = self._stacked()
        half = self.layout.n_cars // 2
        # lignes gelées (P·half), regroupées par réseau : slot = rang du réseau dans l'empilement, pos = rang dans
        # son groupe ; les groupes sont complétés à la taille du plus grand (lignes de remplissage ignorées)
        slot = np.repeat(np.searchsorted(np.array(stack.uids, dtype=np.int64), self.assignment), half)
        order = np.argsort(slot, kind="stable")
        counts = np.bincount(slot, minlength=len(stack.uids))
        starts = np.cumsum(counts) - counts
        pos = np.empty_like(order)
        pos[order] = np.arange(order.size) - starts[slot[order]]
        pad = np.zeros((len(stack.uids), int(np.max(counts))), dtype=np.int64)
        pad[slot, pos] = np.arange(order.size)
        pad_t = torch.from_numpy(pad).to(self.device)
        x = obs.index_select(0, self._rows.index_select(0, pad_t.reshape(-1))).view(*pad.shape, -1)
        out = stack.forward(x)  # (U, Bmax, A)
        flat = torch.from_numpy(slot * pad.shape[1] + pos).to(self.device)
        logits.index_copy_(0, self._rows, out.reshape(-1, out.shape[-1]).index_select(0, flat))
        return logits

    def after_step(self, term: BOOL, trunc: BOOL, info: Mapping[str, Any]) -> None:
        """Fin d'épisode des arènes « passé » : bilan (q mis à jour si la politique courante a marqué), puis nouveau
        tirage pour l'épisode suivant (l'env a déjà réinitialisé ces arènes dans le même step)."""
        c = self.layout.n_cars
        first = self.layout.arenas * c
        done = term[first] | trunc[first]
        if not done.any():
            return
        if "goal_net" not in info:
            raise KeyError("info['goal_net'] absent : requis par le pool d'adversaires")
        net = np.asarray(info["goal_net"], dtype=np.int64)[self.layout.arenas]
        ended = np.flatnonzero(done)
        for j in ended.tolist():
            if net[j] == GOAL_NONE:
                self.draws += 1
            elif net[j] == self.layout.frozen_team[j]:  # but dans le filet du camp gelé
                self.wins += 1
                self.pool.record_win(int(self.assignment[j]))
            else:
                self.losses += 1
        self._assign(ended)
        self._release()

    # -- reprise

    def reseed(self, seed: int) -> None:
        self.pool.reseed(seed)

    def state_dict(self) -> dict[str, Any]:
        live = sorted(self.nets)
        return {
            "pool": self.pool.state_dict(),
            "assignment": self.assignment.copy(),
            "net_uids": live,
            "nets": [self.nets[u].state_dict() for u in live],
            "counts": [self.wins, self.losses, self.draws],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        assignment = np.array(state["assignment"], dtype=np.int64)
        if assignment.shape != self.assignment.shape:
            raise ValueError(f"affectation {assignment.shape} ≠ {self.assignment.shape} (répartition différente)")
        self.pool.load_state_dict(state["pool"])
        nets: dict[int, PolicyNet] = {}
        for uid, weights in zip(state["net_uids"], state["nets"], strict=True):
            net = self.make_policy().to(self.device)
            net.load_state_dict(weights)
            net.requires_grad_(False)
            net.eval()
            nets[int(uid)] = net
        missing = {int(u) for u in assignment if u >= 0} | set(self.pool.uids)
        if not missing <= set(nets):
            raise ValueError(f"poids manquants pour les instantanés {sorted(missing - set(nets))}")
        self.nets, self.assignment = nets, assignment
        self._stack = None  # un uid peut désigner d'autres poids après une restauration
        self.wins, self.losses, self.draws = (int(x) for x in state["counts"])

    def scalars(self) -> dict[str, float]:
        """Scalaires de journal (préfixe `pool/`)."""
        q = [self.pool.q[u] for u in self.pool.uids]
        return {
            "pool/size": float(len(self.pool)),
            "pool/q_max": max(q) if q else float("nan"),
            "pool/q_min": min(q) if q else float("nan"),
            "pool/newest_iteration": float(self.pool.iteration[self.pool.uids[-1]]) if q else float("nan"),
            "pool/wins": float(self.wins),
            "pool/losses": float(self.losses),
            "pool/draws": float(self.draws),
        }
