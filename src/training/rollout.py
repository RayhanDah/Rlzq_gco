"""Tampon de rollout (T, N, …) et collecteur (ADR-003 §5–§6).

- `RolloutBatch` : un lot figé, tableaux numpy du §6, propriété transférée au learner (jamais réécrit tant qu'il l'a).
- `RolloutBuffer` : stockage préalloué, rempli pas à pas par le collecteur.
- `Collector` : fait tourner un `VecEnv` (contrat de `BatchEnv`, §5) avec une politique « acteur » et alterne deux
  tampons (double tampon) : le lot k reste intact pendant la collecte du lot k+1 (recouvrement, ADR-001 §3).
  Avec un crochet `opponents` (pool d'adversaires, `training.selfplay.PastOpponents`), les agents gelés jouent leur
  instantané et sont exclus de la perte par le `learn_mask` du crochet (ADR-005 D5).
Aucune récompense n'est calculée ici : `rew` est recopié tel que l'env le rend (GCO strict, `GCO_SPEC.md` §2).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch

from models.policy import PolicyNet, sample_actions

F32 = npt.NDArray[np.float32]
I64 = npt.NDArray[np.int64]
BOOL = npt.NDArray[np.bool_]


class VecEnv(Protocol):
    """Sous-ensemble du contrat `BatchEnv` (ADR-003 §5) utilisé par la collecte."""

    @property
    def n_agents(self) -> int: ...

    @property
    def obs_dim(self) -> int: ...

    def reset(self, seed: int) -> F32: ...

    def step(self, actions: I64) -> tuple[F32, F32, BOOL, BOOL, Mapping[str, Any]]: ...


@dataclass(frozen=True)
class RolloutBatch:
    """Lot de T steps × N agents (ADR-003 §6). `final_*` : lignes tronquées non terminales seulement."""

    obs: F32  # (T, N, D)
    act: I64  # (T, N)
    logp: F32  # (T, N)
    rew: F32  # (T, N)
    term: BOOL  # (T, N)
    trunc: BOOL  # (T, N)
    final_obs: F32  # (K, D)
    final_t: I64  # (K,)
    final_agent: I64  # (K,)
    last_obs: F32  # (N, D)
    learn_mask: BOOL  # (N,)
    policy_version: int

    @property
    def n_steps(self) -> int:
        return int(self.obs.shape[0])

    @property
    def n_agents(self) -> int:
        return int(self.obs.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[2])

    def validate(self) -> None:
        """Vérifie formes, dtypes et la cohérence `final_*` ↔ `trunc ∧ ¬term` ; lève `ValueError`."""
        if self.obs.ndim != 3:
            raise ValueError(f"obs doit être (T, N, D), reçu {self.obs.shape}")
        t_len, n, d = self.obs.shape
        k = self.final_obs.shape[0] if self.final_obs.ndim == 2 else -1
        expected: dict[str, tuple[tuple[int, ...], type[np.generic]]] = {
            "obs": ((t_len, n, d), np.float32),
            "act": ((t_len, n), np.int64),
            "logp": ((t_len, n), np.float32),
            "rew": ((t_len, n), np.float32),
            "term": ((t_len, n), np.bool_),
            "trunc": ((t_len, n), np.bool_),
            "final_obs": ((k, d), np.float32),
            "final_t": ((k,), np.int64),
            "final_agent": ((k,), np.int64),
            "last_obs": ((n, d), np.float32),
            "learn_mask": ((n,), np.bool_),
        }
        for name, (shape, dtype) in expected.items():
            arr = getattr(self, name)
            if arr.shape != shape or arr.dtype != dtype:
                raise ValueError(f"{name} : {arr.shape} {arr.dtype}, attendu {shape} {np.dtype(dtype)}")
        if np.any(self.final_t < 0) or np.any(self.final_t >= t_len):
            raise ValueError("final_t hors de [0, T)")
        if np.any(self.final_agent < 0) or np.any(self.final_agent >= n):
            raise ValueError("final_agent hors de [0, N)")
        cut = self.trunc & ~self.term
        flat = self.final_t * n + self.final_agent
        if np.unique(flat).size != k or not np.array_equal(np.sort(flat), np.flatnonzero(cut)):
            raise ValueError("les lignes final_* ne correspondent pas exactement aux transitions trunc ∧ ¬term")


class RolloutBuffer:
    """Tableaux préalloués (T, N, …) remplis par `add`, figés par `finish`."""

    def __init__(self, n_steps: int, n_agents: int, obs_dim: int) -> None:
        if n_steps <= 0 or n_agents <= 0 or obs_dim <= 0:
            raise ValueError(f"tailles invalides : T={n_steps}, N={n_agents}, D={obs_dim}")
        self.n_steps, self.n_agents, self.obs_dim = n_steps, n_agents, obs_dim
        self.obs = np.zeros((n_steps, n_agents, obs_dim), dtype=np.float32)
        self.act = np.zeros((n_steps, n_agents), dtype=np.int64)
        self.logp = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.rew = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.term = np.zeros((n_steps, n_agents), dtype=np.bool_)
        self.trunc = np.zeros((n_steps, n_agents), dtype=np.bool_)
        self._final_obs: list[F32] = []
        self._final_t: list[I64] = []
        self._final_agent: list[I64] = []
        self._t = 0

    def add(
        self,
        obs: F32,
        act: I64,
        logp: F32,
        rew: F32,
        term: BOOL,
        trunc: BOOL,
        final_obs: F32,
        final_agent: I64,
    ) -> None:
        """Ajoute le step courant. `final_obs`/`final_agent` : lignes finales rendues par l'env (term ou trunc) ;
        seules celles des agents tronqués non terminaux sont gardées (ADR-003 §6)."""
        t = self._t
        if t >= self.n_steps:
            raise RuntimeError("tampon plein : appeler finish()")
        np.copyto(self.obs[t], obs)
        np.copyto(self.act[t], act)
        np.copyto(self.logp[t], logp)
        np.copyto(self.rew[t], rew)
        np.copyto(self.term[t], term)
        np.copyto(self.trunc[t], trunc)
        if final_agent.shape[0] != final_obs.shape[0]:
            raise ValueError("final_obs et final_agent de longueurs différentes")
        keep = np.flatnonzero(self.trunc[t, final_agent] & ~self.term[t, final_agent])
        if keep.size:
            self._final_obs.append(np.array(final_obs[keep], dtype=np.float32))
            self._final_agent.append(np.array(final_agent[keep], dtype=np.int64))
            self._final_t.append(np.full(keep.size, t, dtype=np.int64))
        self._t = t + 1

    def finish(self, last_obs: F32, learn_mask: BOOL, policy_version: int) -> RolloutBatch:
        """Fige le lot (sans copier les tableaux (T, N, …)) et remet le curseur à 0 pour la réutilisation."""
        if self._t != self.n_steps:
            raise RuntimeError(f"tampon incomplet : {self._t}/{self.n_steps} steps")
        d = self.obs_dim
        batch = RolloutBatch(
            obs=self.obs,
            act=self.act,
            logp=self.logp,
            rew=self.rew,
            term=self.term,
            trunc=self.trunc,
            final_obs=np.concatenate(self._final_obs) if self._final_obs else np.zeros((0, d), np.float32),
            final_t=np.concatenate(self._final_t) if self._final_t else np.zeros(0, np.int64),
            final_agent=np.concatenate(self._final_agent) if self._final_agent else np.zeros(0, np.int64),
            last_obs=np.array(last_obs, dtype=np.float32),
            learn_mask=np.array(learn_mask, dtype=np.bool_),
            policy_version=int(policy_version),
        )
        batch.validate()
        self._final_obs, self._final_t, self._final_agent = [], [], []
        self._t = 0
        return batch


class OpponentHook(Protocol):
    """Crochet d'adversaires gelés (`training.selfplay.PastOpponents`)."""

    learn_mask: BOOL

    def override_logits(self, obs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor: ...

    def after_step(self, term: BOOL, trunc: BOOL, info: Mapping[str, Any]) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class Collector:
    """Collecte des lots de `n_steps` × `n_agents` avec une politique acteur (copie de π, ADR-001 §3).

    `policy` est l'acteur : en recouvrement, il ne doit pas être le module mis à jour par le learner. Le collecteur
    garde l'observation courante entre deux lots (les épisodes enjambent les frontières de lot).

    `opponents` : crochet optionnel ; son `learn_mask` remplace `learn_mask` (les deux ensemble : refusé). Les logits
    des agents gelés sont remplacés avant l'échantillonnage (un seul tirage, même générateur) ; sans crochet, la
    collecte est identique bit à bit à celle d'avant le pool. `learnable_steps` compte les transitions collectées qui
    entreront dans la perte (agents de `learn_mask`), `agent_steps` toutes les transitions collectées.
    """

    def __init__(
        self,
        env: VecEnv,
        policy: PolicyNet,
        n_steps: int,
        device: torch.device,
        seed: int,
        learn_mask: BOOL | None = None,
        opponents: OpponentHook | None = None,
    ) -> None:
        self.env, self.policy, self.n_steps, self.device = env, policy, n_steps, device
        self.opponents = opponents
        n, d = env.n_agents, env.obs_dim
        if opponents is not None:
            if learn_mask is not None:
                raise ValueError("learn_mask et opponents sont exclusifs (le masque vient du crochet)")
            learn_mask = opponents.learn_mask
        self.learn_mask = np.ones(n, dtype=np.bool_) if learn_mask is None else np.array(learn_mask, dtype=np.bool_)
        if self.learn_mask.shape != (n,):
            raise ValueError(f"learn_mask {self.learn_mask.shape} ≠ ({n},)")
        self._buffers = (RolloutBuffer(n_steps, n, d), RolloutBuffer(n_steps, n, d))
        self._next_buffer = 0
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self._obs: F32 = np.array(env.reset(seed), dtype=np.float32)
        if self._obs.shape != (n, d):
            raise ValueError(f"reset : obs {self._obs.shape} ≠ ({n}, {d})")
        self.agent_steps = 0
        self.learnable_steps = 0
        self._n_learn = int(self.learn_mask.sum())

    def state_dict(self) -> dict[str, Any]:
        """État du collecteur pour la reprise (T5) : générateur d'échantillonnage, observation courante, compteurs et,
        avec un crochet, état complet du pool (poids, scores, tirage, affectation)."""
        state: dict[str, Any] = {
            "generator": self.generator.get_state(),
            "obs": np.array(self._obs, dtype=np.float32),
            "agent_steps": self.agent_steps,
            "learnable_steps": self.learnable_steps,
        }
        if self.opponents is not None:
            state["opponents"] = self.opponents.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any], obs: F32 | None = None) -> None:
        """`obs` remplace l'observation sauvegardée (env restauré « modulo reset des arènes »)."""
        cur = np.array(state["obs"] if obs is None else obs, dtype=np.float32)
        if cur.shape != self._obs.shape:
            raise ValueError(f"obs {cur.shape} ≠ {self._obs.shape}")
        if (self.opponents is not None) != ("opponents" in state):
            raise ValueError("état de pool d'adversaires présent XOR collecteur avec pool : incohérent")
        if self.opponents is not None:
            self.opponents.load_state_dict(state["opponents"])
        self.generator.set_state(state["generator"].cpu())
        self._obs = cur
        self.agent_steps = int(state["agent_steps"])
        # checkpoints d'avant le pool : tout agent apprenait
        self.learnable_steps = int(state.get("learnable_steps", self.agent_steps))

    def collect(self, policy_version: int) -> RolloutBatch:
        """Joue `n_steps` steps et rend le lot. Le lot rendu reste valide jusqu'au début du collect suivant le suivant
        (deux tampons alternés)."""
        buf = self._buffers[self._next_buffer]
        self._next_buffer ^= 1
        n, d = self.env.n_agents, self.env.obs_dim
        empty_obs, empty_idx = np.zeros((0, d), np.float32), np.zeros(0, np.int64)
        for _ in range(self.n_steps):
            obs = self._obs
            with torch.inference_mode():
                obs_t = torch.from_numpy(obs).to(self.device)
                logits = self.policy(obs_t)
                if self.opponents is not None:
                    logits = self.opponents.override_logits(obs_t, logits)
                act_t, logp_t = sample_actions(logits, self.generator)
                act = act_t.cpu().numpy().astype(np.int64, copy=False)
                logp = logp_t.cpu().numpy().astype(np.float32, copy=False)
            nxt, rew, term, trunc, info = self.env.step(act)
            if self.opponents is not None:
                self.opponents.after_step(term, trunc, info)
            final_obs = np.asarray(info.get("final_obs", empty_obs), dtype=np.float32)
            final_agent = np.asarray(info.get("final_agent", empty_idx), dtype=np.int64)
            buf.add(obs, act, logp, rew, term, trunc, final_obs, final_agent)
            self._obs = np.array(nxt, dtype=np.float32)
            self.agent_steps += n
            self.learnable_steps += self._n_learn
        return buf.finish(self._obs, self.learn_mask, policy_version)
