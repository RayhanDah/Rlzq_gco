"""`BatchEnv` : environnement batché, agents à plat `a = e·C + k` (ADR-003 §5). Sans dépendance à RocketGPU.

Ordre d'un step : contrôles et sim -> but (hausse de score) -> touche -> `RewardFunction.step` (une fois, avant tout
reset) -> obs de s' -> pour les arènes finies : `info["final_obs"]`, reset dans le même step, compteurs et action
précédente à 0, nouvelle obs de ces arènes.

Contrat de `build_obs` : `(BatchState, prev_actions (n,) i64, -1 = aucune) -> (n, obs_dim) f32`, n = E'·C où E' est le
nombre d'arènes de l'état passé (peut être un sous-ensemble : l'env rappelle le constructeur sur les seules arènes
réinitialisées).
Il doit retourner un tableau NEUF à chaque appel (l'env en réécrit des lignes), et ne rien conserver de l'état (vues).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import Any

import numpy as np
import numpy.typing as npt

from actions import ACTION_TABLE, N_ACTIONS
from env.initial_states import InitialStateSampler
from env.sim_adapter import Simulator
from obs import build_obs as default_build_obs
from obs.schema import obs_dim as schema_obs_dim
from obs.state import BatchState
from rewards import GoalNet, RewardFunction

F32Array = npt.NDArray[np.float32]
I64Array = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
ObsBuilder = Callable[[BatchState, I64Array], F32Array]


def _subset_state(state: BatchState, arenas: I64Array) -> BatchState:
    """`BatchState` restreint à `arenas` (copie des seules arènes demandées ; `car_team` est commun)."""
    kwargs: dict[str, Any] = {}
    for f in fields(state):
        value = getattr(state, f.name)
        kwargs[f.name] = value if f.name == "car_team" else value[arenas]
    return BatchState(**kwargs)


class BatchEnv:
    """`build_obs` : défaut `obs.build_obs` (obs de `obs.schema.obs_dim(n_cars)` dims). Un constructeur injecté (tests)
    impose `obs_dim`. `action_table` : optionnelle ; la table est la globale `actions.ACTION_TABLE` (écrite dans le sim
    par `actions.write_controls`) : toute autre table est refusée."""

    def __init__(
        self,
        sim: Simulator,
        reward: RewardFunction,
        sampler: InitialStateSampler,
        *,
        build_obs: ObsBuilder | None = None,
        obs_dim: int | None = None,
        action_table: F32Array | None = None,
    ) -> None:
        if action_table is not None and not np.array_equal(action_table, ACTION_TABLE):
            raise ValueError(
                "action_table différente de actions.ACTION_TABLE : refusée (le sim écrit la table globale)"
            )
        if sampler.n_cars != sim.n_cars:
            raise ValueError(f"sampler.n_cars {sampler.n_cars} != sim.n_cars {sim.n_cars}")
        if build_obs is None:
            if obs_dim is not None and obs_dim != schema_obs_dim(sim.n_cars):
                raise ValueError(f"obs_dim {obs_dim} != {schema_obs_dim(sim.n_cars)} (builder par défaut)")
            build_obs, obs_dim = default_build_obs, schema_obs_dim(sim.n_cars)
        elif obs_dim is None:
            raise ValueError("obs_dim est requis avec un build_obs injecté")
        self._sim, self._build_obs, self._reward, self._sampler = sim, build_obs, reward, sampler
        self.n_envs = sim.n_envs
        self.n_cars = sim.n_cars
        self.n_agents = self.n_envs * self.n_cars
        self.n_actions = N_ACTIONS
        self.tick_skip = sim.tick_skip
        self.obs_dim = int(obs_dim)
        self._cfg = reward.config
        self._prev_idx = np.full(self.n_agents, -1, dtype=np.int64)  # indice d'action précédent, -1 = aucune
        self._episode_ticks = np.zeros(self.n_envs, dtype=np.int64)
        self._ticks_since_touch = np.zeros(self.n_envs, dtype=np.int64)
        self._last_toucher = np.full(self.n_envs, -1, dtype=np.int64)
        self._score_blue, self._score_orange = sim.scores()
        self.n_ambiguous_goals = 0
        """Steps où les deux scores ont monté ; non observé en pratique, compté pour le rapport (cf. `_goal_net`)."""

    def set_p_kick(self, p_kick: float) -> None:
        """Annealing piloté par le trainer (spec §2.3)."""
        self._sampler.set_p_kick(p_kick)

    @property
    def p_kick(self) -> float:
        return self._sampler.p_kick

    # ------------------------------------------------------------------------------------------------ reprise (T5)

    def state_dict(self, include_sim: bool = True) -> dict[str, Any]:
        """État pour la reprise : compteurs, scores, tirage des états initiaux et, si `include_sim` et si le simulateur
        sait se sauvegarder (`state_dict`), l'état du simulateur (reprise exacte ; volumineux, voir rapport T5)."""
        sim_state = None
        if include_sim and hasattr(self._sim, "state_dict"):
            sim_state = self._sim.state_dict()
        return {
            "n_envs": self.n_envs,
            "n_cars": self.n_cars,
            "prev_idx": self._prev_idx.copy(),
            "episode_ticks": self._episode_ticks.copy(),
            "ticks_since_touch": self._ticks_since_touch.copy(),
            "last_toucher": self._last_toucher.copy(),
            "score_blue": self._score_blue.copy(),
            "score_orange": self._score_orange.copy(),
            "n_ambiguous_goals": self.n_ambiguous_goals,
            "sampler": self._sampler.state_dict(),
            "sim": sim_state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> F32Array | None:
        """Restaure l'env. État du simulateur présent : reprise exacte, rend None. Absent : reprise « modulo reset des
        arènes » : tirage et compteurs restaurés, puis toutes les arènes repartent d'un nouveau tirage d'états
        initiaux ; rend l'observation courante `(N, D)`, à passer au collecteur."""
        if int(state["n_envs"]) != self.n_envs or int(state["n_cars"]) != self.n_cars:
            raise ValueError(f"env sauvegardé {state['n_envs']}x{state['n_cars']} != {self.n_envs}x{self.n_cars}")
        sim_state = state.get("sim")
        if sim_state is not None and not hasattr(self._sim, "load_state_dict"):
            raise ValueError("état de simulateur sauvegardé mais ce simulateur ne sait pas le restaurer")
        self._sampler.load_state_dict(state["sampler"])
        self.n_ambiguous_goals = int(state["n_ambiguous_goals"])
        if sim_state is not None:
            self._sim.load_state_dict(sim_state)  # type: ignore[attr-defined]
            self._prev_idx = np.array(state["prev_idx"], dtype=np.int64)
            self._episode_ticks = np.array(state["episode_ticks"], dtype=np.int64)
            self._ticks_since_touch = np.array(state["ticks_since_touch"], dtype=np.int64)
            self._last_toucher = np.array(state["last_toucher"], dtype=np.int64)
            self._score_blue = np.array(state["score_blue"], dtype=np.int64)
            self._score_orange = np.array(state["score_orange"], dtype=np.int64)
            return None
        self._reset_arenas(np.arange(self.n_envs, dtype=np.int64))
        return self._as_obs(self._build_obs(self._sim.state(), self._prev_idx))

    # ------------------------------------------------------------------------------------------------ resets

    def _reset_arenas(self, arenas: I64Array) -> None:
        """Remet `arenas` à un nouvel état initial et leurs compteurs à zéro (n'appelle pas `build_obs`)."""
        draw = self._sampler.sample(int(arenas.size))
        self._sim.reset_to_kickoff(arenas, draw.kickoff_seeds)
        if draw.generic is not None:
            self._sim.set_states(arenas[draw.generic_mask], draw.generic)
        self._episode_ticks[arenas] = 0
        self._ticks_since_touch[arenas] = 0
        self._last_toucher[arenas] = -1
        agents = self._agents_of(arenas)
        self._prev_idx[agents] = -1
        self._score_blue, self._score_orange = self._sim.scores()

    def _agents_of(self, arenas: I64Array) -> I64Array:
        return (arenas[:, None] * self.n_cars + np.arange(self.n_cars, dtype=np.int64)).reshape(-1)

    def reset(self, seed: int) -> F32Array:
        """Réinitialise toutes les arènes (graine du tirage des états initiaux) et retourne l'obs `(N, D)`."""
        self._sampler.reseed(seed)
        self._reset_arenas(np.arange(self.n_envs, dtype=np.int64))
        obs = self._as_obs(self._build_obs(self._sim.state(), self._prev_idx))
        return obs

    def _as_obs(self, obs: F32Array) -> F32Array:
        if obs.ndim != 2 or obs.shape[0] != self.n_agents or obs.dtype != np.float32:
            raise ValueError(f"build_obs: ({self.n_agents}, D) float32 attendu, reçu {obs.dtype} {obs.shape}")
        if not obs.flags.writeable:
            obs = obs.copy()
        return obs

    # ------------------------------------------------------------------------------------------------ step

    def _goal_net(self, blue: I64Array, orange: I64Array) -> I64Array:
        d_blue = blue - self._score_blue
        d_orange = orange - self._score_orange
        if bool(((d_blue < 0) | (d_orange < 0)).any()):
            raise RuntimeError("score en baisse : l'env ne remet jamais le tableau des scores à zéro")
        net = np.full(self.n_envs, int(GoalNet.NONE), dtype=np.int64)
        net[d_blue > 0] = int(GoalNet.ORANGE)  # bleu marque dans le filet orange
        net[d_orange > 0] = int(GoalNet.BLUE)
        both = (d_blue > 0) & (d_orange > 0)
        if both.any():  # cas non observé : le plus grand incrément l'emporte, égalité => filet orange
            self.n_ambiguous_goals += int(both.sum())
            net[both] = np.where(d_orange[both] > d_blue[both], int(GoalNet.BLUE), int(GoalNet.ORANGE))
        return net

    def step(self, actions: I64Array) -> tuple[F32Array, F32Array, BoolArray, BoolArray, dict[str, Any]]:
        if actions.shape != (self.n_agents,) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError(f"actions: ({self.n_agents},) entiers attendu, reçu {actions.dtype} {actions.shape}")
        if bool(((actions < 0) | (actions >= self.n_actions)).any()):
            raise ValueError(f"actions hors de [0, {self.n_actions})")
        sim, e, c = self._sim, self.n_envs, self.n_cars
        tick_before = sim.env_ticks()
        sim.step(actions)
        self._prev_idx = actions.copy()
        self._episode_ticks += self.tick_skip

        blue, orange = sim.scores()
        goal_net = self._goal_net(blue, orange)
        self._score_blue, self._score_orange = blue, orange

        touch = sim.touch_ticks()
        touched = touch >= tick_before[:, None]  # -1 (aucune touche) est toujours < tick_before
        any_touch = touched.any(axis=1)
        newest = np.argmax(np.where(touched, touch, -1), axis=1)
        self._last_toucher[any_touch] = newest[any_touch]
        self._ticks_since_touch = np.where(any_touch, 0, self._ticks_since_touch + self.tick_skip)

        state = sim.state()
        scorer = None if self._cfg.rules.team_attribution else self._last_toucher
        out = self._reward.step(goal_net, state.car_team, self._ticks_since_touch, self._episode_ticks, scorer)

        obs = self._as_obs(self._build_obs(state, self._prev_idx))
        done = out.terminated | out.truncated
        info: dict[str, Any] = {
            "goal_net": goal_net,
            "last_toucher": self._last_toucher.copy(),
            "episode_ticks": self._episode_ticks.copy(),
        }
        done_idx = np.flatnonzero(done).astype(np.int64)
        agents = self._agents_of(done_idx)
        info["final_obs"] = obs[agents].copy()
        info["final_agent"] = agents
        if done_idx.size:
            self._reset_arenas(done_idx)
            sub = _subset_state(sim.state(), done_idx)
            fresh = self._as_obs_rows(self._build_obs(sub, self._prev_idx[agents]), agents.size)
            obs[agents] = fresh

        reward = out.rewards.reshape(e * c)
        return obs, reward, np.repeat(out.terminated, c), np.repeat(out.truncated, c), info

    def _as_obs_rows(self, obs: F32Array, n: int) -> F32Array:
        if obs.shape != (n, self.obs_dim) or obs.dtype != np.float32:
            raise ValueError(
                f"build_obs (sous-ensemble): ({n}, {self.obs_dim}) float32 attendu, reçu {obs.dtype} {obs.shape}"
            )
        return obs
