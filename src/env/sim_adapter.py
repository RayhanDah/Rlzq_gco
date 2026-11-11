"""Interface étroite du simulateur (`Simulator`) et son implémentation RocketGPU (CPU), ADR-003 §5.

Seul fichier de `src/` qui importe RocketGPU (import paresseux, sans stub : `Any`). Les états sont exposés en
`BatchState` de vues sans copie sur `host_struct_array` : valides tant que la mémoire du sim n'est pas réécrite
(step, reset, téléportation) ; aucun consommateur ne les conserve d'un step à l'autre.

Pièges RocketGPU mesurés (tests/integration/test_physics_parity.py, classe `RocketGPUWorld`) :
- `upload_car_states` ignore `.rot` si `basis_authoritative = 1` : on écrit `rot` avec `basis_authoritative = 0`, le sim
  recalcule `basis_*` depuis `rot` (`CanonicalizeImportedCarOrientation`, math.hpp:1306) ;
- `pad_frame_pos_valid` doit être remis à 0 quand on téléporte une voiture ;
- les tableaux de structures se copient octet à octet (`bytearray`), sinon `BatchSim.step` peut planter.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from actions import write_controls
from env.initial_states import InitialStates
from obs.state import N_PADS, BatchState, default_car_team

I64Array = npt.NDArray[np.int64]
U64Array = npt.NDArray[np.uint64]

DEFAULT_DLL_DIR = r"C:\ProgramData\mingw64\mingw64\bin"
KICKOFF_SEED_MODE_FIXED = 1
"""`EnvCommand.kickoff_seed_mode` : le mélange des positions dépend de `kickoff_seed` (rocketgpu.h:597-598)."""


class SimulatorUnavailableError(RuntimeError):
    """RocketGPU (module, DLL ou meshes) introuvable : les tests d'intégration se sautent sur cette exception."""


class Simulator(Protocol):
    """Ce que `BatchEnv` exige du simulateur. Arènes `e` en `[0, n_envs)`, voitures `e·C + k` (slot k)."""

    @property
    def n_envs(self) -> int: ...

    @property
    def n_cars(self) -> int: ...

    @property
    def tick_skip(self) -> int: ...

    def state(self) -> BatchState:
        """Vues vivantes (alias de la mémoire du sim)."""

    def step(self, actions: I64Array) -> None:
        """`actions` (E·C,) indices de `actions.ACTION_TABLE`, tenus tick_skip ticks."""

    def scores(self) -> tuple[I64Array, I64Array]:
        """Copies (E,) des scores (bleu, orange). Monotones : les resets ne les remettent jamais à zéro."""

    def env_ticks(self) -> I64Array:
        """Copie (E,) de l'horloge de chaque arène (ticks physiques, monotone)."""

    def touch_ticks(self) -> I64Array:
        """(E, C) : horodatage (horloge de l'arène) de la dernière touche de balle de chaque voiture, -1 si aucune."""

    def reset_to_kickoff(self, arena_idx: I64Array, kickoff_seeds: U64Array) -> None:
        """Remet les arènes `arena_idx` au kickoff (pads actifs, scores et horloge conservés)."""

    def set_states(self, arena_idx: I64Array, states: InitialStates) -> None:
        """Téléporte les arènes `arena_idx` sur `states` (ligne i de `states` -> arène `arena_idx[i]`)."""


def _import_rocketgpu(py_dir: str | None, dll_dir: str | None) -> tuple[Any, Any]:
    py = py_dir or os.environ.get("ROCKETGPU_PY_DIR", "")
    if not py or not Path(py).is_dir():
        raise SimulatorUnavailableError("ROCKETGPU_PY_DIR absent ou invalide")
    dll = dll_dir or os.environ.get("ROCKETGPU_DLL_DIR", DEFAULT_DLL_DIR)
    if os.name == "nt" and Path(dll).is_dir():
        os.add_dll_directory(dll)
    if py not in sys.path:
        sys.path.insert(0, py)
    try:
        rg = importlib.import_module("rocketgpu_py")
        views = importlib.import_module("rocketgpu_py_cuda")
    except ImportError as exc:
        raise SimulatorUnavailableError(f"import RocketGPU impossible : {exc}") from exc
    if rg.is_cuda_build:
        raise SimulatorUnavailableError("build CUDA : l'adaptateur n'implémente que le build CPU (host_struct_array)")
    return rg, views


def _raw_copy(view: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Copie inscriptible octet à octet d'un tableau de structures (les dtypes RocketGPU ont des trous)."""
    out: npt.NDArray[Any] = np.frombuffer(bytearray(view.tobytes()), dtype=view.dtype)
    return out


class RocketGPUSimulator:
    """RocketGPU CPU, `n_envs` arènes de `n_cars` voitures, `tick_skip` ticks par step."""

    def __init__(
        self,
        n_envs: int,
        n_cars: int = 2,
        tick_skip: int = 8,
        *,
        py_dir: str | None = None,
        dll_dir: str | None = None,
        mesh_dir: str | None = None,
    ) -> None:
        if n_envs < 1:
            raise ValueError(f"n_envs doit être >= 1, reçu {n_envs}")
        if tick_skip < 1:
            raise ValueError(f"tick_skip doit être >= 1, reçu {tick_skip}")
        team = default_car_team(n_cars)  # valide n_cars
        mesh = mesh_dir or os.environ.get("ROCKETGPU_COLLISION_MESH_DIR", "")
        if not mesh or not Path(mesh).is_dir():
            raise SimulatorUnavailableError("ROCKETGPU_COLLISION_MESH_DIR absent ou invalide")
        os.environ["ROCKETGPU_COLLISION_MESH_DIR"] = str(Path(mesh).resolve())
        rg, views = _import_rocketgpu(py_dir, dll_dir)
        self._rg, self._views = rg, views
        self._n_envs, self._n_cars, self._tick_skip = n_envs, n_cars, tick_skip

        cfg = rg.SimConfig()
        cfg.env_count = n_envs
        cfg.max_cars_per_env = n_cars
        self._sim = rg.BatchSim(cfg)
        self._sim.set_env_car_count([n_cars] * n_envs)
        self._sim.set_teams([rg.Team.Blue if t == 0 else rg.Team.Orange for t in team] * n_envs)
        self._sim.reset_to_kickoff(1)
        if self._sim.car_capacity() != n_envs * n_cars:
            raise RuntimeError(f"car_capacity {self._sim.car_capacity()} != {n_envs * n_cars}")

        self._ctrl = np.zeros(self._sim.car_capacity(), dtype=views.state_dtype("controls"))
        self._cars: npt.NDArray[Any] = views.host_struct_array(self._sim, "car_states")
        self._balls: npt.NDArray[Any] = views.host_struct_array(self._sim, "ball_states")
        self._env: npt.NDArray[Any] = views.host_struct_array(self._sim, "env_states")
        self._pads: npt.NDArray[Any] = views.host_struct_array(self._sim, "boost_pad_states")
        if self._pads.shape[0] != n_envs * N_PADS:
            raise RuntimeError(f"boost_pad_states: {self._pads.shape[0]} != {n_envs * N_PADS} (34 pads par arène)")
        self._commands = [rg.EnvCommand() for _ in range(n_envs)]

        e, c = n_envs, n_cars
        cars, balls, pads = self._cars, self._balls, self._pads
        self._state = BatchState(
            ball_pos=balls["pos"],
            ball_vel=balls["vel"],
            ball_ang_vel=balls["ang_vel"],
            car_pos=cars["pos"].reshape(e, c, 3),
            car_vel=cars["vel"].reshape(e, c, 3),
            car_ang_vel=cars["ang_vel"].reshape(e, c, 3),
            car_quat=cars["rot"].reshape(e, c, 4),
            car_boost=cars["boost"].reshape(e, c),
            car_on_ground=cars["is_on_ground"].reshape(e, c),
            car_has_jumped=cars["has_jumped"].reshape(e, c),
            car_has_double_jumped=cars["has_double_jumped"].reshape(e, c),
            car_has_flipped=cars["has_flipped"].reshape(e, c),
            car_is_demoed=cars["is_demoed"].reshape(e, c),
            car_air_time_since_jump=cars["air_time_since_jump"].reshape(e, c),
            pad_active=pads["is_active"].reshape(e, N_PADS),
            pad_cooldown=pads["cooldown"].reshape(e, N_PADS),
            car_team=team,
        )

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def n_cars(self) -> int:
        return self._n_cars

    @property
    def tick_skip(self) -> int:
        return self._tick_skip

    @property
    def raw(self) -> Any:
        """Le `BatchSim` brut (diagnostics et tests uniquement)."""
        return self._sim

    @property
    def raw_arrays(self) -> dict[str, npt.NDArray[Any]]:
        """Tableaux de structures bruts, en lecture seule (diagnostics et tests uniquement)."""
        return {
            "car_states": self._cars,
            "ball_states": self._balls,
            "env_states": self._env,
            "boost_pad_states": self._pads,
        }

    def state(self) -> BatchState:
        return self._state

    def step(self, actions: I64Array) -> None:
        if actions.shape != (self._n_envs * self._n_cars,):
            raise ValueError(f"actions: forme {actions.shape} != {(self._n_envs * self._n_cars,)}")
        write_controls(self._ctrl, actions)  # un seul endroit pour le dtype et l'ordre des champs (src/actions)
        self._sim.step_with_device_controls(self._ctrl.ctypes.data, self._tick_skip, 0)  # tenus tick_skip ticks

    def scores(self) -> tuple[I64Array, I64Array]:
        return self._env["blue_score"].astype(np.int64), self._env["orange_score"].astype(np.int64)

    def env_ticks(self) -> I64Array:
        return self._env["tick"].astype(np.int64)

    def touch_ticks(self) -> I64Array:
        cars = self._cars.reshape(self._n_envs, self._n_cars)
        valid = cars["ball_hit_info.is_valid"] != 0
        out: I64Array = np.where(valid, cars["ball_hit_info.tick_count_when_hit"].astype(np.int64), -1)
        return out

    def reset_to_kickoff(self, arena_idx: I64Array, kickoff_seeds: U64Array) -> None:
        if arena_idx.size == 0:
            return
        if kickoff_seeds.shape != arena_idx.shape:
            raise ValueError("kickoff_seeds doit avoir la forme de arena_idx")
        cmds = self._commands
        for e, seed in zip(arena_idx.tolist(), kickoff_seeds.tolist(), strict=True):
            c = cmds[e]
            c.reset_to_kickoff = 1
            c.kickoff_seed_mode = KICKOFF_SEED_MODE_FIXED
            c.kickoff_seed = int(seed)
        self._sim.apply_env_commands(cmds[: int(np.max(arena_idx)) + 1])
        for e in arena_idx.tolist():
            cmds[e].reset_to_kickoff = 0  # keep_boost_pads, reset_scoreboard, reset_tick_count restent à 0

    def set_states(self, arena_idx: I64Array, states: InitialStates) -> None:
        k = int(arena_idx.size)
        if k == 0:
            return
        states.check_shapes(self._n_cars)
        if states.n_arenas != k:
            raise ValueError(f"{states.n_arenas} états pour {k} arènes")
        e, c = self._n_envs, self._n_cars
        cars = _raw_copy(self._cars)
        grid = cars.reshape(e, c)
        grid["pos"][arena_idx] = states.car_pos
        grid["rot"][arena_idx] = states.car_quat
        grid["vel"][arena_idx] = states.car_vel
        grid["ang_vel"][arena_idx] = states.car_ang_vel
        grid["boost"][arena_idx] = states.car_boost
        grid["is_on_ground"][arena_idx] = states.car_on_ground
        grid["has_jumped"][arena_idx] = states.car_has_jumped
        grid["has_double_jumped"][arena_idx] = states.car_has_double_jumped
        grid["has_flipped"][arena_idx] = states.car_has_flipped
        grid["air_time_since_jump"][arena_idx] = states.car_air_time_since_jump
        grid["air_time"][arena_idx] = states.car_air_time_since_jump
        grid["basis_authoritative"][arena_idx] = 0  # le sim recalcule basis_* depuis rot
        grid["tick_count_since_update"][arena_idx] = 0  # pas de restauration des champs « reportés » (rocketgpu.h:374)
        grid["pad_frame_pos_valid"][arena_idx] = 0
        grid["boost_pad_touching"][arena_idx] = 0
        self._sim.upload_state_array("car_states", cars.ctypes.data, cars.shape[0])

        balls = _raw_copy(self._balls)
        balls["pos"][arena_idx] = states.ball_pos
        balls["vel"][arena_idx] = states.ball_vel
        balls["ang_vel"][arena_idx] = states.ball_ang_vel
        balls["basis_forward"][arena_idx] = (1.0, 0.0, 0.0)
        balls["basis_right"][arena_idx] = (0.0, 1.0, 0.0)
        balls["basis_up"][arena_idx] = (0.0, 0.0, 1.0)
        balls["basis_authoritative"][arena_idx] = 1
        balls["sleeping"][arena_idx] = 0
        balls["deactivation_time"][arena_idx] = 0.0
        balls["constrained_2d"][arena_idx] = 0
        self._sim.upload_state_array("ball_states", balls.ctypes.data, balls.shape[0])
