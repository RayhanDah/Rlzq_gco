"""États initiaux d'épisode (`docs/GCO_SPEC.md` §2.3) : kickoffs (proportion `p_kick`) et états génériques non ciblés.

Pur numpy, sans RocketGPU. Le tirage est piloté par un `numpy.random.Generator` seedé : même seed et même suite d'appels
=> mêmes tirages. `p_kick` se modifie en cours de run (annealing piloté de l'extérieur, §2.3).

Les kickoffs eux-mêmes sont produits par le simulateur (5 positions officielles) : on ne tire ici que la graine de son
mélange des positions. Les états génériques sont des positions, vitesses et orientations aléatoires de la balle et des
voitures, un boost aléatoire, et, pour une fraction des arènes, une balle placée près d'une voiture. Aucune mécanique
n'est ciblée (§2.3 interdit : flip-reset-ready, air-dribble setup, AirPlay/AirDrill, replays, setters appris) : toutes
les lois sont indépendantes entre champs, sans condition sur l'état du jeu autre que « pas de recouvrement ».
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

F32Array = npt.NDArray[np.float32]
F64Array = npt.NDArray[np.float64]
U8Array = npt.NDArray[np.uint8]
U64Array = npt.NDArray[np.uint64]
BoolArray = npt.NDArray[np.bool_]
IndexArray = npt.NDArray[np.intp]

# Terrain Soccar (uu). Les marges gardent voitures et balle loin des murs et des buts (y = ±5120 est la ligne de but).
WALL_X = 4096.0
GOAL_LINE_Y = 5120.0
CEILING_Z = 2044.0
BALL_RADIUS = 91.25
CAR_REST_Z = 17.0
"""Hauteur du centre de la voiture posée à plat (à confirmer par la mesure RocketGPU, voir rapport T2)."""

MAX_CAR_SPEED = 2300.0
MAX_CAR_ANG_SPEED = 5.5

CAR_X = 3600.0
CAR_Y = 4600.0
CAR_AIR_Z = (120.0, 1500.0)
BALL_X = 3800.0
BALL_Y = 4800.0
BALL_Z = (BALL_RADIUS + 10.0, 1800.0)
BALL_SPEED_MAX = 2500.0
BALL_ANG_MAX = 3.0
BALL_NEAR_CAR_DIST = (250.0, 2000.0)
MIN_CAR_CAR_DIST = 300.0
MIN_CAR_BALL_DIST = 250.0
P_CAR_AIRBORNE = 0.3
P_BALL_NEAR_CAR = 0.5
P_CAR_HAS_JUMPED = 0.5
P_CAR_SECOND_JUMP_USED = 0.3
"""Parmi les voitures en l'air déjà sautées : proportion ayant consommé double saut ou flip (marginal, pas ciblé)."""
MAX_REDRAWS = 200

KICKOFF_SEED_RANGE = (1, 2**32 - 1)
"""Graine de mélange des positions de kickoff (uint32 non nul : 0 et 0xFFFFFFFF ont un sens spécial côté sim)."""

_STATE_FIELDS = (
    "ball_pos",
    "ball_vel",
    "ball_ang_vel",
    "car_pos",
    "car_vel",
    "car_ang_vel",
    "car_quat",
    "car_boost",
    "car_on_ground",
    "car_has_jumped",
    "car_has_double_jumped",
    "car_has_flipped",
    "car_air_time_since_jump",
)


@dataclass(frozen=True, slots=True)
class InitialStates:
    """États publics de K arènes de C voitures, convention `BatchState` (uu, uu/s, rad/s, quaternion (w, x, y, z))."""

    ball_pos: F32Array
    ball_vel: F32Array
    ball_ang_vel: F32Array
    car_pos: F32Array
    car_vel: F32Array
    car_ang_vel: F32Array
    car_quat: F32Array
    car_boost: F32Array
    car_on_ground: U8Array
    car_has_jumped: U8Array
    car_has_double_jumped: U8Array
    car_has_flipped: U8Array
    car_air_time_since_jump: F32Array

    @property
    def n_arenas(self) -> int:
        return int(self.ball_pos.shape[0])

    @property
    def n_cars(self) -> int:
        return int(self.car_pos.shape[1])

    def check_shapes(self, n_cars: int | None = None) -> None:
        """Lève `ValueError` si une forme ou un dtype est incohérent. Ne regarde pas les valeurs."""
        k = self.ball_pos.shape[0]
        c = self.car_pos.shape[1] if self.car_pos.ndim == 3 else -1
        if n_cars is not None and c != n_cars:
            raise ValueError(f"n_cars attendu {n_cars}, reçu {c}")
        spec: dict[str, tuple[type, tuple[int, ...]]] = {
            "ball_pos": (np.float32, (k, 3)),
            "ball_vel": (np.float32, (k, 3)),
            "ball_ang_vel": (np.float32, (k, 3)),
            "car_pos": (np.float32, (k, c, 3)),
            "car_vel": (np.float32, (k, c, 3)),
            "car_ang_vel": (np.float32, (k, c, 3)),
            "car_quat": (np.float32, (k, c, 4)),
            "car_boost": (np.float32, (k, c)),
            "car_on_ground": (np.uint8, (k, c)),
            "car_has_jumped": (np.uint8, (k, c)),
            "car_has_double_jumped": (np.uint8, (k, c)),
            "car_has_flipped": (np.uint8, (k, c)),
            "car_air_time_since_jump": (np.float32, (k, c)),
        }
        for name, (dtype, shape) in spec.items():
            arr = getattr(self, name)
            if arr.dtype != dtype or arr.shape != shape:
                raise ValueError(
                    f"InitialStates.{name}: attendu {np.dtype(dtype)} {shape}, reçu {arr.dtype} {arr.shape}"
                )


@dataclass(frozen=True, slots=True)
class InitialDraw:
    """Tirage de K conditions initiales : graines de kickoff pour toutes, états explicites pour les génériques."""

    kickoff_seeds: U64Array
    """(K,) graines du mélange de kickoff du simulateur (utilisées pour les arènes en kickoff)."""
    generic_mask: BoolArray
    """(K,) vrai si l'arène démarre sur un état générique (proportion 1 - p_kick)."""
    generic: InitialStates | None
    """États des seules arènes de `generic_mask`, dans l'ordre ; None si aucune."""


def _unit_vectors(rng: np.random.Generator, shape: tuple[int, ...]) -> F64Array:
    v = rng.standard_normal((*shape, 3))
    out: F64Array = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return out


def _f32(x: F64Array) -> F32Array:
    return x.astype(np.float32)


def _u8(x: BoolArray) -> U8Array:
    return x.astype(np.uint8)


def _draw_generic(rng: np.random.Generator, k: int, c: int) -> InitialStates:
    """Un tirage brut de K arènes (sans contrôle de recouvrement)."""
    airborne = rng.random((k, c)) < P_CAR_AIRBORNE
    car_pos = np.empty((k, c, 3), dtype=np.float64)
    car_pos[..., 0] = rng.uniform(-CAR_X, CAR_X, (k, c))
    car_pos[..., 1] = rng.uniform(-CAR_Y, CAR_Y, (k, c))
    car_pos[..., 2] = np.where(airborne, rng.uniform(*CAR_AIR_Z, (k, c)), CAR_REST_Z)

    # Orientation : lacet seul au sol, orientation uniforme sur SO(3) en l'air (quaternion gaussien normalisé).
    yaw = rng.uniform(-np.pi, np.pi, (k, c))
    ground_q = np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=-1)
    air_q = rng.standard_normal((k, c, 4))
    air_q /= np.linalg.norm(air_q, axis=-1, keepdims=True)
    car_quat = np.where(airborne[..., None], air_q, ground_q)

    speed = rng.uniform(0.0, MAX_CAR_SPEED, (k, c))
    heading = rng.uniform(-np.pi, np.pi, (k, c))
    ground_vel = np.stack([speed * np.cos(heading), speed * np.sin(heading), np.zeros_like(speed)], axis=-1)
    air_vel = _unit_vectors(rng, (k, c)) * speed[..., None]
    car_vel = np.where(airborne[..., None], air_vel, ground_vel)
    ang = rng.uniform(-MAX_CAR_ANG_SPEED, MAX_CAR_ANG_SPEED, (k, c, 3))
    car_ang_vel = np.where(airborne[..., None], ang, 0.0)

    has_jumped = airborne & (rng.random((k, c)) < P_CAR_HAS_JUMPED)
    second = has_jumped & (rng.random((k, c)) < P_CAR_SECOND_JUMP_USED)
    flipped = second & (rng.random((k, c)) < 0.5)
    double_jumped = second & ~flipped
    air_time = np.where(has_jumped, rng.uniform(0.0, 2.0, (k, c)), 0.0)
    boost = rng.uniform(0.0, 100.0, (k, c))

    # Balle : position et vitesse aléatoires ; pour une fraction des arènes, près d'une voiture tirée au hasard.
    ball_pos = np.empty((k, 3), dtype=np.float64)
    ball_pos[:, 0] = rng.uniform(-BALL_X, BALL_X, k)
    ball_pos[:, 1] = rng.uniform(-BALL_Y, BALL_Y, k)
    ball_pos[:, 2] = BALL_Z[0] + (BALL_Z[1] - BALL_Z[0]) * rng.random(k) ** 2  # plus de balles basses que hautes
    near = rng.random(k) < P_BALL_NEAR_CAR
    anchor = rng.integers(0, c, k)
    offset = _unit_vectors(rng, (k,)) * rng.uniform(*BALL_NEAR_CAR_DIST, k)[:, None]
    near_pos = car_pos[np.arange(k), anchor] + offset
    near_pos[:, 0] = np.clip(near_pos[:, 0], -BALL_X, BALL_X)
    near_pos[:, 1] = np.clip(near_pos[:, 1], -BALL_Y, BALL_Y)
    near_pos[:, 2] = np.clip(near_pos[:, 2], *BALL_Z)
    ball_pos = np.where(near[:, None], near_pos, ball_pos)
    ball_vel = _unit_vectors(rng, (k,)) * rng.uniform(0.0, BALL_SPEED_MAX, k)[:, None]
    ball_ang = rng.uniform(-BALL_ANG_MAX, BALL_ANG_MAX, (k, 3))

    return InitialStates(
        ball_pos=_f32(ball_pos),
        ball_vel=_f32(ball_vel),
        ball_ang_vel=_f32(ball_ang),
        car_pos=_f32(car_pos),
        car_vel=_f32(car_vel),
        car_ang_vel=_f32(car_ang_vel),
        car_quat=_f32(car_quat),
        car_boost=_f32(boost),
        car_on_ground=_u8(~airborne),
        car_has_jumped=_u8(has_jumped),
        car_has_double_jumped=_u8(double_jumped),
        car_has_flipped=_u8(flipped),
        car_air_time_since_jump=_f32(air_time),
    )


def _overlapping(s: InitialStates) -> BoolArray:
    """Arènes où deux voitures, ou une voiture et la balle, sont trop proches (distances en float64)."""
    car = s.car_pos.astype(np.float64)
    ball = s.ball_pos.astype(np.float64)
    bad: BoolArray = np.linalg.norm(car - ball[:, None, :], axis=-1).min(axis=1) < MIN_CAR_BALL_DIST
    for i in range(s.n_cars):
        for j in range(i + 1, s.n_cars):
            bad |= np.linalg.norm(car[:, i] - car[:, j], axis=-1) < MIN_CAR_CAR_DIST
    return bad


def _assign(dst: InitialStates, idx: IndexArray, src: InitialStates) -> None:
    for name in _STATE_FIELDS:
        getattr(dst, name)[idx] = getattr(src, name)


def sample_generic_states(rng: np.random.Generator, k: int, n_cars: int) -> InitialStates:
    """K états génériques non ciblés, sans recouvrement voiture-voiture ni voiture-balle (arènes redessinées sinon)."""
    if k < 0:
        raise ValueError(f"k doit être >= 0, reçu {k}")
    if n_cars < 2 or n_cars % 2 != 0:
        raise ValueError(f"n_cars doit être pair et >= 2, reçu {n_cars}")
    out = _draw_generic(rng, k, n_cars)
    for _ in range(MAX_REDRAWS):
        bad = np.flatnonzero(_overlapping(out))
        if bad.size == 0:
            return out
        _assign(out, bad, _draw_generic(rng, int(bad.size), n_cars))
    raise RuntimeError(f"recouvrement non résolu après {MAX_REDRAWS} tirages : lois d'états incohérentes")


class InitialStateSampler:
    """Tire les conditions initiales des arènes à réinitialiser : kickoff (proba `p_kick`) ou état générique."""

    __slots__ = ("_n_cars", "_p_kick", "_rng")

    def __init__(self, n_cars: int, p_kick: float, seed: int) -> None:
        if n_cars < 2 or n_cars % 2 != 0:
            raise ValueError(f"n_cars doit être pair et >= 2, reçu {n_cars}")
        self._n_cars = n_cars
        self._p_kick = self._check_p(p_kick)
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _check_p(p_kick: float) -> float:
        p = float(p_kick)
        if not 0.0 <= p <= 1.0:  # NaN => faux
            raise ValueError(f"p_kick doit être dans [0, 1], reçu {p_kick}")
        return p

    @property
    def n_cars(self) -> int:
        return self._n_cars

    @property
    def p_kick(self) -> float:
        return self._p_kick

    def set_p_kick(self, p_kick: float) -> None:
        """Annealing piloté par le trainer (§2.3) : valable pour les tirages suivants."""
        self._p_kick = self._check_p(p_kick)

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def state_dict(self) -> dict[str, Any]:
        """État pour la reprise (T5) : `p_kick` et état du générateur (entiers Python, copie profonde)."""
        return {"p_kick": self._p_kick, "rng": copy.deepcopy(self._rng.bit_generator.state)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(dict(state["rng"]))
        self._rng = rng
        self._p_kick = self._check_p(state["p_kick"])

    def sample(self, k: int) -> InitialDraw:
        if k < 0:
            raise ValueError(f"k doit être >= 0, reçu {k}")
        generic_mask = self._rng.random(k) >= self._p_kick
        seeds = self._rng.integers(*KICKOFF_SEED_RANGE, size=k, dtype=np.uint64)
        n_generic = int(generic_mask.sum())
        generic = sample_generic_states(self._rng, n_generic, self._n_cars) if n_generic else None
        return InitialDraw(kickoff_seeds=seeds, generic_mask=generic_mask, generic=generic)
