"""`BatchState` : tableaux d'état `(n_envs, n_cars, ...)` écrits par `src/env`, lus par `src/obs` (ADR-003 §2).

Pur numpy, sans import de RocketGPU. Les champs sont typiquement des vues sans copie (strided) sur la mémoire du sim :
`BatchState` ne copie jamais, ne conserve aucune vue au-delà du prochain step (responsabilité du consommateur) et
`validate()` ne modifie rien. Repère : monde du sim, z vers le haut, filet bleu en y < 0.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import numpy.typing as npt

from rewards import Team

N_PADS = 34
"""Nombre de pads de boost par env (ADR-003 §2, `detail/constants.hpp:16`)."""

QUAT_NORM_TOL = 1e-3
"""Tolérance sur | ||q|| - 1 |. Une f32 unitaire a une erreur d'arrondi ~1e-7 ; 1e-3 absorbe la dérive d'intégration
entre deux renormalisations du sim sans accepter un quaternion non normalisé (norme 1,01 => échelle d'obs fausse
de 1 %). Valeur de conception, non mesurée sur RocketGPU : à resserrer après mesure en T2."""

# Un pad inactif peut finir un step avec cooldown ≈ -dt (1/120 s) avant son respawn : RocketGPU ne le remet à 0
# qu'au respawn (env_step.hpp:1998-2011, :2531-2534). Marge de 2 ticks ; au-delà, l'état est jugé corrompu.
PAD_COOLDOWN_SLACK_S = 2.0 / 120.0
BOOST_MAX = 100.0
"""Boost maximal (ADR-003 §2 : 0–100)."""

F32Array = npt.NDArray[np.float32]
U8Array = npt.NDArray[np.uint8]
I64Array = npt.NDArray[np.int64]

_F32 = np.dtype(np.float32)
_U8 = np.dtype(np.uint8)
_I64 = np.dtype(np.int64)

# nom -> (dtype, shape logique). 'E' = n_envs, 'C' = n_cars, 'P' = N_PADS, entiers = littéraux.
_SPEC: dict[str, tuple[np.dtype[np.generic], tuple[str | int, ...]]] = {
    "ball_pos": (_F32, ("E", 3)),
    "ball_vel": (_F32, ("E", 3)),
    "ball_ang_vel": (_F32, ("E", 3)),
    "car_pos": (_F32, ("E", "C", 3)),
    "car_vel": (_F32, ("E", "C", 3)),
    "car_ang_vel": (_F32, ("E", "C", 3)),
    "car_quat": (_F32, ("E", "C", 4)),
    "car_boost": (_F32, ("E", "C")),
    "car_on_ground": (_U8, ("E", "C")),
    "car_has_jumped": (_U8, ("E", "C")),
    "car_has_double_jumped": (_U8, ("E", "C")),
    "car_has_flipped": (_U8, ("E", "C")),
    "car_is_demoed": (_U8, ("E", "C")),
    "car_air_time_since_jump": (_F32, ("E", "C")),
    "pad_active": (_U8, ("E", "P")),
    "pad_cooldown": (_F32, ("E", "P")),
    "car_team": (_I64, ("C",)),
}
_BOOL_FIELDS = (
    "car_on_ground",
    "car_has_jumped",
    "car_has_double_jumped",
    "car_has_flipped",
    "car_is_demoed",
    "pad_active",
)


def _shape(symbolic: tuple[str | int, ...], n_envs: int, n_cars: int) -> tuple[int, ...]:
    sizes = {"E": n_envs, "C": n_cars, "P": N_PADS}
    return tuple(sizes[s] if isinstance(s, str) else s for s in symbolic)


def default_car_team(n_cars: int) -> I64Array:
    """Équipe par slot : `k < n_cars/2` bleu, sinon orange (ADR-003 §1). `n_cars` pair et >= 2."""
    if n_cars < 2 or n_cars % 2 != 0:
        raise ValueError(f"n_cars doit être pair et >= 2, reçu {n_cars}")
    team = np.full(n_cars, int(Team.ORANGE), dtype=np.int64)
    team[: n_cars // 2] = int(Team.BLUE)
    return team


@dataclass(frozen=True, slots=True, eq=False)
class BatchState:
    """État du sim pour E envs de C voitures. Unités : uu, uu/s, rad/s, s ; boost 0–100 ; quaternion (w, x, y, z)."""

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
    car_is_demoed: U8Array
    car_air_time_since_jump: F32Array
    pad_active: U8Array
    pad_cooldown: F32Array
    car_team: I64Array

    @property
    def n_envs(self) -> int:
        return int(self.ball_pos.shape[0])

    @property
    def n_cars(self) -> int:
        return int(self.car_team.shape[0])

    def validate(self) -> None:
        """Lève `ValueError` listant les violations (rien n'est modifié). Coût O(taille de l'état)."""
        errors: list[str] = []
        n_envs = self.ball_pos.shape[0] if self.ball_pos.ndim >= 1 else -1
        n_cars = self.car_team.shape[0] if self.car_team.ndim >= 1 else -1
        if n_envs < 1:
            errors.append(f"n_envs doit être >= 1 (ball_pos.shape={self.ball_pos.shape})")
        if n_cars < 2 or n_cars % 2 != 0:
            errors.append(f"n_cars doit être pair et >= 2 (car_team.shape={self.car_team.shape})")

        # Phase 1 : structure (type, dtype, shape). Rien d'autre n'est lu tant qu'elle n'est pas valide.
        for name, (dtype, symbolic) in _SPEC.items():
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                errors.append(f"{name}: ndarray attendu, reçu {type(arr).__name__}")
                continue
            if arr.dtype != dtype:
                errors.append(f"{name}: dtype {arr.dtype} != {dtype}")
            expected = _shape(symbolic, n_envs, n_cars)
            if arr.shape != expected:
                errors.append(f"{name}: shape {arr.shape} != {expected}")
        if errors:
            raise ValueError("BatchState invalide : " + " ; ".join(errors))

        # Phase 2 : valeurs.
        for name, (dtype, _) in _SPEC.items():
            if dtype == _F32 and not bool(np.isfinite(getattr(self, name)).all()):
                errors.append(f"{name}: valeurs non finies (NaN ou inf)")
        for name in _BOOL_FIELDS:
            if not bool(np.isin(getattr(self, name), (0, 1)).all()):
                errors.append(f"{name}: valeurs hors {{0, 1}}")
        norm = np.linalg.norm(self.car_quat.astype(np.float64), axis=-1)
        if not bool((np.abs(norm - 1.0) <= QUAT_NORM_TOL).all()):  # NaN => False
            errors.append(f"car_quat: norme hors 1 ± {QUAT_NORM_TOL:g}")
        if bool(((self.car_boost < 0.0) | (self.car_boost > BOOST_MAX)).any()):
            errors.append(f"car_boost: hors [0, {BOOST_MAX:g}]")
        if bool((self.car_air_time_since_jump < 0.0).any()):
            errors.append("car_air_time_since_jump: valeur négative")
        if bool((self.pad_cooldown < -PAD_COOLDOWN_SLACK_S).any()):
            errors.append(f"pad_cooldown: valeur négative au-delà de {PAD_COOLDOWN_SLACK_S:g} s")
        if not np.array_equal(self.car_team, default_car_team(n_cars)):
            errors.append("car_team: attendu k < C/2 bleu (0), sinon orange (1)")
        if errors:
            raise ValueError("BatchState invalide : " + " ; ".join(errors))

    @classmethod
    def zeros(cls, n_envs: int, n_cars: int) -> BatchState:
        """État valide au repos (quaternion identité, boost 0), pour tests et amorçage. Alloue des tableaux propres."""
        arrays = {
            name: np.zeros(_shape(symbolic, n_envs, n_cars), dtype=dtype) for name, (dtype, symbolic) in _SPEC.items()
        }
        arrays["car_quat"][..., 0] = 1.0
        arrays["car_team"] = default_car_team(n_cars)
        return cls(**arrays)


assert [f.name for f in fields(BatchState)] == list(_SPEC), "_SPEC et BatchState doivent rester alignés"
