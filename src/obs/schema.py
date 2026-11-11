"""Disposition et normalisation de l'observation (ADR-003 §3). Toute modification incrémente `OBS_SCHEMA_VERSION`."""

from __future__ import annotations

import numpy as np

from obs.state import N_PADS

OBS_SCHEMA_VERSION = 1

BALL_DIM = 9
PREV_ACTION_DIM = 8
PAD_DIM = N_PADS
CAR_DIM = 29
FIXED_DIM = BALL_DIM + PREV_ACTION_DIM + PAD_DIM  # 51

BALL = slice(0, BALL_DIM)
PREV_ACTION = slice(BALL.stop, BALL.stop + PREV_ACTION_DIM)
PADS = slice(PREV_ACTION.stop, PREV_ACTION.stop + PAD_DIM)
SELF = slice(PADS.stop, PADS.stop + CAR_DIM)
OTHERS_START = SELF.stop


def obs_dim(n_cars: int) -> int:
    """`51 + 29·C` (ADR-003 §3) : 109 en 1v1."""
    if n_cars < 2 or n_cars % 2 != 0:
        raise ValueError(f"n_cars doit être pair et >= 2, reçu {n_cars}")
    return FIXED_DIM + CAR_DIM * n_cars


def others_slice(n_cars: int) -> slice:
    """Tranche des `n_cars - 1` autres voitures (coéquipiers puis adversaires, ordre de slot)."""
    return slice(OTHERS_START, obs_dim(n_cars))


N_CARS_V0 = 2
OBS_DIM = obs_dim(N_CARS_V0)  # 109

# Disposition du bloc voiture (29), offsets relatifs au début du bloc.
CAR_POS = slice(0, 3)
CAR_FWD = slice(3, 6)
CAR_UP = slice(6, 9)
CAR_VEL = slice(9, 12)
CAR_ANG_VEL = slice(12, 15)
CAR_ANG_VEL_LOCAL = slice(15, 18)
CAR_BALL_POS_LOCAL = slice(18, 21)
CAR_BALL_VEL_LOCAL = slice(21, 24)
CAR_BOOST = 24
CAR_ON_GROUND = 25
CAR_HAS_FLIP_OR_JUMP = 26
CAR_IS_DEMOED = 27
CAR_HAS_JUMPED = 28
# Vecteurs « monde » du bloc voiture : seuls ceux-ci changent de signe (x, y) sous le miroir orange.
CAR_WORLD_VECTORS = (CAR_POS, CAR_FWD, CAR_UP, CAR_VEL, CAR_ANG_VEL)

# Constantes de normalisation. Provenance : ADR-003 §3 ; plafonds de RocketGPU @f0508a1
# (`include/rocketgpu/rocketgpu.h:638-641` : ball_max_ang_speed 6, car_max_speed 2300, car_max_ang_speed 5.5).
POS_SCALE = np.array([4096.0, 5120.0, 2044.0], dtype=np.float32)  # ADR-003 §3 (largeur, longueur, hauteur)
VEL_SCALE = np.float32(2300.0)  # car_max_speed (H:640)
BALL_ANG_VEL_SCALE = np.float32(6.0)  # ball_max_ang_speed (H:638)
CAR_ANG_VEL_SCALE = np.float32(5.5)  # car_max_ang_speed (H:641)
LOCAL_POS_SCALE = np.float32(5120.0)
"""Position balle - voiture en repère local : échelle scalaire (longueur du terrain). Les échelles par axe monde
(4096, 5120, 2044) n'ont pas de sens dans un repère tourné : écart d'interprétation de « le reste comme la balle »."""
BOOST_SCALE = np.float32(100.0)
FLIP_WINDOW_S = np.float32(1.25)
"""Fenêtre de double saut : `CarState::HasFlipOrJump` (`rocketgpu.h:211-213`)."""
