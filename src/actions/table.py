"""Table d'actions discrètes (ADR-003 §4) et écriture dans le tampon de contrôles de RocketGPU.

Colonnes `[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]`. Aucun miroir d'action : les contrôles sont
exprimés dans le repère de la voiture, invariants par la rotation de 180° autour de z.
"""

from __future__ import annotations

import hashlib

import numpy as np
import numpy.typing as npt

ACTION_SCHEMA_VERSION = 1
N_ACTIONS = 90
CONTROL_FIELDS = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")

F32Array = npt.NDArray[np.float32]
I64Array = npt.NDArray[np.int64]


def _build_table() -> F32Array:
    rows: list[list[float]] = []
    # Sol (24) : throttle, steer, boost, handbrake imbriqués dans cet ordre ; boost = 1 seulement avec throttle = 1
    # (RocketGPU freine au sol si on booste sans gaz : écart de physique A1).
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    rows.append([throttle, steer, 0, steer, 0, 0, boost, handbrake])
    # Air (66) : pitch, yaw, roll, jump, boost ; throttle = boost (propulsion aérienne) ; steer = yaw.
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = int(jump == 1 and (pitch != 0 or yaw != 0 or roll != 0))
                        rows.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    table = np.asarray(rows, dtype=np.float32)
    if table.shape != (N_ACTIONS, len(CONTROL_FIELDS)):
        raise AssertionError(f"table d'actions : {table.shape}")
    table.setflags(write=False)
    return table


ACTION_TABLE: F32Array = _build_table()
"""`(90, 8)` float32, lecture seule."""

ACTION_TABLE_SHA256 = "86baa15c48c42c497f3ea0fe62efeb49e4a8241cb3191957822e453cd2d0b655"
"""sha256 des octets (C-order, float32 little-endian) de `ACTION_TABLE`. Figé ; voir `table_sha256()`."""


def table_sha256(table: F32Array = ACTION_TABLE) -> str:
    return hashlib.sha256(np.ascontiguousarray(table, dtype="<f4").tobytes()).hexdigest()


# Disposition du tampon de contrôles de RocketGPU : struct `CarControls`, rocketgpu.h:116-125 ; offsets exposés par
# rocketgpu_bindings.cpp:589-599 (`offsetof`). 5 f32 puis 3 u8, taille 24 (alignement 4).
CONTROLS_DTYPE = np.dtype(
    {
        "names": list(CONTROL_FIELDS),
        "formats": ["<f4"] * 5 + ["u1"] * 3,
        "offsets": [0, 4, 8, 12, 16, 20, 21, 22],
        "itemsize": 24,
    }
)

_AXES: F32Array = np.ascontiguousarray(ACTION_TABLE[:, :5])
_BITS: npt.NDArray[np.uint8] = np.ascontiguousarray(ACTION_TABLE[:, 5:].astype(np.uint8))


def _check_controls_dtype(dtype: np.dtype[np.generic]) -> None:
    fields = dtype.fields
    if dtype.itemsize != 24 or fields is None:
        raise ValueError(f"tampon de contrôles : dtype inattendu {dtype}")
    want = CONTROLS_DTYPE.fields
    assert want is not None
    for name in CONTROL_FIELDS:
        if name not in fields or fields[name][:2] != want[name][:2]:
            raise ValueError(f"tampon de contrôles : champ {name} différent de {want[name][:2]} ({dtype})")


def write_controls(ctrl: npt.NDArray[np.void], idx: I64Array) -> None:
    """Écrit les lignes de la table choisies par `idx` (n,) dans `ctrl[:n]` (slots de voiture, `e·C + k`).

    `ctrl` : `np.zeros(car_capacity, dtype=state_dtype("controls"))`. Les slots au-delà de `n` ne sont pas touchés.
    """
    idx = np.asarray(idx)
    if idx.ndim != 1 or idx.dtype.kind not in "iu":
        raise ValueError(f"idx : tableau 1D d'entiers attendu, reçu {idx.dtype} {idx.shape}")
    n = idx.shape[0]
    if n > ctrl.shape[0]:
        raise ValueError(f"{n} indices pour {ctrl.shape[0]} slots de contrôle")
    if n and (int(np.min(idx)) < 0 or int(np.max(idx)) >= N_ACTIONS):
        raise ValueError(f"indice d'action hors [0, {N_ACTIONS})")
    _check_controls_dtype(ctrl.dtype)
    axes = _AXES[idx]
    bits = _BITS[idx]
    for j, name in enumerate(CONTROL_FIELDS[:5]):
        ctrl[name][:n] = axes[:, j]
    for j, name in enumerate(CONTROL_FIELDS[5:]):
        ctrl[name][:n] = bits[:, j]
