"""Pads de boost : positions, permutation miroir, permutation RocketGPU <-> RocketSim (ADR-003 §3, parité A3)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from obs.state import N_PADS

I64Array = npt.NDArray[np.int64]
F32Array = npt.NDArray[np.float32]

# Positions (x, y, z) dans l'ordre de RocketGPU : `BoostPadLocSoccar`, vendor/src/RocketGPU/include/rocketgpu/
# detail/constants.hpp:57-94 (indices 0-27 petits pads, 28-33 gros pads, `BoostPadIsBigSoccar` :96-98).
_RG_POSITIONS = (
    (0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0),
    (-940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    (3584.0, -2484.0, 70.0),
    (-1788.0, -2302.0, 70.0),
    (1788.0, -2302.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-1788.0, 2302.0, 70.0),
    (1788.0, 2302.0, 70.0),
    (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0),
    (-940.0, 3308.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (3584.0, 0.0, 73.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
)

# Positions dans l'ordre de RocketSim 2.2.1 : `Arena(GameMode.SOCCAR).get_boost_pads()`, relevé le 2026-11-11 avec
# `.venv/Scripts/python` et `rs.init("vendor/collision_meshes")` (docs/reports/physics_parity.md, A3). RocketSim place
# les pads (±1788, ±2300) là où RocketGPU a (±1788, ±2302) : l'appariement se fait donc au plus proche voisin.
_RS_POSITIONS = (
    (0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0),
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (-940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0),
    (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0),
    (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0),
    (-940.0, 3308.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0),
)

PAD_POSITIONS_ROCKETGPU: F32Array = np.asarray(_RG_POSITIONS, dtype=np.float32)
PAD_POSITIONS_ROCKETSIM: F32Array = np.asarray(_RS_POSITIONS, dtype=np.float32)
PAD_POSITIONS_ROCKETGPU.setflags(write=False)
PAD_POSITIONS_ROCKETSIM.setflags(write=False)
if PAD_POSITIONS_ROCKETGPU.shape != (N_PADS, 3) or PAD_POSITIONS_ROCKETSIM.shape != (N_PADS, 3):
    raise AssertionError("34 pads attendus")


def match_pads(src: F32Array, dst: F32Array, *, tol: float = 4.0) -> I64Array:
    """`perm` tel que `dst[perm[i]]` est le plus proche de `src[i]` (xyz). Lève si ce n'est pas une bijection
    à moins de `tol` uu (le plus gros écart connu entre simulateurs est 2 uu)."""
    d = np.linalg.norm(src[:, None, :].astype(np.float64) - dst[None, :, :].astype(np.float64), axis=-1)
    perm = np.argmin(d, axis=1).astype(np.int64)
    if len(set(perm.tolist())) != len(src) or bool((d[np.arange(len(src)), perm] > tol).any()):
        raise ValueError("appariement de pads non bijectif ou trop lointain")
    return perm


MIRROR_SIGN = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
"""M = diag(-1, -1, 1) : rotation de 180° autour de z, appliquée à tout vecteur monde pour le miroir orange."""

PAD_MIRROR_PERM: I64Array = match_pads(PAD_POSITIONS_ROCKETGPU * MIRROR_SIGN, PAD_POSITIONS_ROCKETGPU, tol=1e-3)
"""`loc[perm[i]] = M·loc[i]` (indices RocketGPU). Involution ; exacte : le terrain est symétrique."""

ROCKETGPU_TO_ROCKETSIM: I64Array = match_pads(PAD_POSITIONS_ROCKETGPU, PAD_POSITIONS_ROCKETSIM)
"""`rs_index = ROCKETGPU_TO_ROCKETSIM[rg_index]`. Réordonner un vecteur de pads RocketSim `v` en ordre RocketGPU :
`v[ROCKETGPU_TO_ROCKETSIM]`."""

ROCKETSIM_TO_ROCKETGPU: I64Array = np.argsort(ROCKETGPU_TO_ROCKETSIM).astype(np.int64)
"""Inverse : `rg_index = ROCKETSIM_TO_ROCKETGPU[rs_index]`."""

for _p in (PAD_MIRROR_PERM, ROCKETGPU_TO_ROCKETSIM, ROCKETSIM_TO_ROCKETGPU):
    _p.setflags(write=False)
