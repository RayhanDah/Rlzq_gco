"""Construction vectorisée de l'observation (ADR-003 §3), identique en entraînement et côté RLBot.

Agents à plat `a = e·C + k` (slot de voiture). L'orange voit le terrain comme un bleu : rotation de 180° autour de z
(M = diag(-1, -1, 1)) sur positions, vitesses, vitesses angulaires, orientation et pads. Orientation corps -> monde R
devient M·R : les axes avant/haut sont donc multipliés par M, et les features en repère local sont inchangées.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from actions.table import ACTION_TABLE, N_ACTIONS
from obs import schema as S
from obs.pads import MIRROR_SIGN, PAD_MIRROR_PERM
from obs.state import N_PADS, BatchState

F32Array = npt.NDArray[np.float32]
I64Array = npt.NDArray[np.int64]

_ORANGE = 1  # rewards.Team.ORANGE


def _mirror_sign_car() -> F32Array:
    """(29,) : signe appliqué au bloc voiture d'un agent orange (x, y des vecteurs monde)."""
    sign = np.ones(S.CAR_DIM, dtype=np.float32)
    for sl in S.CAR_WORLD_VECTORS:
        sign[sl] = MIRROR_SIGN
    return sign


_CAR_SIGN = _mirror_sign_car()
_BALL_SIGN = np.tile(MIRROR_SIGN, 3).astype(np.float32)  # (9,)


def _rotation_matrices(q: F32Array) -> F32Array:
    """Quaternions (w, x, y, z) corps -> monde (..., 4) en matrices (..., 3, 3) dont les colonnes sont R·x̂, R·ŷ, R·ẑ
    (`detail/math.hpp:865-869`, `Rotate(q, v) = q v q^-1`)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty((*q.shape[:-1], 3, 3), dtype=np.float32)
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 1, 0] = 2 * (x * y + w * z)
    r[..., 2, 0] = 2 * (x * z - w * y)
    r[..., 0, 1] = 2 * (x * y - w * z)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 2, 1] = 2 * (y * z + w * x)
    r[..., 0, 2] = 2 * (x * z + w * y)
    r[..., 1, 2] = 2 * (y * z - w * x)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def _others_index(n_cars: int, team: I64Array) -> I64Array:
    """(C, C-1) : pour chaque agent, slots des autres voitures, coéquipiers puis adversaires, ordre de slot."""
    idx = np.empty((n_cars, n_cars - 1), dtype=np.int64)
    for k in range(n_cars):
        others = [j for j in range(n_cars) if j != k]
        mates = [j for j in others if team[j] == team[k]]
        foes = [j for j in others if team[j] != team[k]]
        idx[k] = mates + foes
    return idx


def _car_features(state: BatchState) -> F32Array:
    """(E, C, 29) en repère monde non miroité, déjà normalisé."""
    e, c = state.n_envs, state.n_cars
    f = np.empty((e, c, S.CAR_DIM), dtype=np.float32)
    r = _rotation_matrices(state.car_quat)
    rt = np.swapaxes(r, -1, -2)  # R^T : monde -> local
    ang = state.car_ang_vel
    rel_pos = state.ball_pos[:, None, :] - state.car_pos
    rel_vel = state.ball_vel[:, None, :] - state.car_vel

    def local(v: F32Array) -> F32Array:
        res: F32Array = np.einsum("ecij,ecj->eci", rt, v).astype(np.float32, copy=False)
        return res

    f[..., S.CAR_POS] = state.car_pos / S.POS_SCALE
    f[..., S.CAR_FWD] = r[..., :, 0]
    f[..., S.CAR_UP] = r[..., :, 2]
    f[..., S.CAR_VEL] = state.car_vel / S.VEL_SCALE
    f[..., S.CAR_ANG_VEL] = ang / S.CAR_ANG_VEL_SCALE
    f[..., S.CAR_ANG_VEL_LOCAL] = local(ang) / S.CAR_ANG_VEL_SCALE
    f[..., S.CAR_BALL_POS_LOCAL] = local(rel_pos) / S.LOCAL_POS_SCALE
    f[..., S.CAR_BALL_VEL_LOCAL] = local(rel_vel) / S.VEL_SCALE
    on_ground = state.car_on_ground != 0
    has_flip = on_ground | (
        (state.car_has_flipped == 0)
        & (state.car_has_double_jumped == 0)
        & (state.car_air_time_since_jump < S.FLIP_WINDOW_S)
    )  # CarState::HasFlipOrJump, rocketgpu.h:211-213
    f[..., S.CAR_BOOST] = state.car_boost / S.BOOST_SCALE
    f[..., S.CAR_ON_GROUND] = on_ground
    f[..., S.CAR_HAS_FLIP_OR_JUMP] = has_flip
    f[..., S.CAR_IS_DEMOED] = state.car_is_demoed != 0
    f[..., S.CAR_HAS_JUMPED] = state.car_has_jumped != 0
    return f


def build_obs(
    state: BatchState,
    prev_actions: I64Array,
    out: F32Array | None = None,
) -> F32Array:
    """Observation `(n_agents, OBS_DIM)` float32, `n_agents = E·C`.

    `prev_actions` : `(n_agents,)` entiers, indice de la table envoyé au step précédent ; négatif (ex. -1) = aucune
    (après reset) et donne une ligne de zéros. `out` : tampon optionnel `(n_agents, obs_dim(C))` float32 contigu,
    entièrement réécrit puis renvoyé ; sinon un tableau neuf. Ne conserve aucune vue de `state`.
    """
    e, c = state.n_envs, state.n_cars
    n_agents = e * c
    dim = S.obs_dim(c)
    prev = np.asarray(prev_actions)
    if prev.shape != (n_agents,) or prev.dtype.kind not in "iu":
        raise ValueError(f"prev_actions : ({n_agents},) entiers attendus, reçu {prev.dtype} {prev.shape}")
    if n_agents and int(np.max(prev)) >= N_ACTIONS:
        raise ValueError(f"prev_actions : indice >= {N_ACTIONS}")
    if out is None:
        out = np.empty((n_agents, dim), dtype=np.float32)
    elif out.shape != (n_agents, dim) or out.dtype != np.float32 or not out.flags.c_contiguous:
        raise ValueError(f"out : ({n_agents}, {dim}) float32 C-contigu attendu, reçu {out.dtype} {out.shape}")
    o = out.reshape(e, c, dim)  # vue : écrit dans `out`

    orange = state.car_team == _ORANGE  # (C,)
    # Balle (9) : miroir par agent.
    ball = np.concatenate(
        [state.ball_pos / S.POS_SCALE, state.ball_vel / S.VEL_SCALE, state.ball_ang_vel / S.BALL_ANG_VEL_SCALE],
        axis=-1,
    )  # (E, 9)
    ball_sign = np.where(orange[:, None], _BALL_SIGN[None, :], np.float32(1.0)).astype(np.float32)  # (C, 9)
    o[:, :, S.BALL] = ball[:, None, :] * ball_sign[None, :, :]

    # Action précédente (8).
    rows = ACTION_TABLE[np.clip(prev, 0, N_ACTIONS - 1)]
    rows = np.where((prev >= 0)[:, None], rows, np.float32(0.0))
    o[:, :, S.PREV_ACTION] = rows.reshape(e, c, S.PREV_ACTION_DIM)

    # Pads (34) : 1 si actif, sinon 1/(1 + cooldown) ; ordre miroir pour l'orange (out[perm[i]] = val[i], involution).
    cooldown = np.maximum(state.pad_cooldown, np.float32(0.0))  # léger négatif légitime avant respawn (state.py)
    pad_val = np.where(state.pad_active != 0, np.float32(1.0), np.float32(1.0) / (np.float32(1.0) + cooldown))
    pad_idx = np.where(orange[:, None], PAD_MIRROR_PERM[None, :], np.arange(N_PADS)[None, :])  # (C, 34)
    o[:, :, S.PADS] = pad_val[:, pad_idx]

    # Voitures : soi, puis les autres.
    feats = _car_features(state)  # (E, C, 29)
    car_sign = np.where(orange[:, None], _CAR_SIGN[None, :], np.float32(1.0)).astype(np.float32)  # (C, 29)
    o[:, :, S.SELF] = feats * car_sign[None, :, :]
    others = _others_index(c, state.car_team)  # (C, C-1)
    for j in range(c - 1):  # boucle sur les slots (C petit), jamais sur les arènes
        o[:, :, S.OTHERS_START + j * S.CAR_DIM : S.OTHERS_START + (j + 1) * S.CAR_DIM] = (
            feats[:, others[:, j], :] * car_sign[None, :, :]
        )
    return out
