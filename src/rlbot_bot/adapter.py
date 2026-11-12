"""Adaptateur RLBot v5 -> `BatchState` (une arène) et indice d'action -> `ControllerState` (ADR-004).

Types de référence : `rlbot_flatbuffers` 0.19.0 (paquet `rlbot` 2.0.0b56), `rlbot_flatbuffers/__init__.pyi`. Ce module
ne lit les paquets que par attributs (`packet.players[i].physics.location.x`...) ; RLBot n'est importé, paresseusement,
que pour fabriquer un `ControllerState`. Le suivi d'état se limite à l'action précédente et au décompte du tick-skip :
tous les champs de `BatchState` se déduisent du paquet courant (table de correspondance : ADR-004 §2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from actions import ACTION_TABLE, N_ACTIONS
from obs.builder import build_obs
from obs.pads import PAD_POSITIONS_ROCKETGPU, match_pads
from obs.state import BOOST_MAX, N_PADS, BatchState, default_car_team

if TYPE_CHECKING:
    from rlbot import flat

F32Array = npt.NDArray[np.float32]
F64Array = npt.NDArray[np.float64]
I64Array = npt.NDArray[np.int64]

DEFAULT_TICK_SKIP = 8
DOUBLE_JUMP_WINDOW_S = 1.25
"""Fenêtre de double saut après la fin de la poussée du saut : `CarState::HasFlipOrJump` (rocketgpu.h:211-213),
et `PlayerInfo.dodge_timeout` du stub RLBot (« 1.25 seconds after the first jump plus up to 0.2 seconds »)."""
PAD_RESPAWN_S = {True: 10.0, False: 4.0}
"""Réapparition d'un pad (stub RLBot, `BoostPad.is_full_boost` : 10 s gros, 4 s petit)."""

# Valeurs de `AirState` / `MatchPhase` (stub : `int(AirState.OnGround) == 0`, ..., `int(MatchPhase.Kickoff) == 2`).
AIR_ON_GROUND = 0
AIR_JUMPING = 1
PHASE_KICKOFF = 2
PHASE_ACTIVE = 3
_PLAYABLE_PHASES = (PHASE_KICKOFF, PHASE_ACTIVE)


def euler_to_quat(pitch: F64Array, yaw: F64Array, roll: F64Array) -> F64Array:
    """Angles d'Euler RLBot (rad) -> quaternion (w, x, y, z) corps -> monde, forme `(..., 4)`.

    Convention (stub `Rotator` : pitch + = nez en haut, yaw 0 = +x et + vers +y, roll + = penché à droite ; ordre
    Unreal ZYX intrinsèque) : colonnes de la matrice R = (avant, axe y du corps, haut) avec
    avant = (cp·cy, cp·sy, sp), y = (cy·sp·sr - cr·sy, sy·sp·sr + cr·cy, -cp·sr), haut = (-cr·cy·sp - sr·sy,
    -cr·sy·sp + sr·cy, cp·cr). Même repère numérique que `obs.builder._rotation_matrices` (R·x̂ = avant).
    """
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    m = np.empty((*np.shape(pitch), 3, 3), dtype=np.float64)
    m[..., 0, 0], m[..., 1, 0], m[..., 2, 0] = cp * cy, cp * sy, sp
    m[..., 0, 1], m[..., 1, 1], m[..., 2, 1] = cy * sp * sr - cr * sy, sy * sp * sr + cr * cy, -cp * sr
    m[..., 0, 2], m[..., 1, 2], m[..., 2, 2] = -cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr
    return _matrix_to_quat(m)


def _matrix_to_quat(m: F64Array) -> F64Array:
    """Matrice de rotation `(..., 3, 3)` -> quaternion (w, x, y, z) unitaire (branche du plus grand pivot)."""
    q = np.empty((*m.shape[:-2], 4), dtype=np.float64)
    flat_m = m.reshape(-1, 3, 3)
    flat_q = q.reshape(-1, 4)
    for i, mm in enumerate(flat_m):
        tr = mm[0, 0] + mm[1, 1] + mm[2, 2]
        if tr >= max(mm[0, 0], mm[1, 1], mm[2, 2]):
            s = 2.0 * np.sqrt(1.0 + tr)
            flat_q[i] = (0.25 * s, (mm[2, 1] - mm[1, 2]) / s, (mm[0, 2] - mm[2, 0]) / s, (mm[1, 0] - mm[0, 1]) / s)
        elif mm[0, 0] >= mm[1, 1] and mm[0, 0] >= mm[2, 2]:
            s = 2.0 * np.sqrt(1.0 + mm[0, 0] - mm[1, 1] - mm[2, 2])
            flat_q[i] = ((mm[2, 1] - mm[1, 2]) / s, 0.25 * s, (mm[0, 1] + mm[1, 0]) / s, (mm[0, 2] + mm[2, 0]) / s)
        elif mm[1, 1] >= mm[2, 2]:
            s = 2.0 * np.sqrt(1.0 - mm[0, 0] + mm[1, 1] - mm[2, 2])
            flat_q[i] = ((mm[0, 2] - mm[2, 0]) / s, (mm[0, 1] + mm[1, 0]) / s, 0.25 * s, (mm[1, 2] + mm[2, 1]) / s)
        else:
            s = 2.0 * np.sqrt(1.0 - mm[0, 0] - mm[1, 1] + mm[2, 2])
            flat_q[i] = ((mm[1, 0] - mm[0, 1]) / s, (mm[0, 2] + mm[2, 0]) / s, (mm[1, 2] + mm[2, 1]) / s, 0.25 * s)
    flat_q /= np.linalg.norm(flat_q, axis=-1, keepdims=True)
    return q


def controls_from_action(idx: int) -> flat.ControllerState:
    """Ligne `idx` de la table -> `ControllerState` RLBot (axes float, boutons bool). `use_item` reste faux."""
    from rlbot import flat  # import paresseux : seul point d'entrée de RLBot dans ce module

    if not 0 <= idx < N_ACTIONS:
        raise ValueError(f"indice d'action hors [0, {N_ACTIONS}) : {idx}")
    t, s, p, y, r, j, b, h = (float(v) for v in ACTION_TABLE[idx])
    return flat.ControllerState(
        throttle=t, steer=s, pitch=p, yaw=y, roll=r, jump=bool(j), boost=bool(b), handbrake=bool(h), use_item=False
    )


def idle_controls() -> flat.ControllerState:
    """Contrôles inertes (hors phase jouable)."""
    from rlbot import flat

    return flat.ControllerState()


def action_from_controls(ctrl: Any) -> int:
    """Inverse de `controls_from_action` : indice de la ligne égale aux contrôles (exact), sinon `ValueError`."""
    row = np.array(
        [ctrl.throttle, ctrl.steer, ctrl.pitch, ctrl.yaw, ctrl.roll, ctrl.jump, ctrl.boost, ctrl.handbrake],
        dtype=np.float32,
    )
    hits = np.flatnonzero((ACTION_TABLE == row).all(axis=1))
    if hits.size != 1 or ctrl.use_item:
        raise ValueError(f"contrôles absents de la table : {row.tolist()}")
    return int(hits[0])


class PacketAdapter:
    """Convertit les `GamePacket` RLBot en observation d'un agent et gère l'action tenue `tick_skip` ticks.

    `field_info` : `FieldInfo` reçu à la connexion (ordre des pads : y puis x, stub `FieldInfo.boost_pads`). `index` :
    indice de la voiture du bot dans `packet.players`. Les voitures sont rangées par équipe (bleus d'abord) puis par
    indice, comme `default_car_team` ; il faut autant de joueurs dans chaque équipe.
    """

    def __init__(
        self,
        field_info: Any,
        index: int,
        tick_skip: int = DEFAULT_TICK_SKIP,
        *,
        validate: bool = False,
    ) -> None:
        if tick_skip < 1:
            raise ValueError(f"tick_skip doit être >= 1, reçu {tick_skip}")
        pads = field_info.boost_pads
        if len(pads) != N_PADS:
            raise ValueError(f"{len(pads)} pads dans FieldInfo, {N_PADS} attendus")
        xy = np.array([[p.location.x, p.location.y, 0.0] for p in pads], dtype=np.float32)
        ref = PAD_POSITIONS_ROCKETGPU.copy()
        ref[:, 2] = 0.0  # appariement en xy : l'altitude des pads n'est pas fiable entre sources (z = 70 ou 73)
        self._pad_perm: I64Array = match_pads(xy, ref)  # pad RLBot i -> indice RocketGPU perm[i]
        self._pad_respawn = np.array([PAD_RESPAWN_S[bool(p.is_full_boost)] for p in pads], dtype=np.float32)
        self.index = index
        self.tick_skip = tick_skip
        self._validate = validate
        self.reset()

    def reset(self) -> None:
        """Nouvel épisode : plus d'action précédente (indice -1 = bloc de zéros, comme après un reset)."""
        self.prev_action = -1
        self._held: int | None = None
        self._last_decision_frame: int | None = None

    def to_state(self, packet: Any) -> tuple[BatchState, int]:
        """`(BatchState de 1 arène, slot du bot)`. Pure : n'utilise ni n'écrit aucun suivi d'état."""
        players = packet.players
        if not 0 <= self.index < len(players):
            raise ValueError(f"index {self.index} hors de packet.players (n = {len(players)})")
        order = sorted(range(len(players)), key=lambda i: (int(players[i].team), i))
        n = len(order)
        team = default_car_team(n)
        if [int(players[i].team) for i in order] != team.tolist():
            raise ValueError("équipes non équilibrées : autant de joueurs bleus et orange attendus")
        st = BatchState.zeros(1, n)
        ball = packet.balls[0].physics
        st.ball_pos[0] = (ball.location.x, ball.location.y, ball.location.z)
        st.ball_vel[0] = (ball.velocity.x, ball.velocity.y, ball.velocity.z)
        st.ball_ang_vel[0] = (ball.angular_velocity.x, ball.angular_velocity.y, ball.angular_velocity.z)
        rot = np.array(
            [[players[i].physics.rotation.pitch, players[i].physics.rotation.yaw, players[i].physics.rotation.roll]
             for i in order],
            dtype=np.float64,
        )  # fmt: skip
        st.car_quat[0] = euler_to_quat(rot[:, 0], rot[:, 1], rot[:, 2])
        for slot, i in enumerate(order):
            p = players[i]
            ph = p.physics
            st.car_pos[0, slot] = (ph.location.x, ph.location.y, ph.location.z)
            st.car_vel[0, slot] = (ph.velocity.x, ph.velocity.y, ph.velocity.z)
            st.car_ang_vel[0, slot] = (ph.angular_velocity.x, ph.angular_velocity.y, ph.angular_velocity.z)
            st.car_boost[0, slot] = min(max(float(p.boost), 0.0), BOOST_MAX)
            air = int(p.air_state)
            st.car_on_ground[0, slot] = air == AIR_ON_GROUND
            st.car_has_jumped[0, slot] = bool(p.has_jumped)
            st.car_has_double_jumped[0, slot] = bool(p.has_double_jumped)
            st.car_has_flipped[0, slot] = bool(p.has_dodged)
            st.car_is_demoed[0, slot] = float(p.demolished_timeout) > 0.0
            st.car_air_time_since_jump[0, slot] = self._air_time_since_jump(p, air)
        if len(packet.boost_pads) != N_PADS:
            raise ValueError(f"{len(packet.boost_pads)} états de pads, {N_PADS} attendus")
        active = np.array([bool(s.is_active) for s in packet.boost_pads], dtype=np.uint8)
        timer = np.array([float(s.timer) for s in packet.boost_pads], dtype=np.float32)
        cooldown = np.where(active != 0, np.float32(0.0), np.maximum(self._pad_respawn - timer, np.float32(0.0)))
        st.pad_active[0, self._pad_perm] = active
        st.pad_cooldown[0, self._pad_perm] = cooldown.astype(np.float32)
        if self._validate:
            st.validate()
        return st, order.index(self.index)

    @staticmethod
    def _air_time_since_jump(p: Any, air: int) -> float:
        """Temps écoulé depuis la fin de la poussée du saut, reconstruit de `dodge_timeout` (ADR-004 §2).

        `dodge_timeout` = 1,25 - temps depuis la fin du saut tant que le double saut est possible, sinon -1.
        Hors fenêtre : 0 au sol, pendant la poussée (Jumping) ou sans saut (chute) ; 1,25 (expirée ou consommée) si
        la voiture a sauté. Seule la comparaison à 1,25 compte pour l'obs (`has_flip_or_jump`).
        """
        if air in (AIR_ON_GROUND, AIR_JUMPING):
            return 0.0
        timeout = float(p.dodge_timeout)
        if timeout >= 0.0:
            return DOUBLE_JUMP_WINDOW_S - timeout
        return DOUBLE_JUMP_WINDOW_S if bool(p.has_jumped) else 0.0

    def observe(self, packet: Any) -> F32Array:
        """Observation `(OBS_DIM,)` du bot pour ce paquet, avec l'action précédente suivie par l'adaptateur.

        Les autres voitures reçoivent -1 : l'obs d'un agent ne lit que sa propre action précédente (builder.py).
        """
        st, slot = self.to_state(packet)
        prev = np.full(st.n_cars, -1, dtype=np.int64)
        prev[slot] = self.prev_action
        obs: F32Array = build_obs(st, prev)[slot]
        return obs

    def step(self, packet: Any, policy: Callable[[F32Array], int]) -> flat.ControllerState:
        """Contrôles à envoyer pour ce paquet : nouvelle décision tous les `tick_skip` ticks, sinon action tenue.

        Hors Kickoff/Active (compte à rebours, but, replay...) : suivi remis à zéro, contrôles nuls. Les paquets
        manqués sont gérés par `match_info.frame_num` (peut avancer de plus de 1) : on décide dès que
        `frame_num - dernière décision >= tick_skip`, ou si le compteur recule (nouveau match).
        """
        if int(packet.match_info.match_phase) not in _PLAYABLE_PHASES:
            self.reset()
            return idle_controls()
        frame = int(packet.match_info.frame_num)
        last = self._last_decision_frame
        if self._held is None or last is None or frame < last or frame - last >= self.tick_skip:
            action = int(policy(self.observe(packet)))  # observe() lit prev_action : la décision précédente
            if not 0 <= action < N_ACTIONS:
                raise ValueError(f"politique : indice {action} hors [0, {N_ACTIONS})")
            self._held, self._last_decision_frame = action, frame
            self.prev_action = action
        return controls_from_action(self._held)
