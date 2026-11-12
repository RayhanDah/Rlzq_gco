"""Adaptateur RLBot v5 : conversions, correspondance des champs, suivi tick-skip (sans RocketGPU ni jeu)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("rlbot")
from rlbot import flat

from actions import ACTION_TABLE, N_ACTIONS
from obs import schema as S
from obs.pads import PAD_POSITIONS_ROCKETGPU, PAD_POSITIONS_ROCKETSIM
from obs.state import N_PADS
from rlbot_bot import PacketAdapter, action_from_controls, controls_from_action, euler_to_quat

F32 = npt.NDArray[np.float32]
_ORDER = np.lexsort((PAD_POSITIONS_ROCKETSIM[:, 0], PAD_POSITIONS_ROCKETSIM[:, 1]))  # y puis x
_FULL = PAD_POSITIONS_ROCKETSIM[:, 2] == 73.0


def field_info(order: npt.NDArray[np.int64] = _ORDER) -> flat.FieldInfo:
    pads = [
        flat.BoostPad(flat.Vector3(*(float(v) for v in PAD_POSITIONS_ROCKETSIM[i])), is_full_boost=bool(_FULL[i]))
        for i in order
    ]
    return flat.FieldInfo(boost_pads=pads)


def player(
    team: int, *, x: float = 0.0, rot: tuple[float, float, float] = (0.0, 0.0, 0.0), **kw: object
) -> flat.PlayerInfo:
    phys = flat.Physics(flat.Vector3(x, 0.0, 17.0), flat.Rotator(*rot), flat.Vector3(), flat.Vector3())
    args: dict[str, object] = {"boost": 33.0, "air_state": flat.AirState.OnGround, **kw}
    return flat.PlayerInfo(physics=phys, team=team, **args)  # type: ignore[arg-type]


def packet(
    players: list[flat.PlayerInfo] | None = None,
    frame: int = 0,
    phase: flat.MatchPhase | None = None,
    pads: list[flat.BoostPadState] | None = None,
) -> flat.GamePacket:
    ball = flat.BallInfo(physics=flat.Physics(flat.Vector3(0, 0, 93), flat.Rotator(), flat.Vector3(), flat.Vector3()))
    return flat.GamePacket(
        players=players if players is not None else [player(0, x=-100.0), player(1, x=100.0)],
        boost_pads=pads if pads is not None else [flat.BoostPadState(True, 0.0) for _ in range(N_PADS)],
        balls=[ball],
        match_info=flat.MatchInfo(frame_num=frame, match_phase=phase if phase is not None else flat.MatchPhase.Active),
    )


def rotation(q: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


# --- orientation -----------------------------------------------------------------------------------------------------


def test_euler_semantique_documentee() -> None:
    """Stub `Rotator` : pitch + = nez en haut ; yaw 0 = +x, yaw + vers +y ; roll + = penché à droite (axe y du corps,
    côté droit dans le repère gauche de RL, vers le bas)."""
    r = rotation(euler_to_quat(np.array(0.0), np.array(0.0), np.array(0.0)))
    assert np.allclose(r, np.eye(3), atol=1e-12)
    nose_up = rotation(euler_to_quat(np.array(0.5), np.array(0.0), np.array(0.0)))
    assert nose_up[2, 0] == pytest.approx(np.sin(0.5)) and nose_up[0, 0] > 0
    yaw = rotation(euler_to_quat(np.array(0.0), np.array(np.pi / 2), np.array(0.0)))
    assert np.allclose(yaw[:, 0], (0, 1, 0), atol=1e-12)
    roll = rotation(euler_to_quat(np.array(0.0), np.array(0.0), np.array(0.3)))
    assert roll[2, 1] == pytest.approx(-np.sin(0.3))  # axe y du corps vers le bas
    assert roll[2, 2] == pytest.approx(np.cos(0.3))


def test_euler_donne_une_rotation_propre() -> None:
    rng = np.random.default_rng(0)
    n = 200
    p, y, r = rng.uniform(-1.5, 1.5, n), rng.uniform(-np.pi, np.pi, n), rng.uniform(-np.pi, np.pi, n)
    q = euler_to_quat(p, y, r)
    assert q.shape == (n, 4) and np.allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-12)
    for qi in q:
        m = rotation(qi)
        assert np.allclose(m.T @ m, np.eye(3), atol=1e-12) and np.linalg.det(m) == pytest.approx(1.0)


# --- actions ---------------------------------------------------------------------------------------------------------


def test_aller_retour_des_90_actions_et_boost_plein_gaz() -> None:
    for idx in range(N_ACTIONS):
        c = controls_from_action(idx)
        assert action_from_controls(c) == idx
        assert isinstance(c.jump, bool) and isinstance(c.boost, bool) and not c.use_item
        assert np.array_equal(
            np.array([c.throttle, c.steer, c.pitch, c.yaw, c.roll, c.jump, c.boost, c.handbrake], dtype=np.float32),
            ACTION_TABLE[idx],
        )
    assert all(ACTION_TABLE[i, 0] == 1.0 for i in range(24) if ACTION_TABLE[i, 6] == 1.0)  # sol : boost => plein gaz
    assert all(ACTION_TABLE[i, 0] == ACTION_TABLE[i, 6] for i in range(24, N_ACTIONS))  # air : throttle = boost


def test_action_hors_table_refusee() -> None:
    with pytest.raises(ValueError):
        action_from_controls(flat.ControllerState(throttle=0.5))
    with pytest.raises(ValueError):
        controls_from_action(N_ACTIONS)
    with pytest.raises(ValueError):
        action_from_controls(flat.ControllerState(use_item=True))


# --- BatchState ------------------------------------------------------------------------------------------------------


def test_permutation_des_pads_independante_de_l_ordre_du_paquet() -> None:
    """Les pads RLBot (ordre y puis x, ±2300) sont remis dans l'ordre RocketGPU (±2302) par position."""
    rng = np.random.default_rng(1)
    base = PacketAdapter(field_info(), 0)
    shuffled_order = rng.permutation(N_PADS)
    other = PacketAdapter(field_info(_ORDER[shuffled_order]), 0)
    cool = rng.uniform(0.1, 3.9, N_PADS)
    pads = [flat.BoostPadState(False, 4.0 - float(c)) for c in cool]  # petits et gros pads : timer < 4 reste valide
    st_a, _ = base.to_state(packet(pads=pads))
    st_b, _ = other.to_state(packet(pads=[pads[i] for i in shuffled_order]))
    assert np.array_equal(st_a.pad_cooldown, st_b.pad_cooldown)
    # le pad 0 de RLBot (y le plus bas) est à (0, -4240) = indice 0 de RocketGPU
    one = [flat.BoostPadState(True, 0.0)] * N_PADS
    one[0] = flat.BoostPadState(False, 1.0)
    st, _ = base.to_state(packet(pads=one))
    j = int(np.flatnonzero(st.pad_active[0] == 0)[0])
    assert np.allclose(PAD_POSITIONS_ROCKETGPU[j, :2], (0.0, -4240.0))
    assert st.pad_cooldown[0, j] == pytest.approx(3.0)  # petit pad : 4 s - 1 s


def test_timer_gros_pad_10_secondes() -> None:
    ad = PacketAdapter(field_info(), 0)
    big = int(np.flatnonzero(_FULL)[0])
    pos = int(np.flatnonzero(_ORDER == big)[0])
    pads = [flat.BoostPadState(True, 0.0)] * N_PADS
    pads[pos] = flat.BoostPadState(False, 2.5)
    st, _ = ad.to_state(packet(pads=pads))
    assert st.pad_cooldown[0].max() == pytest.approx(7.5) and st.pad_active[0].sum() == N_PADS - 1


def test_slots_par_equipe_et_slot_du_bot() -> None:
    ps = [player(1, x=1.0), player(0, x=2.0)]  # l'orange est l'indice 0 du paquet
    st, slot = PacketAdapter(field_info(), 0).to_state(packet(ps))
    assert slot == 1 and st.car_pos[0, 0, 0] == 2.0 and st.car_pos[0, 1, 0] == 1.0
    st.validate()
    with pytest.raises(ValueError, match="équilibrées"):
        PacketAdapter(field_info(), 0).to_state(packet([player(0), player(0)]))
    with pytest.raises(ValueError, match="hors"):
        PacketAdapter(field_info(), 5).to_state(packet())


def test_champs_de_vol() -> None:
    """has_jumped / double saut / dodge, démolition, air_time_since_jump (ADR-004 §2)."""
    ad = PacketAdapter(field_info(), 0)

    def st_of(**kw: object) -> tuple[float, int, int, int, int]:
        p = player(0, **kw)
        s, _ = ad.to_state(packet([p, player(1)]))
        return (
            float(s.car_air_time_since_jump[0, 0]),
            int(s.car_on_ground[0, 0]),
            int(s.car_has_flipped[0, 0]),
            int(s.car_has_double_jumped[0, 0]),
            int(s.car_is_demoed[0, 0]),
        )

    air = {"air_state": flat.AirState.InAir}
    assert st_of()[:2] == (0.0, 1)
    t, g, *_ = st_of(**air, has_jumped=True, dodge_timeout=0.75)
    assert t == pytest.approx(0.5) and g == 0  # 0,5 s après la fin du saut
    assert st_of(**air, has_jumped=True, dodge_timeout=-1.0)[0] == 1.25  # fenêtre expirée
    assert st_of(**air, has_jumped=False, dodge_timeout=-1.0)[0] == 0.0  # chute sans saut : flip reset
    assert st_of(air_state=flat.AirState.Jumping, has_jumped=True, dodge_timeout=-1.0)[:2] == (0.0, 0)
    assert st_of(**air, has_dodged=True)[2] == 1 and st_of(**air, has_double_jumped=True)[3] == 1
    assert st_of(demolished_timeout=1.5)[4] == 1 and st_of(demolished_timeout=-1.0)[4] == 0


def test_boost_borne() -> None:
    st, _ = PacketAdapter(field_info(), 0, validate=True).to_state(packet([player(0, boost=101.0), player(1)]))
    assert st.car_boost[0, 0] == 100.0


# --- tick-skip et action précédente -----------------------------------------------------------------------------


def test_decision_tous_les_8_ticks_et_action_precedente() -> None:
    ad = PacketAdapter(field_info(), 0)
    seen: list[tuple[int, int]] = []  # (indice choisi, indice précédent lu dans l'obs)
    choices = iter([10, 20, 30, 40])

    def policy(obs: F32) -> int:
        prev_row = obs[S.PREV_ACTION]
        prev = -1 if not prev_row.any() and not seen else int(np.flatnonzero((ACTION_TABLE == prev_row).all(axis=1))[0])
        a = next(choices)
        seen.append((a, prev))
        return a

    outs = [action_from_controls(ad.step(packet(frame=f), policy)) for f in range(20)]
    assert outs == [10] * 8 + [20] * 8 + [30] * 4
    assert [a for a, _ in seen] == [10, 20, 30]
    assert [p for _, p in seen] == [-1, 10, 20]


def test_paquets_manques_et_nouveau_match() -> None:
    ad = PacketAdapter(field_info(), 0)
    calls: list[int] = []

    def policy(obs: F32) -> int:
        calls.append(len(calls))
        return calls[-1]

    for f in (100, 103, 109, 110, 140, 20):  # trous : 103 tenu, 109 (>= 100+8) décide, 140 décide, 20 (recul) décide
        ad.step(packet(frame=f), policy)
    assert len(calls) == 4


def test_phase_non_jouable_remet_le_suivi_a_zero() -> None:
    ad = PacketAdapter(field_info(), 0)
    ad.step(packet(frame=0), lambda _o: 5)
    assert ad.prev_action == 5
    idle = ad.step(packet(frame=1, phase=flat.MatchPhase.Replay), lambda _o: 6)
    assert ad.prev_action == -1 and not (idle.boost or idle.jump or idle.throttle)
    first: list[F32] = []
    ad.step(packet(frame=2, phase=flat.MatchPhase.Kickoff), lambda o: first.append(o) or 7)
    assert not first[0][S.PREV_ACTION].any()  # après reset : bloc d'action précédente à zéro


def test_politique_hors_table_refusee() -> None:
    with pytest.raises(ValueError, match="hors"):
        PacketAdapter(field_info(), 0).step(packet(), lambda _o: N_ACTIONS)
    with pytest.raises(ValueError, match="pads"):
        PacketAdapter(flat.FieldInfo(boost_pads=[]), 0)
