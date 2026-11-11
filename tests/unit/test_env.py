"""Tests de `src/env` sans RocketGPU : simulateur factice derrière l'interface `Simulator` (ADR-003 §5, spec §7)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from env.batch_env import BatchEnv
from env.initial_states import (
    BALL_X,
    BALL_Y,
    CAR_REST_Z,
    CAR_X,
    CAR_Y,
    CEILING_Z,
    MIN_CAR_BALL_DIST,
    MIN_CAR_CAR_DIST,
    InitialStates,
    InitialStateSampler,
    sample_generic_states,
)
from obs.state import BatchState
from rewards import RewardConfig, RewardFunction, RewardVariant

F32 = npt.NDArray[np.float32]
I64 = npt.NDArray[np.int64]
TICK_SKIP = 8
N_FAKE_ACTIONS = 4  # le sim factice n'interprète pas les indices : il les recopie dans l'état


class FakeSim:
    """Simulateur factice : la balle avance de 1 en x par step, événements scriptés par (step, arène)."""

    def __init__(self, n_envs: int, n_cars: int = 2) -> None:
        self._e, self._c = n_envs, n_cars
        self._state = BatchState.zeros(n_envs, n_cars)
        self._blue = np.zeros(n_envs, dtype=np.int64)
        self._orange = np.zeros(n_envs, dtype=np.int64)
        self._tick = np.zeros(n_envs, dtype=np.int64)
        self._touch = np.full((n_envs, n_cars), -1, dtype=np.int64)
        self.n_steps = 0
        self.goals: dict[tuple[int, int], tuple[str, int]] = {}  # (step, arène) -> (équipe qui marque, incrément)
        self.touches: dict[tuple[int, int], int] = {}  # (step, arène) -> slot de la voiture qui touche (dernier tick)
        self.touches_first_tick: dict[tuple[int, int], int] = {}  # idem, au tout premier tick du step
        self.reset_calls: list[I64] = []
        self.set_state_calls: list[I64] = []
        self.kickoff_seeds: list[Any] = []

    n_envs = property(lambda self: self._e)
    n_cars = property(lambda self: self._c)
    tick_skip = property(lambda self: TICK_SKIP)

    def state(self) -> BatchState:
        return self._state

    def step(self, actions: I64) -> None:
        assert actions.shape == (self._e * self._c,)
        tick_before = self._tick.copy()
        self._tick += TICK_SKIP  # le réel horodate les touches dans [tick_before, tick_after - 1]
        self._state.ball_pos[:, 0] += 1.0
        self._state.car_pos[:, :, 0] = actions.reshape(self._e, self._c) + 1  # écho de l'indice d'action tenu
        for (step, arena), (team, inc) in self.goals.items():
            if step == self.n_steps:
                (self._blue if team == "blue" else self._orange)[arena] += inc
        for (step, arena), slot in self.touches.items():
            if step == self.n_steps:
                self._touch[arena, slot] = self._tick[arena] - 1
        for (step, arena), slot in self.touches_first_tick.items():
            if step == self.n_steps:
                self._touch[arena, slot] = tick_before[arena]
        self.n_steps += 1

    def scores(self) -> tuple[I64, I64]:
        return self._blue.copy(), self._orange.copy()

    def env_ticks(self) -> I64:
        return self._tick.copy()

    def touch_ticks(self) -> I64:
        return self._touch.copy()

    def reset_to_kickoff(self, arena_idx: I64, kickoff_seeds: Any) -> None:
        self.reset_calls.append(arena_idx.copy())
        self.kickoff_seeds.append(kickoff_seeds.copy())
        self._state.ball_pos[arena_idx] = (0.0, 0.0, 92.75)
        self._state.car_pos[arena_idx] = 0.0

    def set_states(self, arena_idx: I64, states: InitialStates) -> None:
        self.set_state_calls.append(arena_idx.copy())
        self._state.ball_pos[arena_idx] = states.ball_pos
        self._state.car_pos[arena_idx] = states.car_pos


def fake_obs(state: BatchState, prev_actions: I64) -> F32:
    """Obs factice (D = 7) : balle (3), voiture (3), indice d'action précédente (1, -1 = aucune). Tableau neuf."""
    e, c = state.n_envs, state.n_cars
    ball = np.repeat(state.ball_pos, c, axis=0)
    car = state.car_pos.reshape(e * c, 3)
    assert prev_actions.dtype == np.int64 and prev_actions.shape == (e * c,)
    return np.concatenate([ball, car, prev_actions[:, None]], axis=1).astype(np.float32)


def make_env(
    n_envs: int = 3,
    p_kick: float = 1.0,
    seed_sampler: int = 0,
    variant: RewardVariant = RewardVariant.GCO_STD,
    no_touch_s: float = 20.0,
    max_s: float = 120.0,
) -> tuple[BatchEnv, FakeSim, InitialStateSampler]:
    sim = FakeSim(n_envs)
    sampler = InitialStateSampler(sim.n_cars, p_kick, seed_sampler)
    reward = RewardFunction(RewardConfig(variant=variant, no_touch_timeout_s=no_touch_s, max_episode_s=max_s))
    return BatchEnv(sim, reward, sampler, build_obs=fake_obs, obs_dim=7), sim, sampler


def zero_actions(env: BatchEnv, value: int = 0) -> I64:
    return np.full(env.n_agents, value, dtype=np.int64)


# ------------------------------------------------------------------------------------------------ BatchEnv


def test_terminal_et_troncature_distincts() -> None:
    """Spec §7.4 : but => terminated seul ; no-touch et durée max => truncated seul."""
    env, sim, _ = make_env(n_envs=3, no_touch_s=1.0, max_s=3.0)
    sim.goals[(2, 0)] = ("blue", 1)  # arène 0 : but au step 2
    for t in range(60):  # arène 1 touchée tous les 5 steps (jamais de no-touch) ; arène 2 jamais touchée
        if t % 5 == 0:
            sim.touches[(t, 1)] = 0
    env.reset(0)
    seen: dict[str, list[int]] = {"term0": [], "trunc1": [], "trunc2": []}
    for t in range(50):
        _, rew, term, trunc, _ = env.step(zero_actions(env))
        assert not (term & trunc).any()  # jamais les deux
        if term[0]:
            seen["term0"].append(t)
            assert not trunc[0] and rew[0] == 1.0 and rew[1] == -1.0  # agents 0 (bleu) et 1 (orange) de l'arène 0
        if trunc[4]:
            seen["trunc2"].append(t)
            assert not term[4]
        if trunc[2]:
            seen["trunc1"].append(t)
    assert seen["term0"] == [2]
    assert seen["trunc2"][0] == 14  # 15 steps = 120 ticks sans touche (>= 1 s), step d'indice 14
    assert seen["trunc1"][0] == 44  # arène 1 : touchée tous les 5 steps, durée max 3 s = 45 steps


def test_troncature_duree_max_seule() -> None:
    env, sim, _ = make_env(n_envs=1, no_touch_s=20.0, max_s=3.0)
    for t in range(60):
        sim.touches[(t, 0)] = 0  # touche à chaque step : seule la durée peut tronquer
    env.reset(0)
    for t in range(45):
        _, rew, term, trunc, _ = env.step(zero_actions(env))
        assert (not trunc.any()) if t < 44 else trunc.all()  # 3 s = 360 ticks = 45 steps
        assert not term.any() and (rew == 0).all()


def test_aucune_recompense_hors_but() -> None:
    """Spec §7.3 : sur des épisodes aléatoires, récompense non nulle seulement au step d'un but."""
    rng = np.random.default_rng(0)
    env, sim, _ = make_env(n_envs=8, no_touch_s=1.0, max_s=4.0)
    n_goals = 0
    for t in range(200):
        for a in range(8):
            if rng.random() < 0.02:
                sim.goals[(t, a)] = (("blue", "orange")[int(rng.integers(2))], int(rng.integers(1, 9)))
            if rng.random() < 0.1:
                sim.touches[(t, a)] = int(rng.integers(2))
    env.reset(1)
    n_trunc = 0
    for _ in range(200):
        _, rew, term, trunc, _ = env.step(rng.integers(0, 4, env.n_agents).astype(np.int64))
        n_goals += int(term.sum())
        n_trunc += int(trunc.sum())
        assert np.array_equal(rew != 0, term)  # non nulle <=> but, y compris aux troncatures et resets
        assert (np.abs(rew[term]) == 1.0).all()
    assert n_goals > 0 and n_trunc > 0  # le test exerce bien les deux types de fin


def test_but_multi_increments_donne_une_seule_recompense() -> None:
    """RocketGPU mesuré : le score monte à chaque tick tant que la balle est dans le filet (jusqu'à +8 par step)."""
    env, sim, _ = make_env(n_envs=1)
    sim.goals[(0, 0)] = ("orange", 8)
    env.reset(0)
    _, rew, term, _, info = env.step(zero_actions(env))
    assert rew.tolist() == [-1.0, 1.0] and term.all() and info["goal_net"].tolist() == [0]  # filet bleu : orange marque
    _, rew, term, _, _ = env.step(zero_actions(env))  # le step suivant ne rejoue pas le but (scores déjà absorbés)
    assert (rew == 0).all() and not term.any()


def test_obs_finale_est_celle_d_avant_le_reset() -> None:
    env, sim, _ = make_env(n_envs=3)
    sim.goals[(2, 1)] = ("blue", 1)
    env.reset(0)
    for _ in range(2):
        env.step(zero_actions(env, 1))
    obs, _, term, _, info = env.step(zero_actions(env, 2))
    assert term.tolist() == [False, False, True, True, False, False]
    assert info["final_agent"].tolist() == [2, 3]
    final = info["final_obs"]
    assert final.shape == (2, 7)
    assert (final[:, 0] == 3.0).all()  # balle avancée de 3 steps : état s' d'avant reset
    assert (final[:, 6] == 2.0).all()  # action précédente = indice 2 envoyé à ce step
    assert (obs[2:4, 0] == 0.0).all()  # obs retournée : état du nouvel épisode (kickoff factice)
    assert (obs[2:4, 6] == -1.0).all()  # action précédente remise à « aucune » (-1) au reset
    assert (obs[[0, 1, 4, 5], 0] == 3.0).all()  # les autres arènes continuent
    assert (obs[[0, 1, 4, 5], 6] == 2.0).all()


def test_reset_seulement_des_arenes_finies_et_compteurs_remis_a_zero() -> None:
    env, sim, _ = make_env(n_envs=3, no_touch_s=1.0, max_s=120.0)
    sim.goals[(2, 1)] = ("orange", 1)
    for t in range(60):
        sim.touches[(t, 0)] = 1  # arène 0 touchée en permanence, arène 2 jamais
    env.reset(0)
    sim.reset_calls.clear()
    trunc2: list[int] = []
    for t in range(40):
        _, _, _, trunc, info = env.step(zero_actions(env))
        if t == 2:
            assert [r.tolist() for r in sim.reset_calls] == [[1]]  # seule l'arène 1 (but) est réinitialisée
            assert info["last_toucher"].tolist() == [1, -1, -1]
        if trunc[4]:
            trunc2.append(t)
            assert sim.reset_calls[-1].tolist() == [2]
    # arène 2 : 15 steps sans touche (120 ticks), troncature au 15e step, puis le compteur repart de zéro : +15
    assert trunc2[:2] == [14, 29]
    # arène 1 : compteurs remis à 0 par le but (au step 2) : sa troncature no-touch tombe 15 steps plus tard (step 17)
    assert len([r for r in sim.reset_calls if r.tolist() == [1]]) >= 2


def test_dernier_toucheur_et_ticks_sans_touche() -> None:
    env, sim, _ = make_env(n_envs=1, no_touch_s=20.0)
    sim.touches[(1, 0)] = 1
    sim.touches[(3, 0)] = 0
    env.reset(0)
    last = []
    for _ in range(5):
        _, _, _, _, info = env.step(zero_actions(env))
        last.append(int(info["last_toucher"][0]))
    assert last == [-1, 1, 1, 0, 0]


def test_scorer_only_recoit_le_dernier_toucheur() -> None:
    env, sim, _ = make_env(n_envs=1, variant=RewardVariant.SCORER_ONLY)
    sim.touches[(0, 0)] = 0
    sim.goals[(1, 0)] = ("blue", 1)
    env.reset(0)
    env.step(zero_actions(env))
    _, rew, term, _, _ = env.step(zero_actions(env))
    assert term.all() and rew.tolist() == [1.0, -1.0]  # le buteur (slot 0, bleu) reçoit +1, l'orange -1


def test_p_kick_modifiable_en_cours_de_run() -> None:
    env, sim, _ = make_env(n_envs=4, p_kick=0.0, no_touch_s=0.125, max_s=120.0)  # 15 ticks = 2 steps sans touche
    env.reset(0)
    assert len(sim.set_state_calls) == 1 and sim.set_state_calls[0].tolist() == [0, 1, 2, 3]  # tous génériques
    env.set_p_kick(1.0)
    assert env.p_kick == 1.0
    sim.set_state_calls.clear()
    for _ in range(6):
        env.step(zero_actions(env))
    assert len(sim.reset_calls) > 1 and sim.set_state_calls == []  # que des kickoffs : aucun état générique téléporté
    with pytest.raises(ValueError):
        env.set_p_kick(1.5)


def test_validation_des_entrees() -> None:
    env, _, _ = make_env(n_envs=2)
    env.reset(0)
    with pytest.raises(ValueError):
        env.step(np.full(env.n_agents, 90, dtype=np.int64))  # hors table (90 actions)
    with pytest.raises(ValueError):
        env.step(np.zeros(env.n_agents + 1, dtype=np.int64))
    with pytest.raises(ValueError):
        env.step(np.zeros(env.n_agents, dtype=np.float32))


def _run(seed: int, sampler_seed: int, p_kick: float) -> list[Any]:
    rng = np.random.default_rng(seed)
    env, sim, _ = make_env(n_envs=6, p_kick=p_kick, seed_sampler=0, no_touch_s=0.5, max_s=2.0)
    for t in range(60):
        for a in range(6):
            if rng.random() < 0.05:
                sim.goals[(t, a)] = ("blue", 1)
    out: list[Any] = [env.reset(sampler_seed).copy()]
    act = np.random.default_rng(9)
    for _ in range(60):
        obs, rew, term, trunc, info = env.step(act.integers(0, 4, env.n_agents).astype(np.int64))
        out.append((obs.copy(), rew.copy(), term.copy(), trunc.copy(), info["final_obs"].copy()))
    out.append([s.tolist() for s in sim.kickoff_seeds])
    return out


def _same(a: list[Any], b: list[Any]) -> bool:
    if not np.array_equal(a[0], b[0]):
        return False
    for x, y in zip(a[1:-1], b[1:-1], strict=True):
        if not all(np.array_equal(u, v) for u, v in zip(x, y, strict=True)):
            return False
    return bool(a[-1] == b[-1])


def test_determinisme_a_seed_fixe() -> None:
    a, b = _run(1, 7, 0.5), _run(1, 7, 0.5)
    assert _same(a, b)
    assert not _same(a, _run(1, 8, 0.5))  # une autre graine d'états initiaux change les tirages


# ------------------------------------------------------------------------------------------------ états initiaux


def test_proportion_p_kick_respectee() -> None:
    """100 000 tirages à p = 0,2 : écart-type 0,00126, tolérance 5 sigma (0,0063)."""
    n, p = 100_000, 0.2
    s = InitialStateSampler(2, p, seed=123)
    draw = s.sample(n)
    kick_frac = 1.0 - float(draw.generic_mask.mean())
    sigma = (p * (1 - p) / n) ** 0.5
    assert abs(kick_frac - p) < 5 * sigma
    assert draw.generic is not None and draw.generic.n_arenas == int(draw.generic_mask.sum())
    assert (
        draw.kickoff_seeds.shape == (n,) and (draw.kickoff_seeds >= 1).all() and (draw.kickoff_seeds < 2**32 - 1).all()
    )
    s.set_p_kick(0.8)
    assert abs((1.0 - float(s.sample(n).generic_mask.mean())) - 0.8) < 5 * (0.8 * 0.2 / n) ** 0.5
    s.set_p_kick(1.0)
    assert s.sample(1000).generic is None
    s.set_p_kick(0.0)
    assert s.sample(1000).generic_mask.all()
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            s.set_p_kick(bad)


def _as_batch_state(g: InitialStates) -> BatchState:
    st = BatchState.zeros(g.n_arenas, g.n_cars)
    st.ball_pos[:], st.ball_vel[:], st.ball_ang_vel[:] = g.ball_pos, g.ball_vel, g.ball_ang_vel
    st.car_pos[:], st.car_vel[:], st.car_ang_vel[:], st.car_quat[:] = g.car_pos, g.car_vel, g.car_ang_vel, g.car_quat
    st.car_boost[:], st.car_on_ground[:] = g.car_boost, g.car_on_ground
    st.car_has_jumped[:], st.car_has_double_jumped[:], st.car_has_flipped[:] = (
        g.car_has_jumped,
        g.car_has_double_jumped,
        g.car_has_flipped,
    )
    st.car_air_time_since_jump[:] = g.car_air_time_since_jump
    return st


def test_etats_generiques_valides_et_dans_le_terrain() -> None:
    g = sample_generic_states(np.random.default_rng(5), 5000, 2)
    g.check_shapes(2)
    _as_batch_state(g).validate()  # finis, drapeaux 0/1, quaternions unitaires, boost dans [0, 100], air_time >= 0
    assert (np.abs(g.car_pos[..., 0]) <= CAR_X).all() and (np.abs(g.car_pos[..., 1]) <= CAR_Y).all()
    assert (g.car_pos[..., 2] >= CAR_REST_Z).all() and (g.car_pos[..., 2] < CEILING_Z).all()
    assert (np.abs(g.ball_pos[:, 0]) <= BALL_X).all() and (np.abs(g.ball_pos[:, 1]) <= BALL_Y).all()
    assert (g.ball_pos[:, 2] > 91.25).all() and (g.ball_pos[:, 2] < CEILING_Z - 91.25).all()
    assert (
        np.linalg.norm(g.car_vel, axis=-1).max() <= 2300.0 + 1.0 and np.linalg.norm(g.ball_vel, axis=-1).max() <= 6000.0
    )
    grounded = g.car_on_ground == 1
    assert (g.car_pos[..., 2][grounded] == np.float32(CAR_REST_Z)).all() and (g.car_vel[..., 2][grounded] == 0).all()
    assert grounded.any() and (~grounded).any()
    d_cc = np.linalg.norm(g.car_pos[:, 0] - g.car_pos[:, 1], axis=-1)
    d_cb = np.linalg.norm(g.car_pos - g.ball_pos[:, None, :], axis=-1).min(axis=1)
    assert d_cc.min() >= MIN_CAR_CAR_DIST - 0.5 and d_cb.min() >= MIN_CAR_BALL_DIST - 0.5  # f32 : arrondi < 0,5 uu
    near = float((d_cb < 2100.0).mean())
    assert near > 0.5  # au moins la fraction « balle près d'une voiture » (proba 0,5 + hasard)


def test_etats_generiques_deterministes() -> None:
    a = sample_generic_states(np.random.default_rng(9), 100, 2)
    b = sample_generic_states(np.random.default_rng(9), 100, 2)
    c = sample_generic_states(np.random.default_rng(10), 100, 2)
    assert all(np.array_equal(getattr(a, f), getattr(b, f)) for f in a.__slots__)
    assert not np.array_equal(a.ball_pos, c.ball_pos)


def test_duree_tirage_et_step_rapides() -> None:
    t0 = time.perf_counter()
    env, _, _ = make_env(n_envs=64, p_kick=0.5)
    env.reset(0)
    for _ in range(50):
        env.step(zero_actions(env))
    sample_generic_states(np.random.default_rng(0), 4096, 2)
    assert time.perf_counter() - t0 < 5.0


def test_dernier_toucheur_remis_a_zero_au_reset() -> None:
    env, sim, _ = make_env(n_envs=1)
    sim.touches[(0, 0)] = 1
    sim.goals[(1, 0)] = ("blue", 1)
    env.reset(0)
    env.step(zero_actions(env))
    _, _, term, _, info = env.step(zero_actions(env))
    assert term.all() and info["last_toucher"].tolist() == [1]  # valeur de l'épisode fini
    _, _, _, _, info = env.step(zero_actions(env))
    assert info["last_toucher"].tolist() == [-1] and info["episode_ticks"].tolist() == [TICK_SKIP]


def test_touche_au_premier_tick_du_step_est_detectee() -> None:
    """Le réel horodate dans [tick_before, tick_after - 1] : un stamp == tick_before compte comme touche du step."""
    env, sim, _ = make_env(n_envs=1, no_touch_s=1.0)  # 15 steps sans touche => troncature
    for t in range(40):
        sim.touches_first_tick[(t, 0)] = 1
    env.reset(0)
    for _ in range(40):
        _, _, term, trunc, info = env.step(zero_actions(env))
        assert not trunc.any() and not term.any() and info["last_toucher"].tolist() == [1]


def test_prev_actions_indices_et_table_refusee() -> None:
    from actions import ACTION_TABLE

    env, _, _ = make_env(n_envs=2)
    assert env.n_actions == 90 and env.obs_dim == 7  # fixés dès le constructeur, avant reset()
    sim = FakeSim(1)
    sampler = InitialStateSampler(2, 1.0, 0)
    reward = RewardFunction(RewardConfig())
    with pytest.raises(ValueError):
        BatchEnv(sim, reward, sampler, action_table=ACTION_TABLE[:4].copy())  # table différente
    with pytest.raises(ValueError):
        BatchEnv(sim, reward, sampler, build_obs=fake_obs)  # obs_dim requis avec un builder injecté
    default = BatchEnv(sim, reward, sampler, action_table=ACTION_TABLE.copy())
    assert default.obs_dim == 109


def test_builder_par_defaut_reset_step_et_obs_109() -> None:
    """Régression : `obs.build_obs` exige des indices int64 (-1 après reset) ; l'env doit les lui passer."""
    env = BatchEnv(FakeSim(3), RewardFunction(RewardConfig()), InitialStateSampler(2, 0.5, 0))
    obs = env.reset(0)
    assert obs.shape == (6, 109) and np.isfinite(obs).all()
    sim = env._sim
    assert isinstance(sim, FakeSim)
    sim.goals[(0, 1)] = ("blue", 1)
    obs, _, term, _, info = env.step(np.arange(6, dtype=np.int64))
    assert obs.shape == (6, 109) and term.tolist() == [False, False, True, True, False, False]
    assert info["final_obs"].shape == (2, 109) and np.isfinite(obs).all()
