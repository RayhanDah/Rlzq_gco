"""Variante `dense-diag` (ADR-011, HORS SPEC) : formule, miroir, zéros, buts, troncatures, isolation.

Les 8 tests de la spec §7 des variantes in-spec restent dans `test_rewards_gco.py` (non modifiés)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import training.checkpoint as ck
from rewards import (
    GoalNet,
    RewardConfig,
    RewardFunction,
    RewardVariant,
    dense_shaping,
    gco_team_rewards,
    rules_for,
    uses_shaping,
)
from rewards.dense import BLUE_GOAL_BACK, ORANGE_GOAL_BACK

ROOT = Path(__file__).resolve().parents[2]
TEAM = np.array([0, 1], dtype=np.int64)
GOAL_Z = 321.3875


@dataclass
class S:
    """État minimal (E, 2 voitures), repère du sim : bleu attaque y > 0."""

    ball_pos: Any
    ball_vel: Any
    car_pos: Any
    car_vel: Any
    car_on_ground: Any
    car_team: Any = field(default_factory=lambda: TEAM)


def _state(ball_pos, ball_vel, car_pos, car_vel, ground=(1, 1)) -> S:  # type: ignore[no-untyped-def]
    f = lambda x: np.asarray([x], dtype=np.float32)  # noqa: E731
    return S(f(ball_pos), f(ball_vel), f(car_pos), f(car_vel), np.asarray([ground], dtype=np.uint8))


FAR = [[0.0, 0.0, 0.0], [5000.0, 0.0, 0.0]]
STILL = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_cibles_de_but_et_constantes_rlgym() -> None:
    assert ORANGE_GOAL_BACK == (0.0, 6000.0, GOAL_Z) and BLUE_GOAL_BACK == (0.0, -6000.0, GOAL_Z)


def test_terme_vitesse_vers_la_balle_calcule_a_la_main() -> None:
    # voiture 0 (bleu) à l'origine, balle à (1000,0,0), v = (1150,0,0) : 0,001 * 1150/2300 = 0,0005 ;
    # voiture 1 s'éloigne de la balle (u = -x, v = +x) : max(0, -1150) = 0
    s = _state([1000, 0, 0], [0, 0, 0], [[0, 0, 0], [2000, 0, 0]], [[1150, 0, 0], [1150, 0, 0]])
    out = dense_shaping(s)
    assert out.dtype == np.float32 and out.shape == (1, 2)
    np.testing.assert_allclose(out[0], [0.0005, 0.0], rtol=0, atol=1e-9)  # atol : float32 sur des valeurs ~5e-4


def test_terme_balle_vers_le_but_adverse_calcule_a_la_main() -> None:
    # balle à (0,0,321,3875) : u(balle->but orange) = +y exactement ; v_balle = (0,3000,0) : 0,01 * 3000/6000 = 0,005
    # pour le bleu ; pour l'orange (cible -y) le produit est négatif donc 0
    ball = [0, 0, GOAL_Z]
    out = dense_shaping(_state(ball, [0, 3000, 0], FAR, STILL))
    np.testing.assert_allclose(out[0], [0.005, 0.0], rtol=0, atol=1e-9)  # float32
    out = dense_shaping(_state(ball, [0, -3000, 0], FAR, STILL))
    np.testing.assert_allclose(out[0], [0.0, 0.005], rtol=0, atol=1e-9)


def test_terme_en_l_air_calcule_a_la_main() -> None:
    out = dense_shaping(_state([0, 0, 0], [0, 0, 0], FAR, STILL, ground=(0, 1)))
    np.testing.assert_allclose(out[0], [0.0002, 0.0], rtol=0, atol=1e-10)  # float32 sur 2e-4


def test_somme_des_trois_termes_et_borne_par_pas() -> None:
    # tous les termes au maximum : voiture 0 à 2300 vers la balle, balle à 6000 vers le but, en l'air
    s = _state([0, 0, GOAL_Z], [0, 6000, 0], [[-1000, 0, GOAL_Z], [5000, 0, 0]], [[2300, 0, 0], [0, 0, 0]], (0, 1))
    out = dense_shaping(s)
    np.testing.assert_allclose(out[0, 0], 0.001 + 0.01 + 0.0002, rtol=1e-6, atol=0)  # float32 : ~6e-8 relatif
    rng = np.random.default_rng(0)
    n = 500

    def capped(shape: tuple[int, ...], vmax: float) -> Any:  # vitesses physiques : norme <= vmax (le bornage l'exige)
        v = rng.uniform(-1, 1, shape)
        return (v * vmax / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1.0)).astype(np.float32)

    big = S(
        rng.uniform(-4000, 4000, (n, 3)).astype(np.float32),
        capped((n, 3), 6000.0),
        rng.uniform(-4000, 4000, (n, 2, 3)).astype(np.float32),
        capped((n, 2, 3), 2300.0),
        rng.integers(0, 2, (n, 2)).astype(np.uint8),
    )
    r = dense_shaping(big)
    assert (r >= 0).all() and (r <= 0.0112 + 1e-7).all()  # 1e-7 : arrondi float32 de 0,0112


def test_voiture_immobile_au_sol_et_balle_immobile_donnent_zero_exact() -> None:
    kick = _state([0, 0, 93], [0, 0, 0], [[0, -2560, 17], [0, 2560, 17]], STILL)
    assert (dense_shaping(kick) == 0).all()
    on_ball = _state([0, 0, 93], [0, 0, 0], [[0, 0, 93], [0, 0, 93]], STILL)  # voiture sur la balle : u nul, pas de NaN
    out = dense_shaping(on_ball)
    assert np.isfinite(out).all() and (out == 0).all()


def test_miroir_bleu_orange() -> None:
    # (x,y,z) -> (-x,-y,z) et échange des deux voitures : la récompense de l'agent 0 devient celle de l'agent 1
    rng = np.random.default_rng(7)
    n = 200
    bp, bv = rng.uniform(-3000, 3000, (n, 3)), rng.uniform(-5000, 5000, (n, 3))
    cp, cv = rng.uniform(-3000, 3000, (n, 2, 3)), rng.uniform(-2000, 2000, (n, 2, 3))
    g = rng.integers(0, 2, (n, 2))
    m = np.array([-1.0, -1.0, 1.0])
    a = S(
        bp.astype(np.float32), bv.astype(np.float32), cp.astype(np.float32), cv.astype(np.float32), g.astype(np.uint8)
    )
    b = S(
        (bp * m).astype(np.float32),
        (bv * m).astype(np.float32),
        (cp[:, ::-1] * m).astype(np.float32),
        (cv[:, ::-1] * m).astype(np.float32),
        g[:, ::-1].astype(np.uint8),
    )
    np.testing.assert_allclose(dense_shaping(a), dense_shaping(b)[:, ::-1], rtol=1e-5, atol=1e-9)  # float32


def _fn() -> RewardFunction:
    return RewardFunction(RewardConfig(variant=RewardVariant.DENSE_DIAG))


def test_but_donne_plus_ou_moins_un_plus_le_façonnage_du_pas() -> None:
    s = _state([1000, 0, 0], [0, 0, 0], [[0, 0, 0], [2000, 0, 0]], [[1150, 0, 0], [1150, 0, 0]])
    shaping = dense_shaping(s)
    out = _fn().dense_step([GoalNet.ORANGE], TEAM, [0], [8], s)  # le bleu marque
    np.testing.assert_allclose(out.rewards[0], [1.0 + shaping[0, 0], -1.0 + shaping[0, 1]], rtol=0, atol=1e-7)
    assert out.terminated[0] and not out.truncated[0]
    assert abs(out.rewards[0].sum()) > 0  # le zéro-somme exact de §2.1 n'est plus vrai (documenté)
    assert abs(float(out.rewards[0, 0]) - 1.0 - 0.0005) < 1e-7  # calcul à la main : 1 + 0,0005


def test_troncature_garde_le_façonnage_du_pas() -> None:
    cfg = RewardConfig(variant=RewardVariant.DENSE_DIAG)
    s = _state([1000, 0, 0], [0, 0, 0], [[0, 0, 0], [2000, 0, 0]], [[1150, 0, 0], [1150, 0, 0]])
    out = RewardFunction(cfg).dense_step([GoalNet.NONE], TEAM, [cfg.no_touch_ticks], [8], s)
    assert out.truncated[0] and not out.terminated[0]
    np.testing.assert_allclose(out.rewards[0], [0.0005, 0.0], rtol=0, atol=1e-9)  # float32
    out = RewardFunction(cfg).dense_step([GoalNet.NONE], TEAM, [0], [cfg.max_episode_ticks], s)
    assert out.truncated[0] and out.rewards[0, 0] > 0


def test_sans_but_la_recompense_est_le_facconnage_seul_et_gco_inchange() -> None:
    s = _state([1000, 0, 0], [0, 0, 0], FAR, STILL)
    assert (_fn().dense_rewards([GoalNet.NONE], TEAM, s) == dense_shaping(s)).all()
    std = RewardFunction(RewardConfig(variant=RewardVariant.GCO_STD))
    assert (std.rewards([GoalNet.NONE], TEAM) == 0).all()
    assert (std.rewards([GoalNet.ORANGE], TEAM) == gco_team_rewards([GoalNet.ORANGE], TEAM)).all()


def test_chemins_dense_et_gco_sont_separes() -> None:
    s = _state([0, 0, 0], [0, 0, 0], FAR, STILL)
    with pytest.raises(ValueError, match="dense_rewards"):
        _fn().rewards([GoalNet.NONE], TEAM)  # `rewards` (signature fermée §7.6) refuse dense-diag
    with pytest.raises(ValueError, match="dense_rewards"):
        _fn().step([GoalNet.NONE], TEAM, [0], [8])
    with pytest.raises(ValueError, match="car_team"):
        _fn().dense_rewards([GoalNet.NONE], [1, 0], s)
    for v in (RewardVariant.GCO_S, RewardVariant.GCO_STD, RewardVariant.GCO_EXT):
        with pytest.raises(ValueError, match="dense_rewards"):
            RewardFunction(RewardConfig(variant=v)).dense_rewards([GoalNet.NONE], TEAM, s)


def test_regles_hash_et_etiquette_hors_spec() -> None:
    dense = RewardConfig(variant=RewardVariant.DENSE_DIAG)
    rules = rules_for(RewardVariant.DENSE_DIAG)
    assert not rules.in_spec and rules.ablation_only and uses_shaping(RewardVariant.DENSE_DIAG)
    assert not any(uses_shaping(v) for v in RewardVariant if v is not RewardVariant.DENSE_DIAG)
    hashes = {RewardConfig(variant=v).config_hash() for v in RewardVariant}
    assert len(hashes) == len(RewardVariant) and dense.config_hash() in hashes
    from training import RunConfig
    from training.run import _spec_label

    cfg = RunConfig.from_yaml(ROOT / "configs" / "diag_dense.yaml")
    assert _spec_label(cfg) == {"hors_spec": True, "label": "HORS SPEC", "reward_variant": "dense-diag"}
    std = RunConfig.from_yaml(ROOT / "configs" / "run_4a_std.yaml")
    assert _spec_label(std) == {}


def _state_ck(variant: str, reward_hash: str) -> dict[str, Any]:
    return {
        "format_version": ck.CHECKPOINT_FORMAT_VERSION,
        "identity": {"reward_config_hash": reward_hash},
        "train_config": {"reward": {"variant": variant}},
        "train_config_hash": "x",
    }


def test_checkpoint_dense_refuse_pour_init_pool_et_export() -> None:
    dense_hash = RewardConfig(variant=RewardVariant.DENSE_DIAG).config_hash()
    std_hash = RewardConfig(variant=RewardVariant.GCO_STD).config_hash()
    dense = _state_ck("dense-diag", dense_hash)
    # initialiser un run GCO : refusé par `restore` (avant tout accès au trainer) et par `check_identity`
    expected = ck.CheckpointIdentity.current(reward_config_hash=std_hash, obs_dim=7, tick_skip=8)
    with pytest.raises(ck.CheckpointIncompatibleError, match="HORS SPEC"):
        ck.check_identity(dense, expected)
    with pytest.raises(ck.CheckpointIncompatibleError, match="HORS SPEC"):
        ck.restore(dense, None, expected=expected)  # type: ignore[arg-type]
    # pool d'adversaires d'entraînement et export : refusés ; évaluation : permise
    for purpose in ("init", "training_opponent", "export"):
        with pytest.raises(ck.CheckpointIncompatibleError, match="HORS SPEC"):
            ck.require_in_spec(dense, purpose)
    ck.require_in_spec(_state_ck("GCO-std", std_hash), "export")  # in-spec : accepté


def test_chargement_de_politique_refuse_hors_spec_sauf_pour_l_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import eval.match as match

    f = tmp_path / "dense.pt"
    f.write_bytes(b"x")
    monkeypatch.setattr(match.CheckpointManager, "load", staticmethod(lambda _p: _state_ck("dense-diag", "h")))
    for purpose in ("training_opponent", "export"):
        with pytest.raises(ck.CheckpointIncompatibleError, match="HORS SPEC"):
            match.load_checkpoint_policy(str(f), mode="argmax", seed=0, obs_dim=7, tick_skip=8, purpose=purpose)  # type: ignore[arg-type]
        with pytest.raises(ck.CheckpointIncompatibleError, match="HORS SPEC"):
            match.load_policy(str(f), mode="argmax", seed=0, obs_dim=7, tick_skip=8, purpose=purpose)  # type: ignore[arg-type]
    # purpose="eval" (défaut) : la garde HORS SPEC ne se déclenche pas (échec éventuel ailleurs)
    with pytest.raises(Exception) as exc:
        match.load_checkpoint_policy(str(f), mode="argmax", seed=0, obs_dim=7, tick_skip=8)
    assert "HORS SPEC" not in str(exc.value)


def test_batch_env_applique_le_facconnage_a_chaque_pas() -> None:
    from test_env import make_env, zero_actions

    for variant, nonzero in ((RewardVariant.DENSE_DIAG, True), (RewardVariant.GCO_STD, False)):
        env, sim, _ = make_env(n_envs=1, variant=variant)
        env.reset(0)
        _, rew, term, trunc, _ = env.step(zero_actions(env))
        assert not term.any() and not trunc.any()
        expected = dense_shaping(sim.state())[0]  # état de fin de pas (aucun reset : pas de fin)
        if nonzero:
            assert expected.min() > 0  # le sim factice laisse les voitures « en l'air » : 0,0002 par agent
            np.testing.assert_allclose(rew, expected, rtol=1e-6, atol=0)  # float32
        else:
            assert (rew == 0).all()
