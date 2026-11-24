"""Checkpoints (T5) : atomicité, intégrité, chargement sûr, identité, rotation, reprise exacte (env jouet, CPU).

L'env jouet `Corridor` et l'Adam manuel viennent de `test_ppo.py` (même dossier, import à plat par pytest) :
créer un vrai
`torch.optim.Adam` importe `torch._dynamo` (≈ 12 à 47 s) ; la sérialisation d'un vrai Adam est testée en intégration
(`tests/integration/test_resume_rocketgpu.py`).
"""

from __future__ import annotations

import copy
import hashlib
import io
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_ppo import Corridor, _ManualAdam, _small_cfg  # noqa: E402

import training.checkpoint as ck  # noqa: E402
from env.batch_env import BatchEnv  # noqa: E402
from env.initial_states import InitialStates, InitialStateSampler  # noqa: E402
from models import PolicyNet, ValueNet  # noqa: E402
from obs.state import BatchState  # noqa: E402
from rewards import RewardConfig, RewardFunction  # noqa: E402
from training import (  # noqa: E402
    CheckpointCorruptError,
    CheckpointIdentity,
    CheckpointIncompatibleError,
    CheckpointManager,
    Collector,
    Learner,
    OverlapTrainer,
    capture,
    restore,
)

CPU = torch.device("cpu")
IDENT = CheckpointIdentity.current(reward_config_hash="r" * 64, obs_dim=7, tick_skip=8)


class _Adam(_ManualAdam):
    def state_dict(self) -> dict[str, Any]:
        return {"t": self.t, "m": list(self.m), "v": list(self.v), "lr": self.param_groups[0]["lr"]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.t = int(state["t"])
        for dst, src in zip(self.m + self.v, list(state["m"]) + list(state["v"]), strict=True):
            dst.copy_(src)
        self.param_groups[0]["lr"] = state["lr"]


class _ToyEnv:
    """Adaptateur `StatefulEnv` du Corridor (état exact)."""

    def __init__(self, env: Corridor) -> None:
        self.env = env

    def state_dict(self) -> dict[str, Any]:
        e = self.env
        return {
            "pos": e.pos.copy(),
            "ctx": e.ctx.copy(),
            "t": e.t.copy(),
            "rng": e.rng.bit_generator.state,
            "returns": list(e.returns),
        }

    def load_state_dict(self, state: Any) -> None:
        e = self.env
        e.pos, e.ctx, e.t = state["pos"].copy(), state["ctx"].copy(), state["t"].copy()
        e.rng.bit_generator.state = state["rng"]
        e.returns = list(state["returns"])
        return None


def _make(seed: int, overlap: bool) -> tuple[_ToyEnv, OverlapTrainer, Any]:
    torch.manual_seed(seed)
    cfg = _small_cfg(lr_start=3e-3, lr_end=3e-3)
    env = Corridor(16)
    pol, val = PolicyNet(7, n_actions=3, hidden=(32, 32)), ValueNet(7, hidden=(32, 32))
    actor = PolicyNet(7, n_actions=3, hidden=(32, 32))
    learner = Learner(pol, val, cfg, CPU, optimizer=_Adam([*pol.parameters(), *val.parameters()], 3e-3), seed=seed)
    collector = Collector(env, actor, cfg.rollout_steps(16), CPU, seed=seed)
    return _ToyEnv(env), OverlapTrainer(learner, collector, overlap=overlap), cfg


def _fingerprint(tr: OverlapTrainer, env: _ToyEnv) -> dict[str, Any]:
    ls = tr.learner.state_dict()
    pend = tr.pending_batch
    assert pend is not None
    return {
        "policy": {k: v.clone() for k, v in ls["policy"].items()},
        "value": {k: v.clone() for k, v in ls["value"].items()},
        "adam_m": [t.clone() for t in ls["optimizer"]["m"]],
        "adam_v": [t.clone() for t in ls["optimizer"]["v"]],
        "adam_t": ls["optimizer"]["t"],
        "version": ls["version"],
        "gen": ls["generator"].clone(),
        "agent_steps": tr.collector.agent_steps,
        "cgen": tr.collector.generator.get_state(),
        "obs": tr.collector._obs.copy(),
        "pending_obs": pend.obs.copy(),
        "pending_act": pend.act.copy(),
        "pending_v": pend.policy_version,
        "env": env.state_dict(),
        "actor": {k: v.clone() for k, v in tr.collector.policy.state_dict().items()},
    }


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor):
        return bool(torch.equal(a, b))
    if isinstance(a, np.ndarray):
        return bool(np.array_equal(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


@pytest.mark.parametrize("overlap", [False, True], ids=["synchrone", "recouvrement"])
def test_reprise_exacte_n_iterations_egale_k_plus_sauvegarde_plus_n_moins_k(tmp_path: Path, overlap: bool) -> None:
    # Arrange : référence, N = 5 itérations d'affilée
    n, k = 5, 2
    env_a, tr_a, cfg = _make(7, overlap)
    ref_stats = [tr_a.iterate().stats for _ in range(n)]
    ref_fp = _fingerprint(tr_a, env_a)
    # Act : k itérations, sauvegarde sur disque, objets NEUFS (autre seed, donc rien n'est hérité), reprise, n - k
    env_b, tr_b, _ = _make(7, overlap)
    got = [tr_b.iterate().stats for _ in range(k)]
    mgr = CheckpointManager(tmp_path)
    state = capture(
        tr_b, identity=IDENT, train_config=cfg, iteration=k, env_state=env_b.state_dict(), extra={"p_kick": 0.5}
    )
    pending_version = tr_b.pending_batch.policy_version  # type: ignore[union-attr]
    path = mgr.save(state)
    tr_b.close()
    env_c, tr_c, _ = _make(999, overlap)
    info = restore(mgr.load(path), tr_c, expected=IDENT, train_config=cfg, env=env_c)
    assert info.iteration == k and info.exact_env is True and info.extra == {"p_kick": 0.5}
    assert tr_c.pending_batch is not None and tr_c.pending_batch.policy_version == pending_version
    got += [tr_c.iterate().stats for _ in range(n - k)]
    # Assert
    assert got == ref_stats  # PPOStats dataclass : égalité exacte de tous les champs (flottants compris)
    fp = _fingerprint(tr_c, env_c)
    for key, val in ref_fp.items():
        assert _same(val, fp[key]), f"état divergent : {key}"


def test_recouvrement_lot_en_attente_fait_partie_de_l_etat(tmp_path: Path) -> None:
    """Sans le lot k+1 sauvegardé, la reprise diverge : preuve qu'il est nécessaire."""
    _, tr_a, cfg = _make(3, True)
    ref = [tr_a.iterate().stats for _ in range(3)]
    env_b, tr_b, _ = _make(3, True)
    tr_b.iterate()
    state = capture(tr_b, identity=IDENT, train_config=cfg, iteration=1, env_state=env_b.state_dict())
    state["trainer"]["pending"] = None  # lot en attente perdu : le trainer recollecte avec l'acteur courant
    mgr = CheckpointManager(tmp_path)
    env_c, tr_c, _ = _make(3, True)
    restore(mgr.load(mgr.save(state)), tr_c, expected=IDENT, env=env_c)
    assert [tr_c.iterate().stats for _ in range(2)] != ref[1:]


# ---------------------------------------------------------------------------------------------- fichier


_STATE_TEMPLATE: dict[str, Any] = {}


def _state(it: int, cfg: Any = None) -> dict[str, Any]:
    """État capturé après une itération (graine 0). Construit une seule fois (≈ 0,2 s : réseaux, collecte, update),
    puis copié en profondeur à chaque appel : seul `iteration` varie entre les appels, et chaque test reçoit un
    dictionnaire à lui qu'il peut muter."""
    if "state" not in _STATE_TEMPLATE:
        _, tr, c = _make(0, False)
        tr.iterate()
        _STATE_TEMPLATE["state"] = capture(tr, identity=IDENT, train_config=c if cfg is None else cfg, iteration=0)
    assert cfg is None, "le gabarit est capturé avec la config par défaut du test"
    state: dict[str, Any] = copy.deepcopy(_STATE_TEMPLATE["state"])
    state["iteration"] = state["counters"]["iteration"] = it
    return state


def test_ecriture_interrompue_laisse_l_ancien_checkpoint_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = CheckpointManager(tmp_path)
    good = mgr.save(_state(1))
    before = good.read_bytes()
    st2 = _state(2)

    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("coupure simulée")

    monkeypatch.setattr(ck.os, "replace", boom)  # écriture complète du temporaire, mais renommage impossible
    with pytest.raises(OSError):
        mgr.save(st2)
    monkeypatch.undo()
    monkeypatch.setattr(ck.os, "fsync", boom)  # exception au milieu de l'écriture
    with pytest.raises(OSError):
        mgr.save(st2)
    monkeypatch.undo()
    assert sorted(p.name for p in tmp_path.iterdir()) == [good.name]  # aucun temporaire ni fichier partiel
    assert good.read_bytes() == before
    assert mgr.load(good)["iteration"] == 1


def test_temporaire_orphelin_ignore_et_conserve(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path)
    mgr.save(_state(1))
    orphan = tmp_path / ".ckpt_000001_it00000002.pt.1.deadbeef.part"
    orphan.write_bytes(b"RLGCOCK1 tronque")
    _, st = mgr.load_latest() or (None, None)
    assert st is not None and st["iteration"] == 1
    mgr.save(_state(3))
    assert orphan.exists()  # la rotation ne touche que le format de checkpoint


def test_checkpoint_corrompu_jamais_charge_et_n_ecrase_rien(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path)
    p1, p2 = mgr.save(_state(1)), mgr.save(_state(2))
    raw = bytearray(p2.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    p2.write_bytes(bytes(raw))
    with pytest.raises(CheckpointCorruptError, match="somme de contrôle"):
        mgr.load(p2)
    p2.write_bytes(bytes(raw[: len(raw) // 2]))  # tronqué
    with pytest.raises(CheckpointCorruptError):
        mgr.load(p2)
    found = mgr.load_latest()
    assert found is not None and found[0] == p1 and found[1]["iteration"] == 1
    assert p2.exists()  # le fichier corrompu n'est ni supprimé ni modifié par le chargement


def test_chargement_refuse_un_pickle_arbitraire(tmp_path: Path) -> None:
    class Evil:
        def __reduce__(self) -> tuple[Any, ...]:
            return (print, ("pwned",))

    buf = io.BytesIO()
    torch.save({"x": Evil()}, buf)
    meta = b"{}"
    body = struct.pack(">I", len(meta)) + meta + buf.getvalue()
    f = tmp_path / "ckpt_000000_it00000000.pt"
    f.write_bytes(ck.MAGIC + hashlib.sha256(body).digest() + body)  # somme de contrôle valide, charge hostile
    with pytest.raises(CheckpointCorruptError, match="désérialisation refusée"):
        CheckpointManager.load(f)


def test_jamais_d_ecrasement_meme_iteration(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path, keep=5)
    a, b = mgr.save(_state(4)), mgr.save(_state(4))
    assert a != b and a.exists() and b.exists()


def test_rotation_garde_les_n_derniers_et_le_dernier_sain(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path, keep=2)
    mgr.save(_state(1), healthy=True)
    for it in (2, 3, 4, 5):
        mgr.save(_state(it), healthy=False)
    assert [r.iteration for r in mgr.checkpoints()] == [1, 4, 5]
    found = mgr.load_latest(healthy_only=True)
    assert found is not None and found[1]["iteration"] == 1
    mgr.save(_state(6), healthy=True)  # un nouveau sain remplace l'ancien sain dans la protection
    assert [r.iteration for r in mgr.checkpoints()] == [5, 6]


def test_rotation_ne_protege_pas_un_sain_corrompu(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path, keep=1)
    p1 = mgr.save(_state(1), healthy=True)
    p2 = mgr.save(_state(2), healthy=True)  # 1 supprimé (2 est sain et gardé)
    assert not p1.exists() and p2.exists()
    raw = bytearray(p2.read_bytes())
    raw[-1] ^= 1
    p2.write_bytes(bytes(raw))
    mgr.save(_state(3), healthy=False)  # le sain (corrompu) n'est pas « protégé » : rien d'intègre à garder
    assert mgr.load_latest(healthy_only=True) is None


def test_rotation_garde_le_dernier_sain_integre_si_un_plus_recent_est_corrompu(tmp_path: Path) -> None:
    # Revue du 2026-11-13 : un sain corrompu situé dans la fenêtre `keep` était protégé sans vérification, et le
    # dernier sain intègre (plus ancien) était supprimé : plus aucun checkpoint sain chargeable.
    mgr = CheckpointManager(tmp_path, keep=2)
    p_a = mgr.save(_state(1), healthy=True)
    p_b = mgr.save(_state(2), healthy=True)
    raw = bytearray(p_b.read_bytes())
    raw[-1] ^= 1  # corruption du corps, en-tête intact
    p_b.write_bytes(bytes(raw))

    mgr.save(_state(3), healthy=False)

    assert p_a.exists()
    found = mgr.load_latest(healthy_only=True)
    assert found is not None and found[0] == p_a


# ---------------------------------------------------------------------------------------------- identité


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        ({"action_table_sha256": "0" * 64}, "action_table_sha256"),
        ({"reward_config_hash": "z" * 64}, "reward_config_hash"),
        ({"obs_schema_version": 99}, "obs_schema_version"),
        ({"action_schema_version": 99}, "action_schema_version"),
        ({"reward_hash_schema_version": 99}, "reward_hash_schema_version"),
        ({"reward_spec_version": "9.9"}, "reward_spec_version"),
        ({"obs_dim": 8}, "obs_dim"),
        ({"tick_skip": 4}, "tick_skip"),
    ],
)
def test_refus_si_schema_table_ou_reward_differe(change: dict[str, Any], needle: str) -> None:
    import dataclasses

    state = _state(1)
    other = dataclasses.replace(IDENT, **change)
    _, tr, _ = _make(1, False)
    with pytest.raises(CheckpointIncompatibleError, match=needle):
        restore(state, tr, expected=other)


def test_refus_si_config_d_entrainement_change() -> None:
    state = _state(1)
    _, tr, cfg = _make(1, False)
    import dataclasses

    with pytest.raises(CheckpointIncompatibleError, match="train_config_hash"):
        restore(state, tr, expected=IDENT, train_config=dataclasses.replace(cfg, clip=0.3))


TRAJ = {"device": "xpu", "seed": 0, "ppo": {"clip": 0.2, "gamma": 0.995}}


class _SimEnv:
    """Env dont l'état contient un blob de simulateur ; enregistre l'état reçu. Sans blob : reprise « modulo reset des
    arènes », rend l'obs `obs` (comme `BatchEnv.load_state_dict`)."""

    def __init__(self, obs: Any) -> None:
        self.obs = obs
        self.received: Any = None

    def state_dict(self) -> dict[str, Any]:
        return {"counters": 3, "sim": {"internal": b"blob"}}

    def load_state_dict(self, state: Any) -> Any:
        self.received = state
        return None if state["sim"] is not None else self.obs


def _state_traj(traj: dict[str, Any], platform: str | None = None) -> tuple[dict[str, Any], OverlapTrainer, _SimEnv]:
    _, tr, _ = _make(1, False)
    tr.iterate()
    senv = _SimEnv(tr.collector._obs.copy())
    state = capture(tr, identity=IDENT, train_config=traj, iteration=1, env_state=senv.state_dict())
    if platform is not None:
        state["versions"]["platform"] = platform
    _, tr2, _ = _make(2, False)
    return state, tr2, senv


def test_capture_enregistre_la_plateforme() -> None:
    import sys

    state, _, _ = _state_traj(TRAJ)
    assert state["versions"]["platform"] == sys.platform


def test_reprise_meme_device_meme_plateforme_recharge_la_sim() -> None:
    # Arrange
    state, tr, senv = _state_traj(TRAJ)
    # Act
    info = restore(state, tr, expected=IDENT, train_config=TRAJ, env=senv)
    # Assert
    assert info.device_change is None and info.exact_env is True
    assert senv.received["sim"] == {"internal": b"blob"}


def test_reprise_accepte_un_changement_de_device_seul_et_repart_sur_une_sim_neuve() -> None:
    # Arrange : checkpoint du laptop (XPU), reprise sur Kaggle (CUDA), ADR-014 D6 a
    state, tr, senv = _state_traj(TRAJ)
    moved = {**TRAJ, "device": "cuda"}
    # Act
    info = restore(state, tr, expected=IDENT, train_config=moved, env=senv)
    # Assert : poids et compteurs restaurés, blob du simulateur ignoré, le reste de l'état d'env transmis
    assert info.device_change == ("xpu", "cuda")
    assert info.exact_env is False
    assert senv.received == {"counters": 3, "sim": None}
    assert tr.learner.version == state["trainer"]["learner"]["version"]
    assert ck.device_change(state, moved) == ("xpu", "cuda")


def test_changement_de_device_reamorce_le_generateur_du_collecteur() -> None:
    # Arrange : état de générateur illisible ici (produit par un autre device, format non garanti identique)
    state, tr, senv = _state_traj(TRAJ)
    state["trainer"]["collector"]["generator"] = torch.zeros(3, dtype=torch.uint8)
    moved = {**TRAJ, "device": "cuda"}
    # Act
    restore(state, tr, expected=IDENT, train_config=moved, env=senv)
    # Assert : générateur ré-amorcé de façon déterministe (graine 0, itération 1)
    ref = torch.Generator()
    ref.manual_seed(0 * 1_000_003 + 1)
    assert torch.equal(tr.collector.generator.get_state(), ref.get_state())


def test_meme_device_recharge_l_etat_du_generateur_du_collecteur() -> None:
    state, tr, senv = _state_traj(TRAJ)
    restore(state, tr, expected=IDENT, train_config=TRAJ, env=senv)
    assert torch.equal(tr.collector.generator.get_state(), state["trainer"]["collector"]["generator"])


@pytest.mark.parametrize(
    "other",
    [
        {**TRAJ, "device": "cuda", "seed": 1},
        {**TRAJ, "device": "cuda", "ppo": {"clip": 0.3, "gamma": 0.995}},
        {**TRAJ, "seed": 1},
    ],
    ids=["device+seed", "device+ppo", "seed seul"],
)
def test_refus_si_autre_chose_que_le_device_change(other: dict[str, Any]) -> None:
    state, tr, senv = _state_traj(TRAJ)
    with pytest.raises(CheckpointIncompatibleError, match="train_config_hash"):
        restore(state, tr, expected=IDENT, train_config=other, env=senv)
    assert senv.received is None  # rien n'est chargé avant le refus


def test_device_change_sans_cle_device_rend_none() -> None:
    state, _, _ = _state_traj({"seed": 0})
    assert ck.device_change(state, {"seed": 0}) is None
    assert ck.device_change(state, {"seed": 1}) is None


def test_autre_plateforme_meme_device_repart_sur_une_sim_neuve() -> None:
    # Arrange : checkpoint écrit sur une autre plateforme (ADR-014 D6 b : blob propre au build)
    state, tr, senv = _state_traj(TRAJ, platform="autre-plateforme")
    # Act
    info = restore(state, tr, expected=IDENT, train_config=TRAJ, env=senv)
    # Assert
    assert info.device_change is None and info.exact_env is False
    assert senv.received["sim"] is None


def test_checkpoint_sans_plateforme_recharge_la_sim() -> None:
    # Arrange : checkpoints d'avant ADR-014 (sans `platform`), dont la nuit 1 de long_s0
    state, tr, senv = _state_traj(TRAJ)
    del state["versions"]["platform"]
    # Act
    info = restore(state, tr, expected=IDENT, train_config=TRAJ, env=senv)
    # Assert
    assert info.exact_env is True and senv.received["sim"] is not None


def test_refus_si_etat_d_env_incoherent() -> None:
    state = _state(1)
    env, tr, _ = _make(1, False)
    with pytest.raises(CheckpointIncompatibleError, match="env"):
        restore(state, tr, expected=IDENT, env=env)


def test_etat_complet_contenu_adr003_7() -> None:
    state = _state(3)
    assert {"identity", "train_config_hash", "rng", "counters", "versions", "trainer"} <= state.keys()
    assert {"python", "numpy", "torch"} == state["rng"].keys()
    assert state["identity"]["action_table_sha256"] == IDENT.action_table_sha256
    assert state["counters"]["policy_version"] == 1 and state["trainer"]["pending"] is not None
    assert {"git_commit", "rocketgpu_commit", "torch", "numpy"} <= state["versions"].keys()


def test_hash_config_stable_et_sensible() -> None:
    assert ck.hash_config({"a": 1, "b": [1, 2]}) == ck.hash_config({"b": [1, 2], "a": 1})
    assert ck.hash_config({"a": 1}) != ck.hash_config({"a": 2})


# ---------------------------------------------------------------------------------------------- env


def test_sampler_aller_retour_donne_les_memes_tirages() -> None:
    a = InitialStateSampler(2, 0.3, 5)
    a.sample(7)
    b = InitialStateSampler(2, 0.9, 0)
    b.load_state_dict(ck.decode(ck.encode(a.state_dict())))
    assert b.p_kick == 0.3
    da, db = a.sample(20), b.sample(20)
    assert np.array_equal(da.kickoff_seeds, db.kickoff_seeds) and np.array_equal(da.generic_mask, db.generic_mask)
    assert da.generic is not None and db.generic is not None
    assert np.array_equal(da.generic.ball_pos, db.generic.ball_pos)


class _MiniSim:
    """Simulateur minimal sans `state_dict` : la reprise de BatchEnv se fait alors « modulo reset des arènes »."""

    tick_skip = 8

    def __init__(self, e: int) -> None:
        self.n_envs, self.n_cars = e, 2
        self._s = BatchState.zeros(e, 2)
        self._tick = np.zeros(e, np.int64)

    def state(self) -> BatchState:
        return self._s

    def step(self, actions: Any) -> None:
        self._tick += 8

    def scores(self) -> tuple[Any, Any]:
        z = np.zeros(self.n_envs, np.int64)
        return z, z.copy()

    def env_ticks(self) -> Any:
        return self._tick.copy()

    def touch_ticks(self) -> Any:
        return np.full((self.n_envs, 2), -1, np.int64)

    def reset_to_kickoff(self, arena_idx: Any, kickoff_seeds: Any) -> None:
        pass

    def set_states(self, arena_idx: Any, states: InitialStates) -> None:
        pass


def _obs(state: BatchState, prev: Any) -> Any:
    return np.zeros((state.n_envs * state.n_cars, 3), np.float32)


def _benv(seed: int = 0, p: float = 0.5) -> BatchEnv:
    return BatchEnv(
        _MiniSim(3), RewardFunction(RewardConfig()), InitialStateSampler(2, p, seed), build_obs=_obs, obs_dim=3
    )


def test_batch_env_reprise_modulo_reset_des_arenes() -> None:
    a = _benv(1, 0.4)
    a.reset(1)
    a.step(np.zeros(6, np.int64))
    state = ck.decode(ck.encode(a.state_dict()))
    assert state["sim"] is None  # ce simulateur ne sait pas se sauvegarder
    b = _benv(0, 1.0)
    obs = b.load_state_dict(state)
    assert obs is not None and obs.shape == (6, 3)  # l'appelant donne cette obs au collecteur
    assert b.p_kick == 0.4
    assert (b._episode_ticks == 0).all() and (b._prev_idx == -1).all()


def test_batch_env_refuse_une_autre_forme() -> None:
    a = _benv()
    state = a.state_dict()
    state["n_envs"] = 5
    with pytest.raises(ValueError, match="env sauvegardé"):
        a.load_state_dict(state)


# ------------------------------------------------------------------ écriture en flux (format inchangé, pic mémoire)


def _old_pack(state: dict[str, Any], meta: dict[str, Any]) -> bytes:
    """Référence : l'ancien chemin d'écriture (tout en mémoire), reconstitué."""
    import json

    buf = io.BytesIO()
    torch.save(ck.encode(state), buf)
    meta_b = json.dumps(meta, sort_keys=True).encode("utf-8")
    body = struct.pack(">I", len(meta_b)) + meta_b + buf.getvalue()
    return ck.MAGIC + hashlib.sha256(body).digest() + body


def test_ecriture_en_flux_octets_identiques_a_l_ancien_format(tmp_path: Path) -> None:
    st = _state(3)
    mgr = CheckpointManager(tmp_path)
    path = mgr.save(st)
    meta = ck.read_meta(path)
    assert path.read_bytes() == _old_pack(st, meta)


def test_lecture_d_un_fichier_ecrit_par_l_ancien_code(tmp_path: Path) -> None:
    st = _state(5)
    meta = {"format_version": 1, "iteration": 5, "index": 0, "healthy": True, "created_unix": 1.0}
    f = tmp_path / "ckpt_000000_it00000005.pt"
    f.write_bytes(_old_pack(st, meta))
    mgr = CheckpointManager(tmp_path)
    assert mgr.load(f)["iteration"] == 5
    assert mgr._verified(mgr.checkpoints()[0])


def test_octet_altere_refuse_au_chargement_et_par_verified(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path)
    p = mgr.save(_state(1))
    ref = mgr.checkpoints()[0]
    assert mgr._verified(ref)
    raw = bytearray(p.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    p.write_bytes(bytes(raw))
    assert not mgr._verified(ref)
    with pytest.raises(CheckpointCorruptError, match="somme de contrôle"):
        CheckpointManager.load(p)


@pytest.mark.parametrize("cut", [0, 5, 20, 41, -1])
def test_fichier_tronque_refuse(tmp_path: Path, cut: int) -> None:
    mgr = CheckpointManager(tmp_path)
    p = mgr.save(_state(1))
    raw = p.read_bytes()
    p.write_bytes(raw[:cut] if cut >= 0 else raw[:-1])
    assert not mgr._verified(mgr.checkpoints()[0])
    with pytest.raises(CheckpointCorruptError):
        CheckpointManager.load(p)


def test_magie_invalide_refusee_par_verified(tmp_path: Path) -> None:
    mgr = CheckpointManager(tmp_path)
    p = mgr.save(_state(1))
    raw = bytearray(p.read_bytes())
    raw[0] ^= 0xFF
    p.write_bytes(bytes(raw))
    assert not mgr._verified(mgr.checkpoints()[0])


def test_exception_pendant_torch_save_supprime_le_part_et_garde_l_ancien(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr = CheckpointManager(tmp_path)
    good = mgr.save(_state(1))
    before = good.read_bytes()

    def boom(obj: Any, f: Any, *a: Any, **k: Any) -> None:
        f.write(b"debut de charge")  # le .part existe déjà et contient des octets
        raise RuntimeError("torch.save a échoué")

    monkeypatch.setattr(ck.torch, "save", boom)
    with pytest.raises(RuntimeError):
        mgr.save(_state(2))
    monkeypatch.undo()
    assert sorted(p.name for p in tmp_path.iterdir()) == [good.name]
    assert good.read_bytes() == before


def test_save_ne_materialise_pas_le_fichier_en_memoire(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Régression (pic mémoire 4,8x -> 0,8x) : ni `BytesIO.getvalue`, ni `read_bytes` du fichier entier."""

    class Spy(io.BytesIO):
        def getvalue(self) -> bytes:
            raise AssertionError("save ne doit pas matérialiser le fichier (getvalue)")

    import types

    monkeypatch.setattr(ck, "io", types.SimpleNamespace(BytesIO=Spy))  # visible du seul module checkpoint

    def no_read_bytes(self: Path) -> bytes:
        raise AssertionError("la vérification ne doit pas lire le fichier entier")

    monkeypatch.setattr(Path, "read_bytes", no_read_bytes)
    mgr = CheckpointManager(tmp_path, keep=1)
    mgr.save(_state(1))
    mgr.save(_state(2))  # rotation : _verified en flux
    assert [r.iteration for r in mgr.checkpoints()] == [2]
