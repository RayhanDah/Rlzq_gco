"""Boucle de run (T6) avec un simulateur factice (`test_env.FakeSim`) et un Adam manuel, en CPU.

`torch.optim.Adam` n'est pas utilisé (import de `torch._dynamo`, 12 à 47 s) : l'optimiseur réel est exercé par
`tests/integration/test_train_rocketgpu.py`. Les divergences sont injectées par une sous-classe de `Learner`
substituée dans `training.run`.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import itertools
import json
import signal
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_checkpoint import _Adam  # noqa: E402
from test_env import FakeSim, fake_obs  # noqa: E402

run_mod = importlib.import_module("training.run")  # `training.run` (attribut) est la fonction réexportée
from training import (  # noqa: E402
    CheckpointIncompatibleError,
    CheckpointManager,
    Learner,
    LearnerError,
    PPOStats,
    RunConfig,
    RunError,
    make_env,
    run,
    truncate_metrics,
)

BASE: dict[str, Any] = {
    "name": "unit",
    "seed": 5,
    "device": "cpu",
    "env": {"n_envs": 4, "n_cars": 2, "tick_skip": 8},
    # épisodes courts (0,5 s = 60 ticks = 8 steps) pour que des fins d'épisode surviennent
    "reward": {"variant": "GCO-std", "no_touch_timeout_s": 0.5, "max_episode_s": 0.5},
    "initial_states": {"p_kick_start": 0.2, "p_kick_end": 0.8, "anneal_timesteps": 192},
    "model": {"policy_hidden": [16, 16], "value_hidden": [16, 16]},
    "ppo": {"epochs": 1, "minibatch_size": 16, "target_batch_size": 32, "lr_start": 1.0e-5, "lr_end": 1.0e-5},
    "collection": {"steps_per_batch": 4, "overlap": True},
    "budget": {"total_timesteps": 192},  # 32 par lot : 6 itérations
    "checkpoint": {"every_iterations": 2, "keep": 3},
    "kill": {"entropy_floor_frac": 0.001},  # obs factices non normalisées : on ne teste pas ici la règle d entropie
}


class StatefulFakeSim(FakeSim):
    """`FakeSim` restaurable bit à bit (état exact, comme RocketGPU)."""

    hook: Any = None  # appelé à chaque step avec le numéro de step (injection d'événements)

    def step(self, actions: Any) -> None:
        if self.hook is not None:
            self.hook(self.n_steps)
        super().step(actions)

    def state_dict(self) -> dict[str, Any]:
        s = self._state
        return {
            "arrays": {f.name: np.array(getattr(s, f.name)) for f in dataclasses.fields(s) if f.name != "car_team"},
            "blue": self._blue.copy(),
            "orange": self._orange.copy(),
            "tick": self._tick.copy(),
            "touch": self._touch.copy(),
            "n_steps": self.n_steps,
        }

    def load_state_dict(self, state: Any) -> None:
        for name, arr in state["arrays"].items():
            getattr(self._state, name)[...] = arr
        self._blue[...], self._orange[...] = state["blue"], state["orange"]
        self._tick[...], self._touch[...] = state["tick"], state["touch"]
        self.n_steps = int(state["n_steps"])


def make_factory(goals: dict[tuple[int, int], tuple[str, int]] | None = None, hook: Any = None) -> Any:
    def factory(cfg: RunConfig) -> Any:
        sim = StatefulFakeSim(cfg.env.n_envs, cfg.env.n_cars)
        sim.goals = dict(goals or {})
        sim.hook = hook
        return make_env(cfg, sim, build_obs=fake_obs, obs_dim=7)

    return factory


def adam_factory(params: list[Any], ppo: Any, device: Any) -> Any:
    return _Adam(params, ppo.lr_start)


def cfg_of(**over: Any) -> RunConfig:
    d = copy.deepcopy(BASE)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = {**d[k], **v}
        else:
            d[k] = v
    return RunConfig.from_dict(d)


def go(cfg: RunConfig, out: Path, **kw: Any) -> Any:
    kw.setdefault("env_factory", make_factory())
    kw.setdefault("optimizer_factory", adam_factory)
    kw.setdefault("handle_sigint", False)
    return run(cfg, out, **kw)


def rows(out: Path) -> list[dict[str, Any]]:
    p = out / "metrics.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def events(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def inject(monkeypatch: pytest.MonkeyPatch, patch: Any) -> list[int]:
    """Substitue `Learner` par une sous-classe dont `update` passe ses stats (et le numéro d'appel) à `patch`."""
    calls: list[int] = []

    class Injected(Learner):
        def update(self, batch: Any) -> PPOStats:
            stats = super().update(batch)
            calls.append(1)
            return patch(len(calls), stats)  # type: ignore[no-any-return]

    monkeypatch.setattr(run_mod, "Learner", Injected)
    return calls


# ---------------------------------------------------------------------------------------------------- boucle


def test_boucle_complete_journal_checkpoints_et_meta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    build = tmp_path / "build"
    (build / "python").mkdir(parents=True)
    (build / "build_info.json").write_text(json.dumps({"sha": "abc123"}), encoding="utf-8")
    monkeypatch.setenv("ROCKETGPU_PY_DIR", str(build / "python"))
    cfg, out = cfg_of(), tmp_path / "run"
    # Act
    s = go(cfg, out, command=["train.py", "x.yaml"])
    # Assert : budget atteint, journal sans trou, débit et normes de gradient présents
    assert s.status == "completed" and s.end_iteration == 6 == s.total_iterations
    assert s.trained_timesteps == 192 and s.nan_detected == 0 and s.restores == 0
    assert s.collected_timesteps == 7 * 32  # un lot d'avance (recouvrement)
    r = rows(out)
    assert [x["step"] for x in r] == [1, 2, 3, 4, 5, 6]
    for key in (
        "ppo/approx_kl",
        "ppo/grad_norm",
        "ppo/entropy",
        "grad/policy_last_mb_postclip",
        "grad/value_last_mb_postclip",
        "perf/agent_steps_per_s",
        "train/staleness",
        "env/goals",
        "env/p_kick",
        "train/epoch",
    ):
        assert key in r[-1], key
    assert all(x["train/staleness"] == 1.0 for x in r[1:]) and all(np.isfinite(x["ppo/entropy"]) for x in r)
    assert all(x["perf/agent_steps_per_s"] > 0 for x in r)
    # checkpoints : périodiques (2, 4) + final (6), noms jamais réutilisés
    names = [c.path.name for c in CheckpointManager(out / "checkpoints", keep=99).checkpoints()]
    assert names == sorted(set(names)) and names[-1].endswith("it00000006.pt")
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["config_hash"] == cfg.trajectory_hash() and meta["config_full_hash"] == cfg.full_hash()
    assert meta["rocketgpu_commit"] == "abc123" and meta["seed"] == 5 and meta["device"] == "cpu"
    assert meta["torch"] == torch.__version__ and meta["command"] == ["train.py", "x.yaml"]
    assert len(meta["reward_config_hash"]) == 64 and meta["git_commit"] and meta["date_utc"].startswith("20")
    state = CheckpointManager(out / "checkpoints").load_latest()
    assert state is not None
    assert state[1]["versions"]["git_commit"] == meta["git_commit"]
    assert state[1]["identity"]["reward_config_hash"] == meta["reward_config_hash"]
    assert state[1]["train_config_hash"] == cfg.trajectory_hash()
    seg = [json.loads(line) for line in (out / "segments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(seg) == 1 and seg[0]["status"] == "completed"


def test_calendrier_p_kick_applique_a_chaque_iteration(tmp_path: Path) -> None:
    cfg, out = cfg_of(), tmp_path / "run"
    go(cfg, out)
    r = rows(out)
    # l'itération numéro `step` (1-based) tourne avec p_kick_at(step - 1) ; annealing sur 6 itérations (192 / 32)
    assert [x["env/p_kick"] for x in r] == pytest.approx([cfg.p_kick_at(i) for i in range(6)])
    kicks = [x["env/p_kick"] for x in r]
    assert r[0]["env/p_kick"] == pytest.approx(0.2)
    assert all(b > a for a, b in itertools.pairwise(kicks))  # croissant strictement pendant l'annealing
    # revue T6 (A1.5) : la dernière itération de l'annealing tourne AVEC la valeur cible (avant : 0,767 pour 0,8)
    assert r[-1]["env/p_kick"] == pytest.approx(0.8)


def test_p_kick_reste_a_la_cible_quand_l_annealing_est_plus_court_que_le_run(tmp_path: Path) -> None:
    # Arrange : annealing de 2 itérations (64 timesteps) dans un run de 6
    cfg, out = cfg_of(initial_states={"p_kick_start": 0.2, "p_kick_end": 0.8, "anneal_timesteps": 64}), tmp_path / "run"
    # Act
    s = go(cfg, out)
    # Assert
    kicks = [x["env/p_kick"] for x in rows(out)]
    assert kicks == pytest.approx([0.2, 0.8, 0.8, 0.8, 0.8, 0.8]) and s.p_kick_final == pytest.approx(0.8)


def test_buts_comptes_sur_tous_les_lots_collectes(tmp_path: Path) -> None:
    # Arrange : un but bleu dans l'arène 0 au step 2 du simulateur (dans le tout premier lot)
    out = tmp_path / "run"
    # Act
    s = go(cfg_of(), out, env_factory=make_factory(goals={(2, 0): ("blue", 1)}))
    # Assert
    assert s.goals_total == 1.0
    assert rows(out)[-1]["env/goals_total"] == 1.0


def test_aucune_iteration_en_trop_et_plafond_absolu(tmp_path: Path) -> None:
    out = tmp_path / "run"
    s = go(cfg_of(), out, max_iterations=3)
    assert s.status == "paused" and s.end_iteration == 3
    ck = CheckpointManager(out / "checkpoints").checkpoints()
    assert ck[-1].iteration == 3  # checkpoint final


# ---------------------------------------------------------------------------------------------------- gardes


def test_checkpoint_initial_ecrit_et_divergence_precoce_restauree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : NaN dès le 1er appel, avant tout checkpoint périodique (tous les 2) ; avant A1.1 : arrêt
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, policy_loss=float("nan")) if n == 1 else st)
    out = tmp_path / "run"
    # Act
    s = go(cfg_of(), out)
    # Assert : un checkpoint sain à l'itération 0 existe, la restauration le reprend et le run se termine
    assert s.checkpoints[0]["iteration"] == 0 and s.checkpoints[0]["healthy"] is True
    assert s.status == "completed" and s.restores == 1 and s.nan_detected == 1 and s.end_iteration == 6
    assert [e["to_iteration"] for e in events(out) if e["event"] == "restore"] == [0]
    assert [x["step"] for x in rows(out)] == [1, 2, 3, 4, 5, 6]  # la ligne NaN (step 1) est dans metrics_dropped
    dropped = [json.loads(line) for line in (out / "metrics_dropped.jsonl").read_text(encoding="utf-8").splitlines()]
    assert dropped[0]["ppo/policy_loss"] == "nan"


def test_stop_propre_si_aucun_checkpoint_sain_n_est_restaurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : le gestionnaire ne trouve rien (checkpoints illisibles) au moment de restaurer
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, policy_loss=float("nan")) if n == 1 else st)
    monkeypatch.setattr(run_mod.CheckpointManager, "load_latest", lambda self, healthy_only=False: None)
    out = tmp_path / "run"
    # Act
    s = go(cfg_of(), out)
    # Assert : arrêt propre, cause écrite, checkpoint final non sain conservé pour diagnostic
    assert s.status == "stopped" and "aucun checkpoint sain" in (s.cause or "") and s.nan_detected == 1
    assert s.end_iteration == 1 and s.decisions[0]["action"] == "restore" and s.decisions[0]["rule"] == "nonfinite"
    assert rows(out)[0]["ppo/policy_loss"] == "nan"
    assert [c["healthy"] for c in s.checkpoints] == [True, False]  # initial (it. 0) puis final non sain
    meta = json.loads((out / "segments.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert meta["status"] == "stopped"


def test_stop_sur_nan_quand_max_restores_vaut_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, value_loss=float("inf")) if n == 4 else st)
    out = tmp_path / "run"
    s = go(cfg_of(kill={"max_restores": 0}), out)
    assert s.status == "stopped" and s.decisions[0]["rule"] == "max_restores" and s.nan_detected == 1
    assert s.end_iteration == 4 and s.restores == 0


def test_restore_sur_nan_reprend_du_dernier_sain_et_termine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : NaN au 5e appel ; checkpoint sain à l'itération 4 (périodique tous les 2)
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, grad_norm=float("nan")) if n == 5 else st)
    out = tmp_path / "run"
    # Act
    s = go(cfg_of(), out)
    # Assert
    assert s.status == "completed" and s.end_iteration == 6 and s.restores == 1 and s.nan_detected == 1
    restore_events = [e for e in events(out) if e["event"] == "restore"]
    assert len(restore_events) == 1 and restore_events[0]["to_iteration"] == 4
    assert restore_events[0]["exact_env"] is True and restore_events[0]["reseed"] != 5
    # revue T6 (A1.2) : la ligne 5 de la trajectoire écartée n'est plus dans metrics.jsonl (une seule ligne par step)
    assert [x["step"] for x in rows(out)] == [1, 2, 3, 4, 5, 6]
    assert [x["train/epoch"] for x in rows(out)] == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    dropped = [json.loads(line) for line in (out / "metrics_dropped.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [d["step"] for d in dropped] == [5] and dropped[0]["train/epoch"] == 0.0  # rien n'est supprimé
    assert all(np.isfinite(c["bytes"]) for c in s.checkpoints)


def test_restore_sur_kl_patiente_cinq_iterations_puis_reprend_effectivement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : KL à 1,0 aux appels 4 à 8 (cinq de suite) ; checkpoints à chaque itération ; budget 12 itérations
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, approx_kl=1.0) if 4 <= n <= 8 else st)
    cfg = cfg_of(budget={"total_timesteps": 12 * 32}, checkpoint={"every_iterations": 1, "keep": 2})
    out = tmp_path / "run"
    # Act
    s = go(cfg, out)
    # Assert : une seule restauration, au 5e appel de KL haute, vers le dernier sain (itération 3)
    assert s.status == "completed" and s.end_iteration == 12 and s.restores == 1
    assert [d["action"] for d in s.decisions] == ["restore"] and s.decisions[0]["rule"] == "kl"
    ev = [e for e in events(out) if e["event"] == "restore"]
    assert len(ev) == 1 and ev[0]["to_iteration"] == 3
    # le dernier checkpoint sain est conservé par la rotation (keep=2) malgré les checkpoints non sains plus récents
    metas = [c for c in s.checkpoints]
    assert any(not c["healthy"] for c in metas) and any(c["healthy"] for c in metas)
    # après la restauration, les itérations 4.. sont rejouées (lignes de journal marquées epoch 1)
    after = [x for x in rows(out) if x["train/epoch"] == 1.0]
    assert [x["step"] for x in after] == [4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert [x["step"] for x in rows(out)] == list(range(1, 13))  # aucun step en double
    dropped = [json.loads(line)["step"] for line in (out / "metrics_dropped.jsonl").read_text("utf-8").splitlines()]
    assert dropped == [4, 5, 6, 7, 8]


def test_max_restores_arrete_le_run_apres_la_derniere_restauration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : KL haute en permanence à partir du 4e appel, une seule restauration autorisée
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, approx_kl=1.0) if n >= 4 else st)
    cfg = cfg_of(
        budget={"total_timesteps": 20 * 32},
        checkpoint={"every_iterations": 1, "keep": 2},
        kill={"max_restores": 1},
    )
    out = tmp_path / "run"
    # Act
    s = go(cfg, out)
    # Assert
    assert s.status == "stopped" and s.restores == 1
    assert [d["action"] for d in s.decisions] == ["restore", "stop"] and s.decisions[1]["rule"] == "max_restores"
    assert "max_restores" in (s.cause or "")


# ---------------------------------------------------------------------------------------------------- reprise


_VOLATILE = ("t", "perf/agent_steps_per_s")


def _stable(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if k not in _VOLATILE}


def test_reprise_equivaut_a_un_run_continu(tmp_path: Path) -> None:
    # Arrange : référence continue
    cfg = cfg_of()
    a, b = tmp_path / "a", tmp_path / "b"
    go(cfg, a)
    # Act : 3 itérations, arrêt, reprise dans de NOUVEAUX objets, fin du budget
    s1 = go(cfg, b, max_iterations=3)
    s2 = go(cfg, b, resume=True)
    # Assert : reprise exacte (état du simulateur), mêmes lignes de journal, mêmes poids finaux
    assert s1.status == "paused" and s2.status == "completed" and s2.start_iteration == 3 and s2.end_iteration == 6
    assert s2.resumed_from is not None and s2.exact_env is True
    assert [_stable(x) for x in rows(a)] == [_stable(x) for x in rows(b)]
    la, lb = CheckpointManager(a / "checkpoints").load_latest(), CheckpointManager(b / "checkpoints").load_latest()
    assert la is not None and lb is not None
    sa, sb = la[1]["trainer"]["learner"], lb[1]["trainer"]["learner"]
    for net in ("policy", "value"):
        for k, v in sa[net].items():
            assert torch.equal(v, sb[net][k]), (net, k)
    assert la[1]["counters"] == lb[1]["counters"] and la[1]["extra"]["goals"] == lb[1]["extra"]["goals"]


def test_journal_sans_doublon_apres_reprise_et_sans_perte(tmp_path: Path) -> None:
    # Arrange : 4 itérations puis lignes parasites 5 et 6 (un segment qui a dépassé le checkpoint avant de planter)
    cfg, out = cfg_of(), tmp_path / "run"
    go(cfg, out, max_iterations=4)
    before = (out / "metrics.jsonl").read_bytes()
    with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"step": 5, "t": 0.0, "stale": 1.0}) + "\n")
        f.write(json.dumps({"step": 6, "t": 0.0, "stale": 1.0}) + "\n")
    # Act
    go(cfg, out, resume=True)
    # Assert
    steps = [x["step"] for x in rows(out)]
    assert steps == [1, 2, 3, 4, 5, 6]
    assert all("stale" not in x for x in rows(out))
    dropped = [json.loads(line) for line in (out / "metrics_dropped.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [d["step"] for d in dropped] == [5, 6]  # rien n'est perdu
    assert (out / "metrics.jsonl").read_bytes().startswith(before)  # les 4 premières lignes sont intactes


def test_truncate_metrics_ne_supprime_jamais_le_fichier(tmp_path: Path) -> None:
    p, d = tmp_path / "m.jsonl", tmp_path / "d.jsonl"
    p.write_text('{"step": 1}\n{"step": 2}\n', encoding="utf-8")
    assert truncate_metrics(p, 0, d) == 2
    assert p.exists() and p.read_text(encoding="utf-8") == ""
    assert truncate_metrics(tmp_path / "absent.jsonl", 0, d) == 0
    assert truncate_metrics(p, 5, d) == 0


def test_reprise_refusee_si_la_config_d_entrainement_change(tmp_path: Path) -> None:
    out = tmp_path / "run"
    go(cfg_of(), out, max_iterations=2)
    n_ckpt = len(list((out / "checkpoints").iterdir()))
    for changed in (cfg_of(seed=6), cfg_of(ppo={"epochs": 2}), cfg_of(budget={"total_timesteps": 288})):
        with pytest.raises(CheckpointIncompatibleError, match="configuration d'entraînement a changé"):
            go(changed, out, resume=True)
    assert len(list((out / "checkpoints").iterdir())) == n_ckpt  # rien n'a été écrit ni supprimé


def test_reprise_acceptee_si_seule_une_section_operationnelle_change(tmp_path: Path) -> None:
    out = tmp_path / "run"
    go(cfg_of(), out, max_iterations=2)
    s = go(cfg_of(checkpoint={"every_iterations": 3, "keep": 5}, kill={"max_restores": 7}), out, resume=True)
    assert s.status == "completed"


def test_reprise_acceptee_si_seul_le_device_change_sim_neuve_et_journalisee(tmp_path: Path) -> None:
    # Arrange : session 1 en `cpu`, session 2 en `auto` (résolu en CPU ici) : seul `device` diffère (ADR-014 D6 a)
    out = tmp_path / "run"
    go(cfg_of(), out, max_iterations=3)
    # Act
    s = go(cfg_of(device="auto"), out, resume=True)
    # Assert : budget fini, simulateur repris « modulo reset des arènes », changement de device dans le journal
    assert s.status == "completed" and s.start_iteration == 3 and s.exact_env is False
    notes = [e["note"] for e in events(out) if e.get("event") == "start" and e.get("resume")]
    assert len(notes) == 1 and "device cpu -> auto" in notes[0] and "exact_env=False" in notes[0]


def test_reprise_refusee_si_la_recompense_change(tmp_path: Path) -> None:
    out = tmp_path / "run"
    go(cfg_of(), out, max_iterations=2)
    changed = cfg_of(reward={"variant": "GCO-std", "no_touch_timeout_s": 0.25, "max_episode_s": 0.5})
    with pytest.raises(CheckpointIncompatibleError):
        go(changed, out, resume=True)


def test_dossier_deja_utilise_et_reprise_sans_checkpoint(tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(RunError, match="aucun checkpoint"):
        go(cfg_of(), out, resume=True)
    go(cfg_of(), out, max_iterations=2)
    with pytest.raises(RunError, match="déjà utilisé"):
        go(cfg_of(), out)


def test_meta_de_reprise_ecrite_a_cote_sans_ecraser(tmp_path: Path) -> None:
    out = tmp_path / "run"
    go(cfg_of(), out, max_iterations=2)
    first = (out / "run_meta.json").read_bytes()
    go(cfg_of(), out, resume=True)
    assert (out / "run_meta.json").read_bytes() == first
    resumed = json.loads((out / "run_meta_001.json").read_text(encoding="utf-8"))
    assert resumed["resume"] is True and resumed["start_iteration"] == 2


# ---------------------------------------------------------------------------------------------------- arrêt propre


def test_arret_propre_demande_checkpoint_final_et_thread_joint(tmp_path: Path) -> None:
    # Arrange : arrêt demandé quand 2 itérations sont faites
    out, calls = tmp_path / "run", []

    def should_stop() -> bool:
        calls.append(1)
        return len(calls) > 2

    # Act
    s = go(cfg_of(), out, should_stop=should_stop)
    # Assert
    assert s.status == "interrupted" and s.end_iteration == 2
    assert CheckpointManager(out / "checkpoints").checkpoints()[-1].iteration == 2
    assert all(t.name != "ppo-learner" for t in threading.enumerate())
    # et la reprise continue depuis ce checkpoint final
    s2 = go(cfg_of(), out, resume=True)
    assert s2.status == "completed" and s2.start_iteration == 2


def test_ctrl_c_reel_termine_l_iteration_puis_checkpoint_final(tmp_path: Path) -> None:
    # Arrange : SIGINT levé au 10e step du simulateur (2e itération, pendant la collecte)
    fired: list[int] = []

    def hook(step: int) -> None:
        if step == 10 and not fired:
            fired.append(step)
            signal.raise_signal(signal.SIGINT)

    before = signal.getsignal(signal.SIGINT)
    out = tmp_path / "run"
    # Act
    s = go(cfg_of(), out, env_factory=make_factory(hook=hook), handle_sigint=True)
    # Assert : l'itération en cours est allée au bout, état cohérent, gestionnaire d'origine rétabli
    assert fired == [10]
    assert s.status == "interrupted" and s.end_iteration == 2
    assert CheckpointManager(out / "checkpoints").checkpoints()[-1].iteration == 2
    assert signal.getsignal(signal.SIGINT) is before
    assert all(t.name != "ppo-learner" for t in threading.enumerate())
    assert go(cfg_of(), out, resume=True).status == "completed"


def test_synchrone_sans_recouvrement(tmp_path: Path) -> None:
    cfg = cfg_of(collection={"steps_per_batch": 4, "overlap": False})
    s = go(cfg, tmp_path / "run")
    assert s.status == "completed"
    assert all(x["train/staleness"] == 0.0 for x in rows(tmp_path / "run"))


def test_aucun_fichier_supprime_ou_ecrase_pendant_un_run_avec_rotation(tmp_path: Path) -> None:
    # Arrange : checkpoint à chaque itération, keep=2 : la rotation supprime ses propres anciens fichiers seulement
    cfg, out = cfg_of(checkpoint={"every_iterations": 1, "keep": 2}), tmp_path / "run"
    # Act
    s = go(cfg, out)
    # Assert : index strictement croissants, journaux présents
    idx = [c.index for c in CheckpointManager(out / "checkpoints").checkpoints()]
    assert idx == sorted(set(idx)) and len(idx) <= 3
    assert len(s.checkpoints) == 7  # initial (0) + périodiques 1 à 5 + final (6)
    assert {p.name for p in out.iterdir()} >= {"metrics.jsonl", "events.jsonl", "segments.jsonl", "run_meta.json"}


def test_stop_sur_effondrement_d_entropie(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : seuil de la garde = 10 % de ln(90) ; entropie injectée à 0,01 au 3e appel
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, entropy=0.01) if n == 3 else st)
    out = tmp_path / "run"
    # Act
    s = go(cfg_of(kill={"entropy_floor_frac": 0.10}), out)
    # Assert : arrêt (pas de restauration), checkpoint final non sain, run reprenable depuis le dernier sain
    assert s.status == "stopped" and s.decisions[-1]["rule"] == "entropy" and s.end_iteration == 3
    latest = CheckpointManager(out / "checkpoints").load_latest(healthy_only=True)
    assert latest is not None and latest[1]["iteration"] == 2


# ---------------------------------------------------------------------------------------------------- A1 : durcissement


def test_restauration_remet_les_compteurs_de_buts_et_ne_biaise_pas_le_debit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : NaN au 5e appel (restauration vers le checkpoint de l'itération 4). Un but au step 21 du simulateur,
    # collecté par l'itération 5 (écartée) puis rejoué après la restauration (l'état exact du simulateur est restauré).
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, grad_norm=float("nan")) if n == 5 else st)
    out = tmp_path / "run"
    # Act
    s = go(cfg_of(), out, env_factory=make_factory(goals={(21, 0): ("blue", 1)}))
    # Assert : le but n'est compté qu'une fois (avant A1.2 : 2, la trajectoire écartée restait comptée)
    assert s.restores == 1 and s.goals_total == 1.0
    assert rows(out)[-1]["env/goals_total"] == 1.0
    # débit : toutes les transitions réellement collectées (2 lots au démarrage + 6 itérations de 1 lot, dont
    # l'itération 5 rejouée) rapportées à la durée de toutes les itérations ; avant A1.2 : 224 (lots écartés soustraits)
    assert s.segment_collected_timesteps == 8 * 32 and s.discarded_timesteps == 32
    assert s.mean_steps_per_s == pytest.approx(8 * 32 / s.loop_s)
    assert s.collected_timesteps == 7 * 32  # côté trajectoire retenue : inchangé
    assert s.nan_detected == 1 and s.restores == 1  # compteurs de vie du run : jamais remis en arrière


def test_docstring_ne_pretend_pas_qu_aucun_checkpoint_n_est_supprime() -> None:
    # Revue T6 (A1.7) : la rotation supprime les anciens checkpoints ; la garantie doit le dire.
    doc = run_mod.__doc__ or ""
    assert "ROTATION" in doc and "ne supprime ni n'écrase jamais lui-même" in doc
    assert "aucun checkpoint, journal ou méta n'est supprimé" not in doc


def test_reprise_depuis_le_checkpoint_initial_equivaut_a_un_run_continu(tmp_path: Path) -> None:
    # Arrange : référence continue ; puis run « 0 itération » (seul le checkpoint initial est écrit), puis reprise
    cfg = cfg_of()
    a, b = tmp_path / "a", tmp_path / "b"
    go(cfg, a)
    s0 = go(cfg, b, max_iterations=0)
    # Act
    s1 = go(cfg, b, resume=True)
    # Assert
    assert s0.end_iteration == 0 and [c["iteration"] for c in s0.checkpoints] == [0]
    assert s1.status == "completed" and s1.start_iteration == 0 and s1.exact_env is True
    assert [_stable(x) for x in rows(a)] == [_stable(x) for x in rows(b)]
    la, lb = CheckpointManager(a / "checkpoints").load_latest(), CheckpointManager(b / "checkpoints").load_latest()
    assert la is not None and lb is not None
    for net in ("policy", "value"):
        for k, v in la[1]["trainer"]["learner"][net].items():
            assert torch.equal(v, lb[1]["trainer"]["learner"][net][k]), (net, k)


@pytest.mark.parametrize("overlap", [True, False])
def test_exception_du_learner_tracee_avant_de_remonter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overlap: bool
) -> None:
    # Arrange : le learner lève au 3e appel
    def boom(n: int, st: PPOStats) -> PPOStats:
        if n == 3:
            raise RuntimeError("boom learner")
        return st

    inject(monkeypatch, boom)
    out = tmp_path / "run"
    # Act
    with pytest.raises(LearnerError):
        go(cfg_of(collection={"steps_per_batch": 4, "overlap": overlap}), out)
    # Assert : événement et ligne de segment (type, message, itération), thread joint, aucun checkpoint après l'échec
    ev = [e for e in events(out) if e["event"] == "exception"]
    assert len(ev) == 1 and ev[0]["type"] == "LearnerError" and ev[0]["iteration"] == 3
    assert ev[0]["cause_type"] == "RuntimeError" and "boom learner" in ev[0]["cause_message"]
    seg = [json.loads(line) for line in (out / "segments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(seg) == 1 and seg[0]["status"] == "error" and "LearnerError" in seg[0]["cause"]
    assert seg[0]["error"]["iteration"] == 3 and "boom learner" in seg[0]["error"]["cause_message"]
    assert all(t.name != "ppo-learner" for t in threading.enumerate())
    its = [c.iteration for c in CheckpointManager(out / "checkpoints").checkpoints()]
    assert max(its) == 2  # initial (0) et périodique (2) seulement : aucun checkpoint « sain » de l'état cassé
    latest = CheckpointManager(out / "checkpoints").load_latest(healthy_only=True)
    assert latest is not None and latest[1]["iteration"] == 2


def test_exception_de_la_collecte_tracee_avant_de_remonter(tmp_path: Path) -> None:
    # Arrange : le simulateur lève au step 21 (collecte de l'itération 5, pendant que le learner tourne)
    def hook(step: int) -> None:
        if step == 21:
            raise RuntimeError("sim crash")

    out = tmp_path / "run"
    # Act
    with pytest.raises(RuntimeError, match="sim crash"):
        go(cfg_of(), out, env_factory=make_factory(hook=hook))
    # Assert
    ev = [e for e in events(out) if e["event"] == "exception"]
    assert len(ev) == 1 and ev[0]["type"] == "RuntimeError" and ev[0]["message"] == "sim crash"
    assert ev[0]["iteration"] == 5
    seg = [json.loads(line) for line in (out / "segments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert seg[-1]["status"] == "error" and seg[-1]["error"]["type"] == "RuntimeError"
    assert all(t.name != "ppo-learner" for t in threading.enumerate())
    assert max(c.iteration for c in CheckpointManager(out / "checkpoints").checkpoints()) == 4


def test_max_restores_compte_les_restaurations_cumulees_du_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : NaN aux appels 3, 8 et 13, séparés par des itérations saines (la série consécutive repart de 0) ;
    # deux restaurations au total autorisées. Avant A1.4 : le compteur se remettait à 0 et le run finissait.
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, grad_norm=float("nan")) if n in (3, 8, 13) else st)
    cfg = cfg_of(
        budget={"total_timesteps": 20 * 32},
        checkpoint={"every_iterations": 1, "keep": 2},
        kill={"max_restores": 2, "max_consecutive_restores": 5},
    )
    out = tmp_path / "run"
    # Act
    s = go(cfg, out)
    # Assert
    assert s.status == "stopped" and s.restores == 2
    assert [d["action"] for d in s.decisions] == ["restore", "restore", "stop"]
    assert s.decisions[-1]["rule"] == "max_restores"


def test_plafond_de_restaurations_consecutives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : NaN aux appels 3 et 4 (la rediffusion de l'itération 3 diverge encore, sans itération saine entre)
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, grad_norm=float("nan")) if n in (3, 4) else st)
    cfg = cfg_of(kill={"max_restores": 10, "max_consecutive_restores": 1})
    # Act
    s = go(cfg, tmp_path / "run")
    # Assert
    assert s.status == "stopped" and s.decisions[-1]["rule"] == "max_consecutive_restores" and s.restores == 1


def test_compteur_de_restaurations_dans_le_checkpoint_et_a_la_reprise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : une restauration (appel 3), pause à l'itération 4 (appels 4 et 5 : rediffusion), puis reprise dont la
    # 2e itération (appel 7) diverge ; max_restores 1 cumulé
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, grad_norm=float("nan")) if n in (3, 7) else st)
    cfg = cfg_of(kill={"max_restores": 1}, checkpoint={"every_iterations": 1, "keep": 3})
    out = tmp_path / "run"
    s1 = go(cfg, out, max_iterations=4)
    state = CheckpointManager(out / "checkpoints").load_latest()
    assert state is not None and state[1]["extra"]["guard"]["restores_total"] == 1 and s1.restores == 1
    # Act : la reprise repart avec restores_total = 1 ; un 2e NaN est fatal
    s2 = go(cfg, out, resume=True)
    # Assert
    assert s2.restores == 1 and s2.status == "stopped" and s2.decisions[-1]["rule"] == "max_restores"


def test_s1_defaut_jamais_atteint_garde_entropie_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, entropy=0.01) if n == 4 else st)
    s = go(cfg_of(kill={"entropy_floor_frac": 0.10}), tmp_path / "run")
    assert s.status == "stopped" and s.decisions[-1]["rule"] == "entropy" and s.s1_reached is False


def test_s1_atteint_coupe_la_garde_d_entropie(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : S1 atteint à l'itération 3 ; entropie effondrée à partir de l'appel 4 (sans S1 : arrêt)
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, entropy=0.01) if n >= 4 else st)
    seen: list[int] = []

    def s1(iteration: int) -> bool:
        seen.append(iteration)
        return iteration >= 3

    cfg = cfg_of(budget={"total_timesteps": 10 * 32}, kill={"entropy_floor_frac": 0.10})
    out = tmp_path / "run"
    # Act
    s = go(cfg, out, s1_provider=s1)
    # Assert : pas d'arrêt d'entropie après S1 ; transition journalisée une fois ; fournisseur plus appelé ensuite
    assert s.status == "completed" and s.s1_reached is True and s.s1_iteration == 3
    assert [e["iteration"] for e in events(out) if e["event"] == "s1_reached"] == [3]
    assert seen == [1, 2, 3]
    assert rows(out)[1]["train/s1_reached"] == 0.0 and rows(out)[2]["train/s1_reached"] == 1.0


def test_s1_arme_la_regle_explained_variance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : EV négative en permanence à partir de l'appel 4 (positive avant : valeur naturelle des obs factices
    # non pertinente) ; patience 2
    # (`policy_version` = numéro d'itération : le compteur d'appels est partagé entre les deux runs du test)
    inject(
        monkeypatch, lambda n, st: dataclasses.replace(st, explained_variance=-1.0 if st.policy_version >= 4 else 0.5)
    )
    cfg = cfg_of(budget={"total_timesteps": 12 * 32}, kill={"entropy_floor_frac": 0.001, "ev_patience": 2})
    # Act
    without = go(cfg, tmp_path / "a")
    with_s1 = go(cfg, tmp_path / "b", s1_provider=lambda it: it >= 3)
    # Assert : sans S1 la règle n'est jamais armée ; avec S1 (it. 3) : arrêt à la 2e itération négative (appel 5)
    assert without.status == "completed"
    assert with_s1.status == "stopped" and with_s1.decisions[-1]["rule"] == "explained_variance"
    assert with_s1.end_iteration == 5


def test_s1_monotone_sauvegarde_dans_le_checkpoint_et_restaure(tmp_path: Path) -> None:
    # Arrange : atteint à l'itération 2 puis le fournisseur dit « non » ; pause à 3 ; reprise sans fournisseur
    out = tmp_path / "run"
    s1 = go(cfg_of(), out, max_iterations=3, s1_provider=lambda it: it == 2)
    state = CheckpointManager(out / "checkpoints").load_latest()
    # Act
    s2 = go(cfg_of(), out, resume=True)
    # Assert
    assert s1.s1_reached is True and s1.s1_iteration == 2
    assert state is not None and state[1]["extra"]["s1_reached"] is True and state[1]["extra"]["s1_iteration"] == 2
    assert s2.s1_reached is True and s2.s1_iteration == 2
    assert [e["iteration"] for e in events(out) if e["event"] == "s1_reached"] == [2]  # pas de 2e transition


def test_s1_reste_atteint_apres_restauration_vers_un_checkpoint_anterieur(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : S1 atteint à l'itération 5, qui diverge (NaN au 5e appel) : restauration vers le checkpoint de
    # l'itération 4, sauvegardé AVANT S1. L'état S1 ne redevient pas faux.
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, grad_norm=float("nan")) if n == 5 else st)
    cfg = cfg_of(checkpoint={"every_iterations": 1, "keep": 3})
    # Act
    s = go(cfg, tmp_path / "run", s1_provider=lambda it: it >= 5)
    # Assert
    assert s.restores == 1 and s.s1_reached is True and s.s1_iteration == 5 and s.status == "completed"
