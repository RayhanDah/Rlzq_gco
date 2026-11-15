"""Superviseur des runs 4a (`scripts/run_4a.py`) : reprise automatique après perte du GPU, abandon sans progrès,
arrêt sur STOP ou autre erreur, borne de durée, ligne de commande des sessions enfants. Sans simulateur."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from test_run_4a import r4

Seg = dict[str, Any] | None


class Fake:
    """Sessions scriptées : chaque entrée = (code de sortie, statut du segment écrit, checkpoint après la session)."""

    def __init__(self, script: list[tuple[int, str, int | None]], ck0: int | None = 0) -> None:
        self.script = list(script)
        self.seg: Seg = None
        self.ck = ck0
        self.calls: list[tuple[bool, float | None]] = []
        self.slept: list[float] = []

    def spawn(self, resume: bool, hours: float | None) -> int:
        self.calls.append((resume, hours))
        code, status, ck = self.script.pop(0)
        self.seg, self.ck = {"status": status}, ck
        return code

    def state(self) -> tuple[Seg, int | None]:
        return self.seg, self.ck

    def run(self, **kw: Any) -> str:
        kw.setdefault("max_hours", None)
        return str(r4.supervise_run(self.spawn, self.state, sleep=self.slept.append, log=lambda _: None, **kw))


LOST, OK, KILL, ERR = r4.EXIT_DEVICE_LOST, r4.EXIT_OK, r4.EXIT_KILL, r4.EXIT_ERROR


def test_perte_du_gpu_reprise_jusqu_a_la_fin() -> None:
    f = Fake([(LOST, "error", 250), (LOST, "error", 500), (OK, "completed", 19376)])
    assert f.run() == "completed"
    assert [c[0] for c in f.calls] == [False, True, True]  # 1re session normale, puis reprises après panne
    assert f.slept == [r4.RETRY_WAIT_S, r4.RETRY_WAIT_S]


def test_abandon_apres_trois_pannes_sans_nouveau_checkpoint() -> None:
    f = Fake([(LOST, "error", 250)] * 5, ck0=250)
    assert f.run() == "error" and len(f.calls) == r4.MAX_NO_PROGRESS


def test_un_progres_remet_le_compteur_a_zero() -> None:
    script = [(LOST, "error", 250), (LOST, "error", 250), (LOST, "error", 500), (LOST, "error", 500)]
    f = Fake([*script, (LOST, "error", 500), (OK, "completed", 900)], ck0=250)
    # 250 (sans progrès : 1), 250 (2), 500 (progrès : 0), 500 (1), 500 (2), fin
    assert f.run() == "completed" and len(f.calls) == 6


def test_stop_et_autre_erreur_ne_sont_pas_repris() -> None:
    assert Fake([(KILL, "stopped", 250)]).run() == "stopped"
    f = Fake([(ERR, "error", 250)])
    assert f.run() == "error" and len(f.calls) == 1 and f.slept == []


def test_reprise_desactivee() -> None:
    f = Fake([(LOST, "error", 250)])
    assert f.run(auto_resume=False) == "error" and len(f.calls) == 1


def test_session_finie_sans_atteindre_le_budget() -> None:
    assert Fake([(OK, "interrupted", 250)]).run() == "paused"


def test_premiere_session_en_reprise_si_le_run_etait_deja_tombe() -> None:
    f = Fake([(OK, "completed", 900)])
    f.run(resume_first=True)
    assert f.calls[0][0] is True


def test_borne_de_duree_par_run() -> None:
    now = [0.0]
    f = Fake([(LOST, "error", 250), (OK, "completed", 900)])

    def spawn(resume: bool, hours: float | None) -> int:
        now[0] += 3600.0  # chaque session dure 1 h
        return f.spawn(resume, hours)

    out = r4.supervise_run(
        spawn, f.state, max_hours=3.0, sleep=lambda s: None, clock=lambda: now[0], log=lambda _: None
    )
    assert out == "completed" and [h for _, h in f.calls] == [3.0, 2.0]
    g = Fake([(LOST, "error", 250)] * 3 + [(LOST, "error", 500)])
    now[0] = 0.0

    def spawn_g(resume: bool, hours: float | None) -> int:
        now[0] += 3600.0
        return g.spawn(resume, hours)

    assert r4.supervise_run(spawn_g, g.state, max_hours=1.5, sleep=lambda s: None, clock=lambda: now[0]) == "paused"


def test_detection_de_la_perte_du_gpu() -> None:
    msg = "level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)"
    assert r4.is_device_lost(RuntimeError(msg))
    wrapped = RuntimeError("échec du thread learner")
    wrapped.__cause__ = RuntimeError(msg)  # LearnerError enveloppe l'exception du thread learner
    assert r4.is_device_lost(wrapped)
    assert not r4.is_device_lost(RuntimeError("CUDA out of memory")) and not r4.is_device_lost(None)
    assert r4.segment_device_lost({"status": "error", "error": {"message": msg}})
    assert not r4.segment_device_lost({"status": "error", "error": {"message": "autre"}})
    assert not r4.segment_device_lost({"status": "stopped", "cause": msg}) and not r4.segment_device_lost(None)


def test_iteration_du_dernier_checkpoint(tmp_path: Path) -> None:
    assert r4.latest_checkpoint_iteration(tmp_path / "absent") is None
    for name in ("ckpt_000029_it00007250.pt", "ckpt_000030_it00007500.pt", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert r4.latest_checkpoint_iteration(tmp_path) == 7500


def test_ligne_de_commande_enfant() -> None:
    args = argparse.Namespace(
        variant="std",
        runs_root=Path("R"),
        checkpoints_root=Path("C"),
        device=None,
        max_iterations=None,
        allow_dirty=False,
        no_keep_awake=False,
        std_s1_observed=False,
    )
    cmd = r4.child_command(args, 1, True, None)
    assert cmd[2:7] == ["--child", "--variant", "std", "--seed", "1"]
    assert cmd[cmd.index("--max-hours") + 1] == "0.0" and cmd[-1] == "--resume-after-stop"
    assert "--resume-after-stop" not in r4.child_command(args, 1, False, 2.5)
