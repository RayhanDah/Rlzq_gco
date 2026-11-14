"""`training.run` : journal des alertes de garde (K2 à l'entrée et à la sortie du seuil, pas en continu) et exception
d'origine jamais masquée par une erreur d'écriture du journal (revue A4, M2)."""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from test_run import cfg_of, events, go, inject  # noqa: E402

from training import LearnerError, PPOStats, RunError  # noqa: E402

logging_mod = importlib.import_module("training.logging")


def test_erreur_d_ecriture_du_journal_ne_masque_pas_l_exception_d_origine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : le learner lève au 3e appel ; l'écriture de l'événement « exception » et de la ligne de segment échoue
    def boom(n: int, st: PPOStats) -> PPOStats:
        if n == 3:
            raise RuntimeError("boom learner")
        return st

    inject(monkeypatch, boom)
    real = logging_mod.append_jsonl

    def flaky(path: Any, obj: Any, *, fsync: bool = True) -> None:
        if obj.get("event") == "exception" or obj.get("status") == "error":
            raise OSError("disque plein (simulé)")
        real(path, obj, fsync=fsync)

    monkeypatch.setattr(logging_mod, "append_jsonl", flaky)
    # Act / Assert : c'est l'exception du learner qui remonte, pas l'OSError du journal
    with pytest.raises(LearnerError) as info:
        go(cfg_of(), tmp_path / "run")
    assert "boom learner" in str(info.value.__cause__)


def test_alerte_k2_journalisee_a_l_entree_et_a_la_sortie_du_seuil(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : entropie 1,0 (sous 2,25) aux appels 2 à 4, 3,0 sinon (seuil d'arrêt abaissé par la config de test)
    inject(monkeypatch, lambda n, st: dataclasses.replace(st, entropy=1.0 if 2 <= n <= 4 else 3.0))
    # Act
    s = go(cfg_of(), tmp_path / "run")
    # Assert : une ligne à l'entrée (itération 2), une à la sortie (itération 5), rien entre
    assert s.status == "completed"
    al = [e for e in events(tmp_path / "run") if e["event"] in ("alert", "alert_cleared")]
    assert [(e["event"], e["iteration"], e["rule"]) for e in al] == [
        ("alert", 2, "entropy_low"),
        ("alert_cleared", 5, "entropy_low"),
    ]


def test_reprise_refusee_si_le_calendrier_de_p_kick_a_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : run interrompu ; la formule de `p_kick_at` change de version (hash de config inchangé)
    run_mod = importlib.import_module("training.run")
    go(cfg_of(), tmp_path / "run", max_iterations=2)
    monkeypatch.setattr(run_mod, "P_KICK_SCHEDULE_VERSION", run_mod.P_KICK_SCHEDULE_VERSION + 1)
    # Act / Assert
    with pytest.raises(RunError, match="calendrier de p_kick"):
        go(cfg_of(), tmp_path / "run", resume=True)
