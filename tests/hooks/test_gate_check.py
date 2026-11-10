"""Tests de scripts/gate_check.py (stdlib) : un gate ne doit pas passer sur un gabarit vide."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_gate_check():
    spec = importlib.util.spec_from_file_location("gate_check", ROOT / "scripts" / "gate_check.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate_check"] = mod
    spec.loader.exec_module(mod)
    return mod


PLACEHOLDER_ADR = """# ADR
## Mesures
### M1
Commit : abc1234. 21 257 env-steps/s.
## Temps de calcul
_Estimation à produire à partir de M1–M3._
## Décision
"""

FILLED_ADR = """# ADR
## Mesures
Commit : abc1234. 21 257 env-steps/s à 1 processus.
## Temps de calcul
Estimé, non mesuré : 1B steps en ≈ 40 h sur T4.
## Décision
"""


class GateCheckG2(unittest.TestCase):
    def run_check(self, label: str, adr: str) -> str:
        gc = load_gate_check()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "docs" / "adr").mkdir(parents=True)
            (Path(tmp) / "docs" / "adr" / "ADR-infra.md").write_text(adr, encoding="utf-8")
            gc.ROOT = Path(tmp)
            fn = dict(gc.GATES["G2"])[label]
            status, _ = fn()
        return status

    def test_compute_time_placeholder_fails(self) -> None:
        # Régression : le titre seul suffisait à faire passer le critère.
        self.assertEqual(self.run_check("estimation du temps de calcul", PLACEHOLDER_ADR), "FAIL")

    def test_compute_time_filled_passes(self) -> None:
        self.assertEqual(self.run_check("estimation du temps de calcul", FILLED_ADR), "PASS")

    def test_measures_accept_steps_per_second(self) -> None:
        # Régression : seul le mot « SPS » était accepté, l'ADR écrit « env-steps/s ».
        self.assertEqual(self.run_check("mesures dans l'ADR", FILLED_ADR), "PASS")

    def test_measures_without_commit_fail(self) -> None:
        adr = FILLED_ADR.replace("Commit : abc1234. ", "")
        self.assertEqual(self.run_check("mesures dans l'ADR", adr), "FAIL")


if __name__ == "__main__":
    unittest.main()
