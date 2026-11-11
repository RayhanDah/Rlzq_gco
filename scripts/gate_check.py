"""Vérification mécanique des gates (G1, G2, G3). Stdlib uniquement.

Usage : python scripts/gate_check.py G2
Sortie : une ligne par critère (PASS / FAIL / MANUAL) puis un verdict global.
Exit 0 si aucun FAIL. Un PASS ici ne remplace pas le GO humain : il prouve seulement que les artefacts requis existent
et que les commandes de vérification passent.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
Check = tuple[str, Callable[[], tuple[str, str]]]


def exists(rel: str) -> Callable[[], tuple[str, str]]:
    return lambda: ("PASS", rel) if (ROOT / rel).exists() else ("FAIL", f"absent : {rel}")


def contains(rel: str, pattern: str, label: str) -> Callable[[], tuple[str, str]]:
    def f() -> tuple[str, str]:
        p = ROOT / rel
        if not p.exists():
            return "FAIL", f"absent : {rel}"
        ok = re.search(pattern, p.read_text(encoding="utf-8"), re.M | re.S)
        return ("PASS", f"{rel} : {label}") if ok else ("FAIL", f"{rel} : manque {label}")

    return f


PLACEHOLDER_RE = re.compile(r"^\s*_(?:À|à) (?:faire|remplir|produire|compléter)|_Estimation à produire", re.M)


def section(rel: str, heading: str, pattern: str, label: str) -> Callable[[], tuple[str, str]]:
    """La section `## heading` existe, contient `pattern` et n'est pas un gabarit (« _À faire… », etc.)."""

    def f() -> tuple[str, str]:
        p = ROOT / rel
        if not p.exists():
            return "FAIL", f"absent : {rel}"
        m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", p.read_text(encoding="utf-8"), re.M | re.S)
        if not m:
            return "FAIL", f"{rel} : section « {heading} » absente"
        body = m.group(1)
        if PLACEHOLDER_RE.search(body):
            return "FAIL", f"{rel} : section « {heading} » encore à l'état de gabarit"
        if not re.search(pattern, body, re.S):
            return "FAIL", f"{rel} : manque {label}"
        return "PASS", f"{rel} : {label}"

    return f


def venv_cmd(args: list[str], label: str) -> Callable[[], tuple[str, str]]:
    def f() -> tuple[str, str]:
        py = ROOT / ".venv" / "Scripts" / "python.exe"
        if not py.exists():
            py = ROOT / ".venv" / "bin" / "python"
        if not py.exists():
            return "FAIL", "venv absent"
        res = subprocess.run([str(py), *args], cwd=ROOT, capture_output=True, text=True, timeout=600)
        last = (res.stdout + res.stderr).strip().splitlines()[-1:] or [""]
        return ("PASS" if res.returncode == 0 else "FAIL"), f"{label} → exit {res.returncode} ({last[0][:100]})"

    return f


def manual(label: str) -> Callable[[], tuple[str, str]]:
    return lambda: ("MANUAL", label)


GATES: dict[str, list[Check]] = {
    "G1": [
        ("inventaire (local, non versionné)", exists("docs/private/00_inventory.md")),
        ("papiers", exists("docs/01_papers.md")),
        ("spec GCO", contains("docs/GCO_SPEC.md", r"## 2\. Définition retenue", "définition")),
        ("budget de steps", contains("docs/GCO_SPEC.md", r"## 6\. Budget de steps", "budget")),
        ("plan", exists("docs/PLAN.md")),
        ("décisions G1", contains("docs/GCO_SPEC.md", r"## 10\. Décisions du Gate G1", "décisions consignées")),
    ],
    "G2": [
        ("studio Phase 1", exists(".claude/settings.json")),
        ("tests des hooks", lambda: _unittest_hooks()),
        ("sonde Kaggle", exists("docs/research/kaggle_probe.md")),
        ("ADR infra", exists("docs/adr/001-infra.md")),
        (
            "mesures dans l'ADR",
            section(
                "docs/adr/001-infra.md",
                "Mesures",
                r"(?=.*[Cc]ommit)(?=.*(?:SPS|steps/s))",
                "commit et débit (SPS ou steps/s)",
            ),
        ),
        (
            "banc soutenu ≥ 30 min",
            section(
                "docs/adr/001-infra.md",
                "Mesures",
                r"[Ss]outenu[^\n]{0,200}?\b(?:[3-9]\d|\d{3,})\s?min",
                "banc soutenu d'au moins 30 min (throttling, docs/PLAN.md Phase 2)",
            ),
        ),
        (
            "estimation du temps de calcul",
            section("docs/adr/001-infra.md", "Temps de calcul", r"\d", "estimation chiffrée"),
        ),
        ("tests unitaires", venv_cmd(["-m", "pytest", "-x", "-q", "tests/unit"], "pytest tests/unit")),
        ("GO humain", manual("l'utilisateur valide la décision d'infra")),
    ],
    "G3": [
        ("tests unitaires", venv_cmd(["-m", "pytest", "-x", "-q", "tests/unit"], "pytest tests/unit")),
        ("tests reward GCO", exists("tests/unit/test_rewards_gco.py")),
        ("test de parité obs/actions", exists("tests/integration/test_parity.py")),
        ("rapport de smoke test", exists("docs/reports/smoke_test.md")),
        ("rapport du run court (4a)", exists("docs/reports/run_4a.md")),
        ("config du run long", exists("configs/long_run.yaml")),
        ("kill criteria dans la config", contains("configs/long_run.yaml", r"kill", "section kill")),
        ("GO humain", manual("l'utilisateur valide config, critères de succès et d'arrêt")),
    ],
}


def _unittest_hooks() -> tuple[str, str]:
    res = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests/hooks"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return ("PASS" if res.returncode == 0 else "FAIL"), f"unittest tests/hooks → exit {res.returncode}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    gate = (sys.argv[1] if len(sys.argv) > 1 else "").upper()
    if gate not in GATES:
        print(f"usage : python scripts/gate_check.py {{{'|'.join(GATES)}}}")
        return 2
    results = []
    for label, fn in GATES[gate]:
        status, detail = fn()
        results.append(status)
        print(f"[{status:6}] {label} — {detail}")
    if "FAIL" in results:
        verdict = "FAIL"
    elif "MANUAL" in results:
        verdict = "PRÊT POUR REVUE HUMAINE"
    else:
        verdict = "PASS"
    print(f"\nVerdict {gate} : {verdict}")
    return 1 if verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
