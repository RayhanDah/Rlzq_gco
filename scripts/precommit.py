"""Vérifications avant commit (appelé par .githooks/pre-commit). Stdlib uniquement.

1. Aucun fichier staged dans un dossier protégé, aucun secret, aucun artefact binaire de modèle.
2. Si des .py sont staged : ruff check, ruff format --check, mypy (sur src/), via le venv du projet.
3. Si src/ change, tests/ doit changer aussi (règle « pas de commit sans tests »).
Code de sortie non nul = commit refusé.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = ("checkpoints/", "runs/", "data/", "repos/", "threads/", "exports/")
FORBIDDEN_RE = re.compile(r"(^|/)(\.kaggle/|kaggle\.json$|access_token$|\.env($|\.))|\.(pt|pth|onnx|ckpt)$", re.I)


def venv_python() -> Path | None:
    for cand in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if cand.exists():
            return cand
    return None


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def run(cmd: list[str]) -> bool:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode == 0


def main() -> int:
    files = staged_files()
    errors: list[str] = []

    for f in files:
        if f.startswith(PROTECTED):
            errors.append(f"fichier dans un dossier protégé : {f}")
        if FORBIDDEN_RE.search(f):
            errors.append(f"secret ou artefact de modèle : {f}")

    py_files = [f for f in files if f.endswith(".py") and (ROOT / f).exists()]
    if py_files:
        py = venv_python()
        if py is None:
            errors.append(
                "des .py sont staged mais .venv est absent : créer le venv (uv sync) avant de commiter du code."
            )
        else:
            if not run([str(py), "-m", "ruff", "check", *py_files]):
                errors.append("ruff check en échec")
            if not run([str(py), "-m", "ruff", "format", "--check", *py_files]):
                errors.append("ruff format --check en échec (lancer ruff format)")
            src_py = [f for f in py_files if f.startswith("src/")]
            if src_py and not run([str(py), "-m", "mypy", *src_py]):
                errors.append("mypy en échec")

    touches_src = any(f.startswith("src/") and f.endswith(".py") for f in files)
    touches_tests = any(f.startswith("tests/") for f in files)
    if touches_src and not touches_tests:
        errors.append("src/ modifié sans aucun changement dans tests/ (règle : pas de commit sans tests).")

    if errors:
        print("\n[precommit RL-GCO] COMMIT REFUSÉ :", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"[precommit RL-GCO] OK ({len(files)} fichiers, {len(py_files)} .py vérifiés)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
