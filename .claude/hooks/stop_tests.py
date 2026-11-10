"""Stop : tests unitaires rapides si src/ ou tests/ ont changé (non commité). Exit 2 = Claude reprend le travail.

Ne fait rien si stop_hook_active (évite les boucles), si le venv ou tests/unit n'existent pas encore.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_DIR, read_input

TIMEOUT_S = 100


def main() -> None:
    data = read_input()
    if data.get("stop_hook_active"):
        return
    py = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = PROJECT_DIR / ".venv" / "bin" / "python"
    if not py.exists() or not (PROJECT_DIR / "tests" / "unit").is_dir():
        return
    changed = subprocess.run(
        ["git", "-C", str(PROJECT_DIR), "status", "--porcelain", "--", "src", "tests"],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if not changed:
        return
    try:
        res = subprocess.run(
            [str(py), "-m", "pytest", "-x", "-q", "tests/unit"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[stop RL-GCO] tests unitaires > {TIMEOUT_S} s : trop lents pour le Stop gate, à corriger.",
            file=sys.stderr,
        )
        sys.exit(2)
    if res.returncode not in (0, 5):  # 5 = aucun test collecté
        tail = "\n".join((res.stdout + res.stderr).splitlines()[-30:])
        print(f"[stop RL-GCO] tests unitaires en échec, corrige avant de terminer :\n{tail}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
