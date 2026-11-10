"""SessionStart : injecte production/state.md et l'état git dans le contexte de la session."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_DIR

MAX_LINES = 80


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(PROJECT_DIR), *args], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "(git indisponible)"


def main() -> None:
    sys.stdin.read()  # entrée non utilisée
    state = PROJECT_DIR / "production" / "state.md"
    print("## Contexte RL-GCO chargé par le hook SessionStart")
    if state.exists():
        lines = state.read_text(encoding="utf-8").splitlines()
        print("\n".join(lines[:MAX_LINES]))
        if len(lines) > MAX_LINES:
            print(f"... ({len(lines) - MAX_LINES} lignes tronquées, lire production/state.md)")
    else:
        print("production/state.md absent : le recréer avant toute action.")
    print("\n### Git")
    print("branche : " + git("branch", "--show-current"))
    print("dernier commit : " + (git("log", "-1", "--oneline") or "(aucun)"))
    status = git("status", "--short")
    print("modifs non commitées :\n" + (status if status else "(aucune)"))


if __name__ == "__main__":
    main()
