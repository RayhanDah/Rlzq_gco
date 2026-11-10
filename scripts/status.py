"""Résumé de l'état du projet : phase, git, derniers runs, checkpoints. Stdlib uniquement.

Usage : python scripts/status.py [--kaggle user/slug ...]
--kaggle interroge `kaggle kernels status` (le CLI lit lui-même le jeton ; ce script ne le touche jamais).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(*args: str) -> str:
    try:
        return subprocess.run(list(args), cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"({exc})"


def section(title: str) -> None:
    print(f"\n## {title}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    state = ROOT / "production" / "state.md"
    section("Où on en est (production/state.md)")
    if state.exists():
        text = state.read_text(encoding="utf-8")
        block = text.split("## Où on en est", 1)[-1].split("\n## ", 1)[0]
        print(block.strip())
    else:
        print("state.md absent")

    section("Git")
    print(sh("git", "log", "--oneline", "-5") or "(aucun commit)")
    print(sh("git", "status", "--short") or "(propre)")

    section("Derniers runs (production/run_log.jsonl)")
    log = ROOT / "production" / "run_log.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines()[-5:]:
            e = json.loads(line)
            print(f"{e['ts']}  {e['commit']}  {e['command'][:110]}")
    else:
        print("(aucun)")

    section("Checkpoints")
    ck = ROOT / "checkpoints"
    files = sorted(ck.rglob("*"), key=lambda p: p.stat().st_mtime)[-5:] if ck.exists() else []
    print("\n".join(str(p.relative_to(ROOT)) for p in files if p.is_file()) or "(aucun)")

    if "--kaggle" in sys.argv:
        section("Kaggle")
        if shutil.which("kaggle") is None:
            print("CLI kaggle absent")
        else:
            for slug in sys.argv[sys.argv.index("--kaggle") + 1 :]:
                print(f"{slug} : {sh('kaggle', 'kernels', 'status', slug)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
