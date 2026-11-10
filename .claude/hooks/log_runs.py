"""PostToolUse Bash|PowerShell (async) : journalise les commandes de run dans production/run_log.jsonl.

Une ligne JSON par commande de train / eval / benchmark / smoke test / export / kaggle kernels.
Ne journalise jamais l'environnement.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_DIR, read_input

RUN_RE = re.compile(
    r"scripts[/\\](train|eval|benchmark|smoke_test|export_bot|kaggle_\w+)\.py|\bkaggle\s+kernels\s+(push|output|status)\b"
)


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        dirty = subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return (out.stdout.strip() or "none") + ("-dirty" if dirty else "")
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def main() -> None:
    data = read_input()
    cmd = (data.get("tool_input") or {}).get("command") or ""
    if not RUN_RE.search(cmd):
        return
    entry = {
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "session": data.get("session_id"),
        "tool": data.get("tool_name"),
        "commit": git_head(),
        "cwd": data.get("cwd"),
        "command": cmd,
    }
    log = PROJECT_DIR / "production" / "run_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
