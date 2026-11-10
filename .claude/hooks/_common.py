"""Utilitaires partagés par les hooks Claude Code du studio (stdlib uniquement).

Convention : un refus = message sur stderr + exit 2 (bloquant pour PreToolUse/Stop).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])

# Dossiers dont le contenu ne doit jamais être modifié par un agent (écriture via Edit/Write ou suppression via shell).
PROTECTED_DIRS = ("checkpoints", "runs", "data", "repos", "threads")

# Motifs de secrets (jeton Kaggle, .env).
SECRET_PATTERNS = (
    re.compile(r"(^|[/\\])\.kaggle([/\\]|$)", re.I),
    re.compile(r"kaggle\.json", re.I),
    re.compile(r"(^|[/\\])access_token$", re.I),
    re.compile(r"(^|[/\\])\.env(\.|$)", re.I),
)


def read_input() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        deny("entrée JSON du hook illisible (fail-closed)")
    return data if isinstance(data, dict) else {}


def deny(reason: str) -> None:
    print(f"[hook RL-GCO] REFUSÉ : {reason}", file=sys.stderr)
    sys.exit(2)


def norm(path: str) -> str:
    return path.replace("\\", "/")


def rel_to_project(path: str, cwd: str | None = None) -> str | None:
    """Chemin relatif au projet (avec /), ou None s'il est hors projet."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd or PROJECT_DIR) / p
    try:
        rel = p.resolve().relative_to(PROJECT_DIR.resolve())
    except ValueError:
        return None
    return norm(str(rel))


def is_secret(path: str) -> bool:
    n = norm(path)
    return any(pat.search(n) for pat in SECRET_PATTERNS)


def protected_root(rel: str | None) -> str | None:
    if rel is None:
        return None
    first = rel.split("/", 1)[0]
    return first if first in PROTECTED_DIRS else None
