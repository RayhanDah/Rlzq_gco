"""Fixtures communes de `tests/unit` (budget : < 60 s au total, règle `.claude/rules/tests.md`).

- `os.fsync` neutralisé : la durabilité sur disque n'est pas ce que vérifient ces tests (atomicité, noms, rotation,
  reprise), et l'écriture reste réelle (`flush` + `os.replace`). L'intégration garde le vrai fsync.
  Le test d'écriture interrompue remplace lui-même `fsync` par une fonction qui lève : il n'en dépend pas.
- `git_commit` de `training.run` : un sous-processus `git rev-parse` par run (≈ 60 ms) ; la valeur réelle est calculée
  une seule fois par session puis réutilisée (HEAD ne change pas pendant la session).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import pytest

_GIT_CACHE: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _fast_io(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import os

    monkeypatch.setattr(os, "fsync", lambda fd: None)
    run_mod = sys.modules.get("training.run")
    if run_mod is not None and hasattr(run_mod, "git_commit"):
        real = run_mod.git_commit

        def cached(repo_dir: Any) -> str:
            key = str(repo_dir)
            if key not in _GIT_CACHE:
                _GIT_CACHE[key] = real(repo_dir)
            return _GIT_CACHE[key]

        monkeypatch.setattr(run_mod, "git_commit", cached)
    yield
