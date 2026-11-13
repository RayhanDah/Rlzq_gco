"""Configuration pytest commune.

Le paquet d'évaluation `src/eval` porte le même nom que le script `scripts/eval.py`. Plusieurs scripts et tests
insèrent `scripts/` en tête de `sys.path` (`scripts/bench_e2e.py`, `tests/hooks/test_hooks.py`) : un `import eval`
fait ensuite après coup résoudrait vers le script. Le paquet est donc importé ici, avant toute collecte, et reste en
cache dans `sys.modules`.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if SRC not in sys.path:  # normalement déjà fait par `pythonpath` de pyproject.toml
    sys.path.insert(0, SRC)

import eval as _eval_package  # noqa: E402

assert Path(_eval_package.__file__ or "").resolve().parent == Path(SRC, "eval").resolve(), (
    f"`import eval` ne résout pas vers src/eval : {_eval_package.__file__}"
)
