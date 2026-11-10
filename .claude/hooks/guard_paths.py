"""PreToolUse Edit|Write|NotebookEdit : refuse les écritures dans les dossiers protégés et sur les secrets."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import deny, is_secret, protected_root, read_input, rel_to_project


def main() -> None:
    data = read_input()
    tool_input = data.get("tool_input") or {}
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if not path:
        return
    if is_secret(path):
        deny(f"écriture sur un fichier secret ({path}). Le jeton Kaggle est géré par l'utilisateur seul.")
    root = protected_root(rel_to_project(path, data.get("cwd")))
    if root:
        deny(
            f"'{root}/' est protégé (checkpoints, runs, données, sources tierces). "
            "Seuls les scripts d'entraînement y écrivent ; repos/ et threads/ sont en lecture seule."
        )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # fail-closed : une garde qui plante bloque
        deny(f"erreur interne du hook ({type(exc).__name__}: {exc})")
