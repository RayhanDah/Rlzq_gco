"""Kernel Kaggle d'entraînement en relais : squelette sans PPO.

Généré par `scripts/kaggle_run.py start`, qui remplace RUN par la description de la session (run, session, membre,
commit, nonce du verrou, empreinte du checkpoint attendu). Le kernel ne possède aucun identifiant Kaggle et ne publie
rien : il lit le checkpoint du dataset de relais monté en entrée, puis écrit ses sorties dans /kaggle/working.
`scripts/kaggle_run.py finish`, lancé depuis le PC du membre, récupère ces sorties et publie la version suivante.

Sorties (à plat dans /kaggle/working, car `kaggle kernels output` les télécharge à plat) :
- `ckpt-<run>-s<session>-<horodatage>.json` : checkpoint factice, écrit de façon atomique (temporaire puis rename) ;
- `relay-out.json` : nom, empreinte SHA-256 et compteurs du dernier checkpoint sain, écrit lui aussi de façon atomique.
Un kernel tué par le timeout laisse donc toujours un couple cohérent (checkpoint, relay-out.json), ou aucun.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

RUN: dict[str, object] = {}  # remplacé au build par kaggle_run.py
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
T0 = time.time()
KEEP_CKPTS = 2  # /kaggle/working est limité (≈ 20 Go rapportés) : on ne garde que les derniers checkpoints
SAVES = [0]  # compteur de sauvegardes : deux sauvegardes dans la même seconde ne partagent jamais un nom


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    """Écrit dans un temporaire du même dossier, force sur disque, puis rename (atomique sur un même volume)."""
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def find_input(run_id: str) -> tuple[dict[str, object], Path] | None:
    """Cherche le manifest du run dans les datasets montés. Le chemin exact du montage n'est pas vérifié
    (`/kaggle/input/<slug>/` ou une variante) : on parcourt donc tout /kaggle/input."""
    if not INPUT.exists():
        return None
    for manifest_path in sorted(INPUT.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(manifest, dict) and manifest.get("run_id") == run_id:
            return manifest, manifest_path.parent
    return None


def load_parent(expected_sha: object) -> tuple[dict[str, object], str | None]:
    """Retourne (état du checkpoint parent, empreinte). Échoue bruyamment si l'entrée ne correspond pas à l'attendu :
    mieux vaut un kernel en erreur qu'une session qui repart d'un mauvais checkpoint."""
    found = find_input(str(RUN["run_id"]))
    if expected_sha is None:
        if found is not None and found[0].get("checkpoint"):
            raise SystemExit("entrée inattendue : le dataset contient déjà un checkpoint, mais le run démarre de zéro")
        return {"steps": 0}, None
    if found is None:
        raise SystemExit(f"manifest du run {RUN['run_id']} introuvable sous {INPUT}")
    manifest, folder = found
    ckpt = folder / str(manifest.get("checkpoint"))
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint d'entrée absent : {ckpt}")
    sha = sha256_file(ckpt)
    if sha != expected_sha:
        raise SystemExit(
            f"checkpoint d'entrée inattendu : {sha} au lieu de {expected_sha} (version du dataset périmée ?)"
        )
    return json.loads(ckpt.read_text(encoding="utf-8")), sha


def save(state: dict[str, object], parent_sha: str | None) -> None:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    SAVES[0] += 1
    name = f"ckpt-{RUN['run_id']}-s{RUN['session']}-{stamp}-{SAVES[0]:05d}.json"
    body = json.dumps(state, indent=2, sort_keys=True).encode("utf-8")
    atomic_write(WORK / name, body)
    out = {
        "run_id": RUN["run_id"],
        "session": RUN["session"],
        "nonce": RUN["nonce"],
        "commit": RUN["commit"],
        "checkpoint": name,
        "sha256": hashlib.sha256(body).hexdigest(),
        "parent_sha256": parent_sha,
        "steps": state["steps"],
        "saved_at": stamp,
        "elapsed_s": round(time.time() - T0, 1),
    }
    atomic_write(WORK / "relay-out.json", json.dumps(out, indent=2).encode("utf-8"))
    pattern = f"ckpt-{RUN['run_id']}-s{RUN['session']}-*.json"
    for old in sorted(WORK.glob(pattern))[:-KEEP_CKPTS]:
        old.unlink()
    print(f"checkpoint {name} ({state['steps']} steps)", flush=True)


def main() -> int:
    if not RUN:
        print("RUN vide : ce modèle doit être rendu par scripts/kaggle_run.py", file=sys.stderr)
        return 2
    WORK.mkdir(parents=True, exist_ok=True)
    state, parent_sha = load_parent(RUN.get("expected_sha256"))
    budget_s = float(str(RUN["train_seconds"]))
    save_every_s = float(str(RUN["save_every_s"]))
    last_save = time.time()
    steps = int(str(state.get("steps", 0)))
    # Squelette : la boucle PPO viendra ici. Elle avance un compteur factice et sauvegarde périodiquement.
    while time.time() - T0 < budget_s:
        time.sleep(min(1.0, max(0.0, budget_s - (time.time() - T0))))
        steps += 1000
        if time.time() - last_save >= save_every_s:
            save(_state(steps, parent_sha), parent_sha)
            last_save = time.time()
    save(_state(steps, parent_sha), parent_sha)
    return 0


def _state(steps: int, parent_sha: str | None) -> dict[str, object]:
    return {
        "run_id": RUN["run_id"],
        "session": RUN["session"],
        "commit": RUN["commit"],
        "accelerator": RUN["accelerator"],
        "steps": steps,
        "parent_sha256": parent_sha,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dummy": True,
    }


if __name__ == "__main__":
    sys.exit(main())
