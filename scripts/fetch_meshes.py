"""Télécharge les collision meshes soccar livrés par RLGym (commit épinglé) dans vendor/collision_meshes/soccar/.

Données seulement : aucun code tiers n'est exécuté. Chaque fichier est vérifié par son hash de blob git
(sha1 de « blob <taille>\\0<contenu> »), tel que renvoyé par l'API GitHub au commit épinglé. Les meshes dérivent des
assets du jeu : jamais versionnés (vendor/ est ignoré), transmis à Kaggle par dataset privé.

Usage : python scripts/fetch_meshes.py [--dest vendor/collision_meshes]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_SHA = "2fc8d435663454b7fe45d2556777efd65cba8c2d"  # RLGym main, 2026-05-16
API = "https://api.github.com/repos/RLGym/rlgym/contents/rlgym/rocket_league/sim/collision_meshes/{mode}?ref={sha}"
EXPECTED_COUNT = {"soccar": 16}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rl-gco-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return bytes(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=ROOT / "vendor" / "collision_meshes")
    ap.add_argument("--mode", default="soccar", choices=sorted(EXPECTED_COUNT))
    args = ap.parse_args()

    listing = json.loads(_get(API.format(mode=args.mode, sha=PINNED_SHA)))
    files = [x for x in listing if x["type"] == "file" and x["name"].endswith(".cmf")]
    if len(files) != EXPECTED_COUNT[args.mode]:
        print(f"{len(files)} fichiers .cmf, {EXPECTED_COUNT[args.mode]} attendus", file=sys.stderr)
        return 1
    out = args.dest / args.mode
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = out / f["name"]
        if target.exists() and git_blob_sha(target.read_bytes()) == f["sha"]:
            continue
        data = _get(f["download_url"])
        if git_blob_sha(data) != f["sha"]:
            print(f"hash invalide : {f['name']}", file=sys.stderr)
            return 1
        target.write_bytes(data)
    total = sum(p.stat().st_size for p in out.glob("*.cmf"))
    print(f"{len(files)} meshes {args.mode} vérifiés dans {out} ({total} octets), commit {PINNED_SHA[:7]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
