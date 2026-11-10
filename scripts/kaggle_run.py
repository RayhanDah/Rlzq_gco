"""Lanceur de relais Kaggle : un même run PPO avance par sessions chaînées, chacune lancée par un membre du groupe
depuis son PC, avec son propre compte et son propre jeton (installé par lui, jamais lu par ce script).

Protocole complet, mécanismes non vérifiés et risques : `docs/research/kaggle_relay.md`.

Sous-commandes (toutes à blanc par défaut : elles construisent les dossiers sous build/kaggle/relay/ et affichent les
commandes `kaggle` sans les lancer ; `--push` les exécute réellement) :
  init     construit le dataset de relais initial (manifest vide, verrou libre) ; `--push` le crée (privé).
  start    vérifie git, prend le verrou dans le dataset, rend le kernel et le pousse.
  status   affiche le verrou et le manifest du dataset, et le statut du kernel si le verrou est le nôtre.
  finish   récupère les sorties du kernel, vérifie le checkpoint et publie une nouvelle version du dataset
           (checkpoint, manifest, verrou libéré).
  release  libère notre verrou sans publier de checkpoint (session abandonnée).

Exemple :
  python scripts/kaggle_run.py start --user alice --run gco-a --acc t4 --dataset bob/rlgco-relay-gco-a
  python scripts/kaggle_run.py start --user alice --run gco-a --acc t4 --dataset bob/rlgco-relay-gco-a --push

Stdlib uniquement. Le module `kaggle` n'est jamais importé : on appelle le CLI `kaggle` 2.2.5 en sous-processus, via
la seule classe `KaggleCli`, dont l'exécuteur est injectable (faux exécuteur dans les tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "kaggle" / "train" / "train_kernel.py"
BUILD = ROOT / "build" / "kaggle" / "relay"

LEASE_FILE = "lease.json"
MANIFEST_FILE = "manifest.json"
OUT_FILE = "relay-out.json"
DATASET_META = "dataset-metadata.json"  # nom attendu par le CLI (kaggle_api_extended.py:1143)
KERNEL_META = "kernel-metadata.json"  # kaggle_api_extended.py:1147
KERNEL_CODE = "train_kernel.py"

USER_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,48}")
RUN_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,23}")
DATASET_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,48}/[a-z0-9][a-z0-9-]{5,49}")  # slug 6–50 car. (:6136-6137)
CKPT_RE = re.compile(r"ckpt-[a-z0-9-]+-s\d+-\d{8}T\d{6}Z-\d{5}\.json")  # nom écrit par kaggle/train/train_kernel.py

# Motifs qui ne doivent jamais apparaître dans un kernel rendu, des métadonnées, un verrou ou une commande affichée.
# Les noms de variables d'environnement sont sensibles à la casse (sinon « kaggle_run.py » serait un faux positif).
SECRET_RE = re.compile(r"kaggle\.json|access_token|(?-i:KAGGLE_[A-Z])|\.kaggle[/\\]|KGAT_[0-9a-f]", re.I)


@dataclass(frozen=True)
class Accelerator:
    machine_shape: str  # IDs listés dans la doc du CLI (docs/kernels.md, « as of Sep 2026 »)
    session_s: int  # durée max d'une session : 12 h GPU, 9 h TPU (rapporté, docs/research/kaggle.md §2)
    queue_s: int  # attente avant démarrage, couverte par le verrou (TPU : ≥ 35 h rapportées, non mesuré)


ACCELERATORS = {
    "t4": Accelerator("NvidiaTeslaT4", 12 * 3600, 0),
    "tpu": Accelerator("TpuV5E8", 9 * 3600, 40 * 3600),
}
LEASE_MARGIN_S = 2 * 3600  # marge après le timeout : démarrage, récupération des sorties, publication
SAVE_MARGIN_S = 20 * 60  # le kernel finit sa boucle 20 min avant le timeout (règle .claude/rules/training.md)
SAVE_EVERY_S = 20 * 60
DEFAULT_TRAIN_S = 120  # squelette : ne brûle pas le quota ; à remplacer par timeout − marge pour un vrai run
VERIFY_TRIES = 5
VERIFY_DELAY_S = 30.0


class RelayError(RuntimeError):
    """Refus explicite (précondition, verrou, sortie invalide). Message destiné au membre."""


class KaggleError(RuntimeError):
    """Une commande `kaggle` a échoué."""


# ---------------------------------------------------------------------------------------------------------------
# Secrets


def assert_no_secret(text: str, where: str) -> None:
    if SECRET_RE.search(text):
        raise RelayError(f"motif de jeton détecté dans {where} : abandon (rien n'est écrit ni lancé)")


# ---------------------------------------------------------------------------------------------------------------
# Enveloppe du CLI kaggle 2.2.5. Options vérifiées dans .venv/Lib/site-packages/kaggle/cli.py (lignes citées).


@dataclass(frozen=True)
class CmdResult:
    returncode: int
    stdout: str


Executor = Callable[[list[str]], CmdResult]


def redact(text: str) -> str:
    """Masque toute ligne qui ressemble à un jeton avant affichage (sortie brute du CLI)."""
    return "\n".join("[ligne masquée]" if SECRET_RE.search(line) else line for line in text.splitlines())


def subprocess_executor(cmd: list[str]) -> CmdResult:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    if out.strip():
        print(redact(out.rstrip()), flush=True)
    return CmdResult(proc.returncode, out)


# Le CLI 2.2.5 ne sort en code non nul que sur exception (cli.py:106-121) ; plusieurs refus serveur sont seulement
# affichés, avec le code 0 : kaggle_api_extended.py:7300 et :7335 (push), :5955 et :5964 (version), :6237 (création).
SOFT_ERROR_RE = re.compile(r"Kernel push error|Dataset version creation error|Dataset creation error")


def kaggle_exe() -> str:
    for cand in (ROOT / ".venv" / "Scripts" / "kaggle.exe", ROOT / ".venv" / "bin" / "kaggle"):
        if cand.exists():
            return str(cand)
    return "kaggle"


class KaggleCli:
    """Seul point d'appel du CLI. `executor=None` : mode à blanc, les commandes sont affichées et journalisées dans
    `self.log`, aucun sous-processus n'est lancé."""

    def __init__(self, executor: Executor | None, exe: str = "kaggle", echo: Callable[[str], None] = print) -> None:
        self.executor = executor
        self.exe = exe
        self.echo = echo
        self.log: list[list[str]] = []

    @property
    def dry(self) -> bool:
        return self.executor is None

    def _run(self, *args: str) -> str | None:
        cmd = [self.exe, *args]
        shown = shlex.join(cmd)
        assert_no_secret(shown, "une commande kaggle")
        self.log.append(cmd)
        self.echo(("(à blanc) " if self.dry else "+ ") + shown)
        if self.executor is None:
            return None
        res = self.executor(cmd)
        if res.returncode != 0:
            raise KaggleError(f"échec ({res.returncode}) : {shown}")
        soft = SOFT_ERROR_RE.search(res.stdout) if args[1] in ("push", "version", "create") else None
        if soft:
            raise KaggleError(f"refus du serveur malgré le code 0 ({soft.group(0)}) : {shown}")
        return res.stdout

    def kernels_push(self, folder: Path, machine_shape: str, timeout_s: int) -> str | None:
        # cli.py:1414-1430 : -p/--path, -t/--timeout (secondes), --accelerator
        return self._run("kernels", "push", "-p", str(folder), "--accelerator", machine_shape, "-t", str(timeout_s))

    def kernels_status(self, kernel: str) -> str | None:
        # cli.py:1485-1495 : argument positionnel <owner/slug>
        return self._run("kernels", "status", kernel)

    def kernels_output(self, kernel: str, dest: Path) -> str | None:
        # cli.py:1454-1483 : -p/--path, -o/--force ; la pagination est suivie par l'API (kaggle_api_extended.py:7575)
        return self._run("kernels", "output", kernel, "-p", str(dest), "-o")

    def datasets_download(self, dataset: str, dest: Path, file_name: str | None = None) -> str | None:
        # cli.py:1140-1165 : -f/--file, -p/--path, --unzip, -o/--force
        args = ["datasets", "download", dataset, "-p", str(dest), "--unzip", "-o"]
        if file_name is not None:
            args += ["-f", file_name]
        return self._run(*args)

    def datasets_version(self, folder: Path, message: str) -> str | None:
        # cli.py:1198-1233 : -p/--path, -m/--message (requis) ; sous-dossiers ignorés par défaut (-r skip)
        return self._run("datasets", "version", "-p", str(folder), "-m", message)

    def datasets_create(self, folder: Path) -> str | None:
        # cli.py:1168-1195 : -p/--path ; privé par défaut (public seulement avec -u/--public)
        return self._run("datasets", "create", "-p", str(folder))


KERNEL_STATUS_RE = re.compile(r'has status "([^"]+)"')  # format : kaggle_api_extended.py:7674-7678


def parse_kernel_status(stdout: str) -> str:
    """`queued`, `running`, `complete`, `error`, `cancel_requested`, `cancel_acknowledged`, `new_script` ou `unknown`.
    Le CLI affiche `response.status`, un enum KernelWorkerStatus (kagglesdk/kernels/types/kernels_enums.py:31) dont la
    représentation texte exacte n'est pas vérifiée : on accepte `KernelWorkerStatus.COMPLETE` comme `complete`."""
    m = KERNEL_STATUS_RE.search(stdout)
    if not m:
        return "unknown"
    raw = m.group(1).split(".")[-1].strip()
    raw = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", raw).lower().replace("-", "_")
    known = {"queued", "running", "complete", "error", "cancel_requested", "cancel_acknowledged", "new_script"}
    return raw if raw in known else "unknown"


# ---------------------------------------------------------------------------------------------------------------
# Git


GitRunner = Callable[[list[str]], str]


def real_git(args: list[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def check_git(git: GitRunner, fetch: bool) -> str:
    """Arbre propre (fichiers suivis) et HEAD == origin/main. Retourne le commit complet. Les fichiers non suivis sont
    tolérés : ils ne sont jamais embarqués dans le kernel."""
    if git(["status", "--porcelain", "--untracked-files=no"]):
        raise RelayError("arbre git modifié : commiter ou annuler les changements (traçabilité du commit)")
    if fetch:
        git(["fetch", "--quiet", "origin", "main"])
    head = git(["rev-parse", "HEAD"])
    main = git(["rev-parse", "origin/main"])
    if head != main:
        raise RelayError(f"HEAD ({head[:10]}) diffère de origin/main ({main[:10]}) : faire `git pull` sur main")
    return head


# ---------------------------------------------------------------------------------------------------------------
# Verrou et manifest


def iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def lease_state(lease: dict[str, Any] | None, user: str, now: float) -> str:
    """`free` (absent ou libéré), `expired`, `mine` (actif, à nous) ou `other` (actif, à quelqu'un d'autre)."""
    if not lease or lease.get("released") or not lease.get("user"):
        return "free"
    if now >= float(lease["expires_at_unix"]):
        return "expired"
    return "mine" if lease["user"] == user else "other"


def new_lease(
    run_id: str, user: str, kernel: str, commit: str, acc_key: str, timeout_s: int, now: float, nonce: str
) -> dict[str, Any]:
    acc = ACCELERATORS[acc_key]
    expires = now + acc.queue_s + timeout_s + LEASE_MARGIN_S
    return {
        "schema": 1,
        "run_id": run_id,
        "user": user,
        "kernel": kernel,
        "commit": commit,
        "accelerator": acc.machine_shape,
        "nonce": nonce,
        "acquired_at": iso(now),
        "expires_at": iso(expires),
        "expires_at_unix": int(expires),
        "released": False,
    }


def released(lease: dict[str, Any], by: str, now: float, reason: str) -> dict[str, Any]:
    out = dict(lease)
    out.update({"released": True, "released_by": by, "released_at": iso(now), "release_reason": reason})
    return out


def empty_manifest(run_id: str) -> dict[str, Any]:
    return {"schema": 1, "run_id": run_id, "checkpoint": None, "sha256": None, "steps": 0, "sessions": []}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RelayError(f"{path.name} n'est pas un objet JSON")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    assert_no_secret(text, path.name)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def dataset_folder(
    dest: Path, dataset: str, files: dict[str, Path], manifest: dict[str, Any], lease: dict[str, Any]
) -> Path:
    """Dossier d'une nouvelle version, instantané complet : on suppose qu'une version remplace tout le jeu de
    fichiers (non vérifié), donc on y remet toujours checkpoint, manifest et verrou."""
    fresh_dir(dest)
    write_json(dest / DATASET_META, {"id": dataset})  # `id` suffit pour `datasets version` (:5864-5868)
    for name, src in files.items():
        shutil.copyfile(src, dest / name)
    write_json(dest / MANIFEST_FILE, manifest)
    write_json(dest / LEASE_FILE, lease)
    return dest


# ---------------------------------------------------------------------------------------------------------------
# Kernel


def kernel_slug(run_id: str, acc_key: str) -> str:
    return f"rlgco-train-{run_id}-{acc_key}"


def render_kernel(run: dict[str, Any]) -> str:
    src = TEMPLATE.read_text(encoding="utf-8")
    marker = "RUN: dict[str, object] = {}"
    if marker not in src:
        raise RelayError("marqueur RUN introuvable dans le modèle de kernel")
    # json.loads d'un littéral chaîne : `null`/`true` du JSON ne sont pas du Python valide (régression vue en test).
    out = src.replace(marker, f"RUN: dict[str, object] = json.loads({json.dumps(run, ensure_ascii=False)!r})", 1)
    assert_no_secret(out, "le kernel rendu")
    return out


def kernel_metadata(user: str, run_id: str, acc_key: str, dataset: str) -> dict[str, Any]:
    slug = kernel_slug(run_id, acc_key)
    return {
        "id": f"{user}/{slug}",
        "title": slug,  # le slug doit égaler slugify(title) (kaggle_api_extended.py:7197-7204)
        "code_file": KERNEL_CODE,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if acc_key == "t4" else "false",
        "enable_tpu": "true" if acc_key == "tpu" else "false",  # lu en 2.2.5 (kaggle_api_extended.py:7267)
        "enable_internet": "true",  # pip install des dépendances (futur) ; le kernel n'a aucun identifiant
        "machine_shape": ACCELERATORS[acc_key].machine_shape,
        "dataset_sources": [dataset],  # monté en entrée (:7224-7226) ; droits d'un collaborateur non vérifiés
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def build_kernel(dest: Path, run: dict[str, Any], meta: dict[str, Any]) -> Path:
    fresh_dir(dest)
    (dest / KERNEL_CODE).write_text(render_kernel(run), encoding="utf-8")
    write_json(dest / KERNEL_META, meta)
    return dest


# ---------------------------------------------------------------------------------------------------------------
# Contexte et sous-commandes


@dataclass
class Ctx:
    user: str
    run_id: str
    acc_key: str
    dataset: str
    cli: KaggleCli
    git: GitRunner
    now: Callable[[], float]
    sleep: Callable[[float], None]
    work: Path
    train_seconds: int = DEFAULT_TRAIN_S
    salvage: bool = False
    verify_tries: int = VERIFY_TRIES
    verify_delay_s: float = VERIFY_DELAY_S
    nonce_factory: Callable[[], str] = lambda: uuid.uuid4().hex[:16]

    @property
    def kernel(self) -> str:
        return f"{self.user}/{kernel_slug(self.run_id, self.acc_key)}"

    @property
    def timeout_s(self) -> int:
        return ACCELERATORS[self.acc_key].session_s


def say(msg: str) -> None:
    print(msg, flush=True)


def fetch_dataset(ctx: Ctx, name: str) -> tuple[Path, dict[str, Any] | None, dict[str, Any]]:
    """Télécharge la dernière version complète du dataset. Retourne (dossier, verrou, manifest)."""
    dest = fresh_dir(ctx.work / name)
    ctx.cli.datasets_download(ctx.dataset, dest)
    if ctx.cli.dry:
        say("(à blanc) contenu distant inconnu : on suppose un run neuf (manifest vide, verrou libre)")
        return dest, None, empty_manifest(ctx.run_id)
    manifest = read_json(dest / MANIFEST_FILE)
    if manifest is None:
        raise RelayError(f"{MANIFEST_FILE} absent du dataset {ctx.dataset} : lancer `init` d'abord")
    if manifest.get("run_id") != ctx.run_id:
        raise RelayError(f"le dataset {ctx.dataset} porte le run {manifest.get('run_id')!r}, pas {ctx.run_id!r}")
    return dest, read_json(dest / LEASE_FILE), manifest


def checkpoint_files(folder: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    name = manifest.get("checkpoint")
    if not name:
        return {}
    path = folder / str(name)
    if not path.is_file():
        raise RelayError(f"checkpoint {name} listé dans le manifest mais absent du dataset")
    if sha256_file(path) != manifest.get("sha256"):
        raise RelayError(f"empreinte du checkpoint {name} différente de celle du manifest")
    return {str(name): path}


def cmd_init(ctx: Ctx) -> int:
    slug = ctx.dataset.split("/")[1]
    dest = fresh_dir(ctx.work / "dataset_init")
    write_json(
        dest / DATASET_META,
        {"id": ctx.dataset, "title": slug, "licenses": [{"name": "other"}]},  # titre 6–50 car., 1 licence (:6133-6139)
    )
    write_json(dest / MANIFEST_FILE, empty_manifest(ctx.run_id))
    write_json(dest / LEASE_FILE, {"schema": 1, "run_id": ctx.run_id, "user": None, "released": True})
    say(f"dataset initial construit : {dest}")
    ctx.cli.datasets_create(dest)
    say("Ensuite (interface web, non scriptable avec le CLI 2.2.5) : ajouter les 2 autres membres en collaborateurs.")
    return 0


def cmd_start(ctx: Ctx) -> int:
    commit = check_git(ctx.git, fetch=not ctx.cli.dry)
    folder, lease, manifest = fetch_dataset(ctx, "dataset_in")
    now = ctx.now()
    state = lease_state(lease, ctx.user, now)
    if state == "other":
        assert lease is not None
        raise RelayError(
            f"verrou actif de {lease['user']} (kernel {lease['kernel']}) jusqu'à {lease['expires_at']} : "
            "attendre son `finish` ou son `release`"
        )
    if state == "mine":
        assert lease is not None
        raise RelayError(
            f"vous tenez déjà le verrou (kernel {lease['kernel']}, jusqu'à {lease['expires_at']}) : "
            "lancer `finish` ou `release` avant une nouvelle session"
        )
    if state == "expired":
        assert lease is not None
        say(f"verrou expiré de {lease['user']} ({lease['expires_at']}) : reprise. Son kernel ne pourra plus publier.")

    session = len(manifest["sessions"]) + 1
    nonce = ctx.nonce_factory()
    run = {
        "run_id": ctx.run_id,
        "session": session,
        "user": ctx.user,
        "kernel": ctx.kernel,
        "commit": commit,
        "nonce": nonce,
        "accelerator": ACCELERATORS[ctx.acc_key].machine_shape,
        "expected_sha256": manifest.get("sha256"),
        "train_seconds": min(ctx.train_seconds, ctx.timeout_s - SAVE_MARGIN_S),
        "save_every_s": SAVE_EVERY_S,
    }
    kdir = build_kernel(ctx.work / "kernel", run, kernel_metadata(ctx.user, ctx.run_id, ctx.acc_key, ctx.dataset))
    say(f"kernel construit : {kdir} ({ctx.kernel}, session {session}, commit {commit[:10]})")

    mine = new_lease(ctx.run_id, ctx.user, ctx.kernel, commit, ctx.acc_key, ctx.timeout_s, now, nonce)
    files = checkpoint_files(folder, manifest)
    publish(
        ctx,
        "dataset_lease",
        files,
        manifest,
        mine,
        (lease, manifest),
        f"lease {ctx.run_id} s{session} {ctx.user} {commit[:10]}",
    )
    try:
        ctx.cli.kernels_push(kdir, ACCELERATORS[ctx.acc_key].machine_shape, ctx.timeout_s)
    except KaggleError as e:
        # Aucun kernel lancé (ou état inconnu) : on rend le verrou plutôt que de bloquer le groupe jusqu'à expiration.
        # Si un kernel tournait malgré tout, son `finish` serait refusé (nonce), sans corruption.
        rel = released(mine, ctx.user, ctx.now(), "push en échec")
        try:
            publish(
                ctx,
                "dataset_unlease",
                files,
                manifest,
                rel,
                (mine, manifest),
                f"release {ctx.run_id} {ctx.user} (push)",
            )
        except (KaggleError, RelayError) as e2:
            raise RelayError(f"{e} ; libération automatique en échec ({e2}) : lancer `release`") from e2
        raise RelayError(f"{e} ; verrou libéré automatiquement") from e
    write_json(ctx.work / "session.json", {**run, "lease_expires_at": mine["expires_at"], "dataset": ctx.dataset})
    say(
        f"suite : `status` pour suivre, puis `finish` une fois le kernel terminé (verrou jusqu'à {mine['expires_at']})."
    )
    return 0


def read_head(ctx: Ctx, name: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Relit `lease.json` et `manifest.json` de la dernière version visible. À blanc : (None, None)."""
    dest = fresh_dir(ctx.work / name)
    ctx.cli.datasets_download(ctx.dataset, dest, LEASE_FILE)
    ctx.cli.datasets_download(ctx.dataset, dest, MANIFEST_FILE)
    if ctx.cli.dry:
        return None, None
    return read_json(dest / LEASE_FILE), read_json(dest / MANIFEST_FILE)


def publish(
    ctx: Ctx,
    name: str,
    files: dict[str, Path],
    manifest: dict[str, Any],
    lease: dict[str, Any],
    head_before: tuple[dict[str, Any] | None, dict[str, Any] | None],
    message: str,
) -> None:
    """Publie une version (instantané complet) et ne rend la main qu'une fois cette version visible.

    1. Précontrôle : la tête du dataset doit être encore `head_before` (verrou et manifest lus au départ). Sinon un
       autre membre a publié entre-temps : on refuse au lieu d'écraser sa version.
    2. Après `datasets version`, relecture (jusqu'à `verify_tries` fois) : succès si la tête porte exactement notre
       verrou et notre manifest ; conflit si elle porte autre chose que l'ancienne tête.
    Réduit la fenêtre de course sans l'éliminer : Kaggle n'offre pas de compare-and-swap (docs/research/kaggle_relay.md
    §3). Toute version reste dans l'historique du dataset, donc rien n'est détruit."""
    folder = dataset_folder(ctx.work / name, ctx.dataset, files, manifest, lease)
    if ctx.cli.dry:
        read_head(ctx, f"{name}_precheck")
        ctx.cli.datasets_version(folder, message)
        read_head(ctx, f"{name}_check")
        return
    if read_head(ctx, f"{name}_precheck") != head_before:
        raise RelayError("le dataset a changé depuis la lecture (autre membre ?) : rien publié, relancer la commande")
    ctx.cli.datasets_version(folder, message)
    for attempt in range(1, ctx.verify_tries + 1):
        ctx.sleep(ctx.verify_delay_s)
        seen = read_head(ctx, f"{name}_check")
        if seen == (lease, manifest):
            say(f"version confirmée (relecture {attempt})")
            return
        if seen != head_before:
            who = seen[0].get("user") if seen[0] else None
            raise RelayError(f"une autre version (verrou de {who}) a remplacé la nôtre : vérifier `status`")
        say(f"nouvelle version pas encore visible (relecture {attempt}/{ctx.verify_tries})")
    raise RelayError(
        "notre version n'est pas visible après relecture : vérifier `status` avant de relancer "
        "(elle peut apparaître plus tard)."
    )


def cmd_status(ctx: Ctx) -> int:
    lease, manifest_or_none = read_head(ctx, "status")
    if ctx.cli.dry:
        ctx.cli.kernels_status(ctx.kernel)
        return 0
    manifest = manifest_or_none or {}
    state = lease_state(lease, ctx.user, ctx.now())
    say(f"run {manifest.get('run_id')} : {len(manifest.get('sessions', []))} session(s), {manifest.get('steps')} steps")
    say(
        f"verrou : {state}"
        + (f" ({lease['user']}, {lease['kernel']}, jusqu'à {lease['expires_at']})" if lease and state != "free" else "")
    )
    if held_by_me(lease, ctx) and lease is not None:
        stdout = ctx.cli.kernels_status(str(lease["kernel"])) or ""
        say(f"kernel : {parse_kernel_status(stdout)}")
    return 0


def held_by_me(lease: dict[str, Any] | None, ctx: Ctx) -> bool:
    """Verrou à nous, actif ou expiré. Expiré mais jamais repris, la dernière version porte encore notre nonce : publier
    ne prend la place de personne (et `publish` refuse si quelqu'un le reprend entre-temps)."""
    state = lease_state(lease, ctx.user, ctx.now())
    return lease is not None and state in ("mine", "expired") and lease.get("user") == ctx.user


def cmd_finish(ctx: Ctx) -> int:
    _, lease, manifest = fetch_dataset(ctx, "dataset_in")
    out_dir = fresh_dir(ctx.work / "output")
    if ctx.cli.dry:
        ctx.cli.kernels_status(ctx.kernel)
        ctx.cli.kernels_output(ctx.kernel, out_dir)
        publish(ctx, "dataset_out", {}, manifest, {}, (None, None), f"{ctx.run_id} s<n> {ctx.user}")
        return 0
    if not held_by_me(lease, ctx) or lease is None or lease.get("kernel") != ctx.kernel:
        holder = lease.get("user") if lease else None
        raise RelayError(f"le verrou n'est pas (ou plus) à vous pour {ctx.kernel} (détenteur : {holder}) : rien publié")
    status = parse_kernel_status(ctx.cli.kernels_status(ctx.kernel) or "")
    if status in ("queued", "running", "cancel_requested", "new_script", "unknown"):
        raise RelayError(f"kernel {ctx.kernel} au statut {status} : attendre la fin avant `finish`")
    if status != "complete" and not ctx.salvage:
        raise RelayError(f"kernel {ctx.kernel} au statut {status} : `--salvage` publie son dernier checkpoint sain")
    ctx.cli.kernels_output(ctx.kernel, out_dir)
    out = read_json(out_dir / OUT_FILE)
    if out is None:
        raise RelayError(f"{OUT_FILE} absent des sorties : aucun checkpoint à publier (`release` pour libérer)")
    session = len(manifest["sessions"]) + 1
    checks = {
        "run_id": ctx.run_id,
        "nonce": lease["nonce"],
        "session": session,
        "parent_sha256": manifest.get("sha256"),
    }
    for key, want in checks.items():
        if out.get(key) != want:
            raise RelayError(f"{OUT_FILE} : {key} = {out.get(key)!r}, attendu {want!r} (sortie d'une autre session ?)")
    name = str(out.get("checkpoint"))
    ckpt = out_dir / name
    if not CKPT_RE.fullmatch(name) or not ckpt.is_file():
        raise RelayError(f"checkpoint {name!r} absent ou nom invalide")
    sha = sha256_file(ckpt)
    if sha != out.get("sha256"):
        raise RelayError(f"empreinte de {name} ({sha}) différente de {OUT_FILE}")
    now = ctx.now()
    entry = {
        "session": session,
        "user": ctx.user,
        "kernel": ctx.kernel,
        "commit": lease["commit"],
        "accelerator": lease["accelerator"],
        "kernel_status": status,
        "checkpoint": name,
        "sha256": sha,
        "steps": out.get("steps"),
        "finished_at": iso(now),
    }
    new_manifest = {**manifest, "checkpoint": name, "sha256": sha, "steps": out.get("steps")}
    new_manifest["sessions"] = [*manifest["sessions"], entry]
    message = f"{ctx.run_id} s{session} {ctx.user} {str(lease['commit'])[:10]} {out.get('steps')} steps"
    rel = released(lease, ctx.user, now, "finish")
    publish(ctx, "dataset_out", {name: ckpt}, new_manifest, rel, (lease, manifest), message)
    say(f"session {session} publiée et confirmée ({name}, {out.get('steps')} steps) ; verrou libéré.")
    return 0


def cmd_release(ctx: Ctx) -> int:
    folder, lease, manifest = fetch_dataset(ctx, "dataset_in")
    if ctx.cli.dry:
        publish(ctx, "dataset_release", {}, manifest, {}, (None, None), f"release {ctx.run_id} {ctx.user}")
        return 0
    if lease_state(lease, ctx.user, ctx.now()) == "free":
        say("verrou déjà libre : rien à faire")
        return 0
    if not held_by_me(lease, ctx) or lease is None:
        raise RelayError(f"le verrou appartient à {lease['user'] if lease else None} : seul son détenteur le libère")
    rel = released(lease, ctx.user, ctx.now(), "release")
    files = checkpoint_files(folder, manifest)
    publish(ctx, "dataset_release", files, manifest, rel, (lease, manifest), f"release {ctx.run_id} {ctx.user}")
    say(f"verrou libéré. Le kernel {lease['kernel']} n'est pas arrêté : l'annuler dans l'interface web s'il tourne.")
    return 0


COMMANDS: dict[str, Callable[[Ctx], int]] = {
    "init": cmd_init,
    "start": cmd_start,
    "status": cmd_status,
    "finish": cmd_finish,
    "release": cmd_release,
}


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--user", required=True, help="votre pseudo Kaggle (minuscules)")
    ap.add_argument("--run", required=True, help="identifiant du run (minuscules, chiffres, tirets)")
    ap.add_argument("--acc", choices=sorted(ACCELERATORS), default="t4")
    ap.add_argument("--dataset", required=True, help="dataset de relais <propriétaire>/<slug>, privé et partagé")
    ap.add_argument("--push", action="store_true", help="exécute réellement les commandes kaggle (sinon : à blanc)")
    ap.add_argument("--train-seconds", type=int, default=DEFAULT_TRAIN_S, help="durée de la boucle du kernel")
    ap.add_argument("--salvage", action="store_true", help="finish : publier aussi après un kernel en erreur")
    args = ap.parse_args(argv)
    for value, rx, what in (
        (args.user, USER_RE, "pseudo"),
        (args.run, RUN_RE, "run"),
        (args.dataset, DATASET_RE, "dataset"),
    ):
        if not rx.fullmatch(value):
            ap.error(f"{what} invalide : {value!r}")
    if args.train_seconds <= 0:
        ap.error("--train-seconds doit être positif")
    return args


def main(
    argv: Sequence[str] | None = None,
    executor: Executor | None = None,
    git: GitRunner = real_git,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    work_root: Path = BUILD,
) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    if args.push and executor is None:
        executor = subprocess_executor
    cli = KaggleCli(executor if args.push else None, exe=kaggle_exe())
    ctx = Ctx(
        user=args.user,
        run_id=args.run,
        acc_key=args.acc,
        dataset=args.dataset,
        cli=cli,
        git=git,
        now=now,
        sleep=sleep,
        work=work_root / args.run / args.command,
        train_seconds=args.train_seconds,
        salvage=args.salvage,
    )
    if cli.dry:
        say("mode à blanc : aucune commande kaggle n'est lancée (ajouter --push pour agir)")
    try:
        return COMMANDS[args.command](ctx)
    except (RelayError, KaggleError) as e:
        print(f"refus : {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
