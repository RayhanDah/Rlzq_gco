"""Kernel Kaggle T4 du run long de démo (ADR-014 D6) : une session de jour du run `long_s0`, relayée avec le laptop.

Rendu par `scripts/kaggle_long.py start`, qui remplace CONFIG (commit, run, durée de session, mode smoke). Le kernel ne
possède aucun identifiant Kaggle et ne publie rien :
1. clone le dépôt public au commit donné, installe cmake/ninja/pybind11, télécharge les meshes (RLGym, commit épinglé),
   compile RocketGPU en CPU (la simulation tourne sur les 4 vCPU, le learner sur le GPU) ;
2. restaure le run depuis le dataset privé monté en entrée (`bundle.tar` + `long-bundle.json`, écrits par
   `kaggle_long.py pack`), en vérifiant l'empreinte annoncée ;
3. lance `scripts/run_4a.py --variant long --device cuda --max-hours H`, H calculé pour finir avant le timeout ;
4. quoi qu'il arrive, archive le run dans /kaggle/working/bundle.tar et écrit relay-out.json (empreinte, itération,
   statut), récupérés par `kaggle_long.py fetch`.
Mode smoke (CONFIG["smoke"]) : `scripts/train.py configs/smoke.yaml --smoke --device cuda` sur une copie neuve, pas de
reprise du run long ; seul le rapport est produit.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

CONFIG: dict[str, object] = {}  # remplacé au rendu par kaggle_long.py
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
APP = Path("/tmp/rlgco")  # code, meshes et binaires : hors des sorties
RELAY = Path("/tmp/relay")  # runs/ et checkpoints/ du run pendant la session (archivés à la fin)
BUNDLE_MANIFEST = "long-bundle.json"
BUNDLE_TAR = "bundle.tar"
T0 = time.time()
REPORT: dict[str, object] = {"steps": {}}


def log(msg: str) -> None:
    print(f"[long_kernel {time.time() - T0:7.0f} s] {msg}", flush=True)


def save_report() -> None:
    REPORT["elapsed_s"] = round(time.time() - T0, 1)
    tmp = WORK / ".relay-out.json.tmp"
    tmp.write_text(json.dumps(REPORT, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, WORK / "relay-out.json")


def step(
    name: str,
    cmd: list[str],
    timeout_s: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    grace_s: float = 300.0,
) -> bool:
    """Lance une commande, journal complet dans /kaggle/working/logs/<name>.log ; rend True si code 0.
    La commande a son propre groupe de processus. Au timeout, OU si le chef du groupe sort en laissant des descendants
    (superviseur run_4a mort avant sa session enfant), tout le groupe est arrêté avant de rendre la main : rien n'écrit
    plus dans /tmp/relay quand `archive_run` s'exécute."""
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with (logs / f"{name}.log").open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            code = -9
        if group_alive(proc.pid):
            stop_group(proc, grace_s)
            code = code if code != 0 else -15  # descendants encore vivants : session non terminée proprement
    steps = REPORT["steps"]
    assert isinstance(steps, dict)
    steps[name] = {"code": code, "seconds": round(time.time() - t, 1)}
    log(f"{name} : code {code} en {time.time() - t:.0f} s")
    save_report()
    return code == 0


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_group(proc: subprocess.Popen[bytes], grace_s: float) -> None:
    """SIGINT au groupe (arrêt propre : checkpoint final), `grace_s` de grâce, puis SIGKILL ; attend que le groupe
    soit vide (au plus 60 s après SIGKILL)."""
    for sig, wait_s in ((signal.SIGINT, grace_s), (signal.SIGKILL, 60.0)):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, sig)
        end = time.time() + wait_s
        while time.time() < end:
            if proc.poll() is not None and not group_alive(proc.pid):
                log(f"groupe {proc.pid} arrêté ({sig.name})")
                return
            time.sleep(1.0)
    log(f"groupe {proc.pid} : processus encore présents après SIGKILL")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_bundle() -> tuple[dict[str, object], Path]:
    """Manifest et archive du dataset monté (chemin exact du montage non vérifié : on parcourt /kaggle/input)."""
    for manifest_path in sorted(INPUT.rglob(BUNDLE_MANIFEST)):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run") != CONFIG["run"]:
            continue
        tar = manifest_path.parent / BUNDLE_TAR
        if not tar.is_file():
            raise SystemExit(f"{BUNDLE_TAR} absent à côté de {manifest_path} (archive décompressée par Kaggle ?)")
        return manifest, tar
    raise SystemExit(f"aucun {BUNDLE_MANIFEST} du run {CONFIG['run']} sous {INPUT}")


def restore_bundle() -> dict[str, object]:
    manifest, tar = find_bundle()
    sha = sha256_file(tar)
    if sha != manifest.get("sha256") or sha != CONFIG.get("bundle_sha256"):
        raise SystemExit(
            f"archive d'entrée inattendue : {sha} (manifest {manifest.get('sha256')}, attendu "
            f"{CONFIG.get('bundle_sha256')}) : version du dataset périmée ?"
        )
    RELAY.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as tf:
        tf.extractall(RELAY, filter="data")  # refuse les chemins absolus et les liens sortants
    REPORT["start_sha256"] = sha  # comparé par `kaggle_long.py sync` à l'empreinte du pack
    log(f"run restauré depuis {tar} (itération {manifest.get('iteration')})")
    return manifest


def archive_run() -> None:
    """Archive runs/<run> et checkpoints/<run> (dernier checkpoint sain et instantanés) dans /kaggle/working."""
    run = str(CONFIG["run"])
    runs, cks = RELAY / "runs" / run, RELAY / "checkpoints" / run
    if not runs.is_dir() or not cks.is_dir():
        REPORT["bundle"] = None
        return
    tmp = WORK / f".{BUNDLE_TAR}.tmp"

    def no_temp(info: tarfile.TarInfo) -> tarfile.TarInfo | None:  # écritures interrompues : jamais archivées
        base = info.name.rsplit("/", 1)[-1]
        return None if base.endswith((".tmp", ".part", ".sync-tmp")) else info

    with tarfile.open(tmp, "w") as tf:
        tf.add(runs, arcname=f"runs/{run}", filter=no_temp)
        tf.add(cks, arcname=f"checkpoints/{run}", filter=no_temp)
    os.replace(tmp, WORK / BUNDLE_TAR)
    ckpts = sorted(p.name for p in cks.glob("ckpt_*.pt"))
    REPORT["bundle"] = {"file": BUNDLE_TAR, "sha256": sha256_file(WORK / BUNDLE_TAR), "checkpoints": ckpts}
    log(f"run archivé : {BUNDLE_TAR}, checkpoints {ckpts}")


def setup() -> dict[str, str]:
    py = sys.executable
    if not step("clone", ["git", "clone", "--quiet", str(CONFIG["repo"]), str(APP)], 300):
        raise SystemExit("clone du dépôt impossible")
    if not step("checkout", ["git", "-C", str(APP), "checkout", "--quiet", str(CONFIG["commit"])], 60):
        raise SystemExit(f"commit {CONFIG['commit']} introuvable (poussé ?)")
    step("pip", [py, "-m", "pip", "install", "-q", "cmake", "ninja", "pybind11", "pyyaml"], 600)
    meshes = APP / "vendor" / "collision_meshes"
    if not step("meshes", [py, str(APP / "scripts" / "fetch_meshes.py"), "--dest", str(meshes)], 300):
        raise SystemExit("meshes indisponibles")
    build = APP / "vendor" / "build"
    if not step(
        "build_rocketgpu_cpu",
        [
            py,
            str(APP / "scripts" / "build_rocketgpu.py"),
            "--variant",
            "cpu",
            "--out",
            str(build),
            "--meshes",
            str(meshes),
        ],
        1800,
    ):
        raise SystemExit("build RocketGPU CPU en échec")
    env = dict(os.environ)
    env["ROCKETGPU_PY_DIR"] = str(build / "rocketgpu-cpu" / "python")
    env["ROCKETGPU_COLLISION_MESH_DIR"] = str(meshes)
    env["PYTHONIOENCODING"] = "utf-8"
    versions = subprocess.run(
        [
            py,
            "-c",
            "import torch,numpy,sys;print(sys.version.split()[0],torch.__version__,numpy.__version__,"
            "torch.cuda.is_available(),torch.cuda.device_count())",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    REPORT["versions"] = versions
    log(f"python, torch, numpy, cuda, n_gpu : {versions}")
    return env


def main() -> int:
    if not CONFIG:
        print("CONFIG vide : ce modèle doit être rendu par scripts/kaggle_long.py", file=sys.stderr)
        return 2
    WORK.mkdir(parents=True, exist_ok=True)
    REPORT.update({"run": CONFIG["run"], "commit": CONFIG["commit"], "smoke": bool(CONFIG["smoke"]), "status": "setup"})
    save_report()
    py = sys.executable
    try:
        env = setup()
        if CONFIG["smoke"]:
            ok = step(
                "smoke",
                [
                    py,
                    "scripts/train.py",
                    "configs/smoke.yaml",
                    "--smoke",
                    "--device",
                    "cuda",
                    "--out-dir",
                    str(WORK / "smoke"),
                    "--checkpoint-dir",
                    str(RELAY / "smoke_ck"),
                    "--summary",
                    str(WORK / "smoke_summary.json"),
                ],
                1800,
                cwd=APP,
                env=env,
            )
            REPORT["status"] = "smoke_ok" if ok else "smoke_failed"
            return 0
        manifest = restore_bundle()
        REPORT["start_iteration"] = manifest.get("iteration")
        margin_s = float(str(CONFIG["margin_s"]))
        hours = (float(str(CONFIG["session_s"])) - (time.time() - T0) - margin_s) / 3600.0
        if hours <= 0.1:
            raise SystemExit(f"plus assez de temps pour une session ({hours:.2f} h)")
        REPORT["max_hours"] = round(hours, 3)
        REPORT["status"] = "training"
        save_report()
        cmd = [
            py,
            "scripts/run_4a.py",
            "--variant",
            "long",
            "--seed",
            str(CONFIG["seed"]),
            "--device",
            "cuda",
            "--max-hours",
            f"{hours:.3f}",
            "--no-keep-awake",
            "--no-auto-resume",
            "--runs-root",
            str(RELAY / "runs"),
            "--checkpoints-root",
            str(RELAY / "checkpoints"),
        ]
        if CONFIG.get("max_iterations"):  # mode essai : reprise réelle, quelques itérations seulement
            cmd += ["--max-iterations", str(CONFIG["max_iterations"])]
        ok = step("run_4a", cmd, hours * 3600.0 + margin_s / 2, cwd=APP, env=env, grace_s=min(300.0, margin_s / 4))
        REPORT["status"] = "paused" if ok else "run_failed"
        return 0
    except SystemExit as e:
        REPORT["status"] = "error"
        REPORT["error"] = str(e)
        log(f"erreur : {e}")
        return 0  # statut dans relay-out.json ; code 0 pour que Kaggle garde les sorties (non vérifié sinon)
    finally:
        if not CONFIG.get("smoke"):
            try:
                archive_run()
            except Exception as e:  # le rapport doit être écrit quoi qu'il arrive
                REPORT["archive_error"] = repr(e)
        with contextlib.suppress(OSError):  # disque plein : le journal du kernel garde la trace
            save_report()
        shutil.rmtree(RELAY / "smoke_ck", ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
