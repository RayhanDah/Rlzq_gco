"""Checkpoints atomiques, vérifiés et à reprise exacte (ADR-003 §7, `.claude/rules/training.md`, spec §7.7).

Format de fichier : `MAGIC (8) | sha256(corps) (32) | corps`, corps = `uint32 BE longueur méta | méta JSON | charge`,
charge = `torch.save` d'un dictionnaire de tenseurs et de types simples, relu par `torch.load(weights_only=True)`
(aucun pickle arbitraire). La somme de contrôle est vérifiée AVANT toute désérialisation.

Écriture : fichier temporaire du même dossier, flush + fsync, `os.replace`, puis rotation. Un nom de fichier n'est
jamais réutilisé (`ckpt_{index:06d}_it{itération:08d}.pt`, index croissant) : aucun checkpoint n'est écrasé. Les
fichiers `.part` laissés par un arrêt brutal sont ignorés (jamais chargés, jamais supprimés par la rotation).
La rotation ne supprime que des fichiers de ce format, dans le dossier passé en paramètre, et conserve les `keep`
derniers et le dernier checkpoint « sain » (celui dont la vérification passe).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import random
import re
import struct
import subprocess
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch

from actions import ACTION_SCHEMA_VERSION, ACTION_TABLE_SHA256
from obs import OBS_SCHEMA_VERSION
from training.ppo import OverlapTrainer
from training.rollout import RolloutBatch

CHECKPOINT_FORMAT_VERSION = 1
MAGIC = b"RLGCOCK1"
_NAME_RE = re.compile(r"^ckpt_(\d{6})_it(\d{8})\.pt$")
_ND = "__nd__"
_REPLACE_RETRIES = 8


class CheckpointError(RuntimeError):
    """Erreur générique de checkpoint."""


class CheckpointCorruptError(CheckpointError):
    """Fichier tronqué, altéré (somme de contrôle) ou illisible : jamais chargé."""


class CheckpointIncompatibleError(CheckpointError):
    """Checkpoint valide mais produit sous d'autres versions de schéma, table d'actions, config reward ou config."""


class StatefulEnv(Protocol):
    """Env restaurable : `load_state_dict` rend l'observation courante si l'état n'est pas exact, sinon None."""

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> npt.NDArray[np.float32] | None: ...


# ---------------------------------------------------------------------------------------------------- encodage


def encode(obj: Any) -> Any:
    """Rend `obj` sérialisable par `torch.save` ET relisible en `weights_only=True` : ndarray -> {"__nd__": tenseur},
    tenseurs ramenés sur CPU, conteneurs récursifs. Types acceptés : dict à clés str/int, list, tuple, scalaires,
    str, bytes, None, ndarray, tenseur, RolloutBatch."""
    if isinstance(obj, RolloutBatch):
        return {"__batch__": {f.name: encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}}
    if isinstance(obj, np.ndarray):
        return {_ND: torch.from_numpy(np.ascontiguousarray(obj)).clone()}
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, Mapping):
        return {k: encode(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [encode(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, bool | int | float | str | bytes):
        return obj
    raise TypeError(f"type non sérialisable dans un checkpoint : {type(obj).__name__}")


def decode(obj: Any) -> Any:
    """Inverse d'`encode` (les tuples reviennent en listes)."""
    if isinstance(obj, dict):
        if set(obj) == {_ND}:
            arr: Any = obj[_ND].numpy().copy()
            return arr
        if set(obj) == {"__batch__"}:
            return RolloutBatch(**{k: decode(v) for k, v in obj["__batch__"].items()})
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------------------------------- identité


def hash_config(cfg: Any) -> str:
    """SHA-256 d'une configuration (dataclass ou mapping) en JSON canonique (clés triées)."""
    data = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type) else cfg
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(repo_dir: str | Path) -> str:
    """Commit HEAD de `repo_dir`, ou "unknown" si git ou le dépôt est indisponible."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


@dataclass(frozen=True)
class CheckpointIdentity:
    """Ce qui doit coïncider entre le checkpoint et le code qui le charge (ADR-003 §7)."""

    reward_config_hash: str
    reward_spec_version: str
    reward_hash_schema_version: int
    obs_schema_version: int
    obs_dim: int
    action_schema_version: int
    action_table_sha256: str
    tick_skip: int

    @classmethod
    def current(cls, *, reward_config_hash: str, obs_dim: int, tick_skip: int) -> CheckpointIdentity:
        from rewards import HASH_SCHEMA_VERSION, SPEC_VERSION

        return cls(
            reward_config_hash=reward_config_hash,
            reward_spec_version=SPEC_VERSION,
            reward_hash_schema_version=HASH_SCHEMA_VERSION,
            obs_schema_version=OBS_SCHEMA_VERSION,
            obs_dim=obs_dim,
            action_schema_version=ACTION_SCHEMA_VERSION,
            action_table_sha256=ACTION_TABLE_SHA256,
            tick_skip=tick_skip,
        )


@dataclass(frozen=True)
class RestoreInfo:
    iteration: int
    counters: dict[str, Any]
    extra: dict[str, Any]
    exact_env: bool | None
    """True : l'env a été restauré bit à bit ; False : « modulo reset des arènes » ; None : aucun état d'env."""
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------- capture


def _rng_state() -> dict[str, Any]:
    py = random.getstate()
    kind, key, pos, has_gauss, cached = np.random.get_state()  # noqa: NPY002 (état global demandé par ADR-003 §7)
    return {
        "python": [py[0], list(py[1]), py[2]],
        "numpy": [kind, np.asarray(key, dtype=np.int64), int(pos), int(has_gauss), float(cached)],
        "torch": torch.get_rng_state(),
    }


def _set_rng_state(state: Mapping[str, Any]) -> None:
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))  # noqa: NPY002
    torch.set_rng_state(state["torch"])


def capture(
    trainer: OverlapTrainer,
    *,
    identity: CheckpointIdentity,
    train_config: Any,
    iteration: int,
    env_state: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    git_commit_id: str = "unknown",
    rocketgpu_commit: str = "unknown",
) -> dict[str, Any]:
    """État complet à la frontière d'itération (lot k+1 déjà collecté inclus). À appeler hors du thread learner.
    `extra` : compteurs ou calendriers du trainer (γ, annealing de `p_kick`...), types simples uniquement."""
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "identity": dataclasses.asdict(identity),
        "train_config": json.loads(json.dumps(train_config, default=lambda o: dataclasses.asdict(o))),
        "train_config_hash": hash_config(train_config),
        "iteration": int(iteration),
        "counters": {
            "iteration": int(iteration),
            "agent_steps": int(trainer.collector.agent_steps),
            "policy_version": int(trainer.learner.version),
        },
        "extra": dict(extra or {}),
        "versions": {
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "git_commit": git_commit_id,
            "rocketgpu_commit": rocketgpu_commit,
        },
        "rng": _rng_state(),
        "trainer": trainer.state_dict(),
        "env": dict(env_state) if env_state is not None else None,
    }


def check_identity(
    state: Mapping[str, Any],
    expected: CheckpointIdentity,
    train_config: Any | None = None,
) -> None:
    """Lève `CheckpointIncompatibleError` (message clair, toutes les différences) si le checkpoint ne correspond pas."""
    if state.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointIncompatibleError(
            f"format de checkpoint {state.get('format_version')} != {CHECKPOINT_FORMAT_VERSION}"
        )
    saved = state["identity"]
    diffs = [
        f"{k}: checkpoint={saved.get(k)!r}, courant={v!r}"
        for k, v in dataclasses.asdict(expected).items()
        if saved.get(k) != v
    ]
    if train_config is not None and state["train_config_hash"] != hash_config(train_config):
        diffs.append("train_config_hash: la configuration d'entraînement a changé")
    if diffs:
        raise CheckpointIncompatibleError("checkpoint incompatible : " + " ; ".join(diffs))


def restore(
    state: Mapping[str, Any],
    trainer: OverlapTrainer,
    *,
    expected: CheckpointIdentity,
    train_config: Any | None = None,
    env: StatefulEnv | None = None,
) -> RestoreInfo:
    """Valide l'identité puis restaure RNG, env et trainer (objets neufs de même forme). `state` vient de
    `CheckpointManager.load`. Si l'env ne se restaure pas exactement, le collecteur reçoit l'obs du nouveau tirage."""
    check_identity(state, expected, train_config)
    if (state["env"] is None) != (env is None):
        raise CheckpointIncompatibleError("état d'env présent dans le checkpoint XOR env fourni : incohérent")
    obs = None
    exact: bool | None = None
    if env is not None and state["env"] is not None:
        obs = env.load_state_dict(state["env"])
        exact = obs is None
    trainer.load_state_dict(state["trainer"], obs=obs)
    _set_rng_state(state["rng"])
    return RestoreInfo(
        iteration=int(state["iteration"]),
        counters=dict(state["counters"]),
        extra=dict(state["extra"]),
        exact_env=exact,
        meta=dict(state["versions"]),
    )


# ---------------------------------------------------------------------------------------------------- fichier


def _fsync_dir(directory: Path) -> None:
    if os.name == "nt":  # pas de descripteur de dossier sous Windows
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_with_retry(src: Path, dst: Path) -> None:
    """`os.replace` ; sous Windows il échoue (PermissionError) si une des deux extrémités est ouverte ailleurs
    (antivirus, indexeur) : quelques tentatives espacées, puis l'erreur remonte, l'ancien fichier restant intact."""
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _pack(state: Mapping[str, Any], meta: Mapping[str, Any]) -> bytes:
    buf = io.BytesIO()
    torch.save(encode(state), buf)
    meta_b = json.dumps(meta, sort_keys=True).encode("utf-8")
    body = struct.pack(">I", len(meta_b)) + meta_b + buf.getvalue()
    return MAGIC + hashlib.sha256(body).digest() + body


def _read_header(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Vérifie magie et somme de contrôle ; rend (méta, charge)."""
    if len(data) < len(MAGIC) + 32 + 4 or data[: len(MAGIC)] != MAGIC:
        raise CheckpointCorruptError("en-tête de checkpoint absent ou fichier tronqué")
    digest, body = data[len(MAGIC) : len(MAGIC) + 32], data[len(MAGIC) + 32 :]
    if hashlib.sha256(body).digest() != digest:
        raise CheckpointCorruptError("somme de contrôle invalide (fichier altéré ou incomplet)")
    (n,) = struct.unpack(">I", body[:4])
    if 4 + n > len(body):
        raise CheckpointCorruptError("longueur de métadonnées incohérente")
    meta: dict[str, Any] = json.loads(body[4 : 4 + n])
    return meta, body[4 + n :]


def read_meta(path: Path) -> dict[str, Any]:
    """Métadonnées de l'en-tête SANS vérifier la somme de contrôle (listage rapide uniquement)."""
    with open(path, "rb") as f:
        head = f.read(len(MAGIC) + 32 + 4)
        if len(head) < len(MAGIC) + 36 or head[: len(MAGIC)] != MAGIC:
            raise CheckpointCorruptError(f"{path.name} : en-tête invalide")
        (n,) = struct.unpack(">I", head[-4:])
        raw = f.read(n)
    if len(raw) != n:
        raise CheckpointCorruptError(f"{path.name} : métadonnées tronquées")
    meta: dict[str, Any] = json.loads(raw)
    return meta


@dataclass(frozen=True)
class CheckpointRef:
    path: Path
    index: int
    iteration: int


class CheckpointManager:
    """Écrit, liste, charge et fait tourner les checkpoints d'un dossier (`directory`, créé si besoin)."""

    def __init__(self, directory: str | Path, keep: int = 3) -> None:
        if keep < 1:
            raise ValueError("keep >= 1")
        self.directory = Path(directory)
        self.keep = keep
        self.directory.mkdir(parents=True, exist_ok=True)

    def checkpoints(self) -> list[CheckpointRef]:
        """Checkpoints de ce format, du plus ancien au plus récent (par index)."""
        refs = []
        for p in self.directory.iterdir():
            m = _NAME_RE.match(p.name)
            if m and p.is_file():
                refs.append(CheckpointRef(p, int(m.group(1)), int(m.group(2))))
        return sorted(refs, key=lambda r: r.index)

    def save(self, state: Mapping[str, Any], *, healthy: bool = True) -> Path:
        """Écriture atomique puis rotation. En cas d'erreur, rien n'est supprimé et les anciens fichiers restent
        intacts."""
        refs = self.checkpoints()
        index = (refs[-1].index + 1) if refs else 0
        iteration = int(state["iteration"])
        meta = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "iteration": iteration,
            "index": index,
            "healthy": bool(healthy),
            "created_unix": time.time(),
            "train_config_hash": state.get("train_config_hash"),
            "reward_config_hash": state["identity"]["reward_config_hash"],
        }
        data = _pack(state, meta)
        path = self.directory / f"ckpt_{index:06d}_it{iteration:08d}.pt"
        if path.exists():
            raise CheckpointError(f"{path.name} existe déjà : jamais écrasé")
        _atomic_write(path, data)
        self._rotate()
        return path

    def _verified(self, ref: CheckpointRef) -> bool:
        try:
            _read_header(ref.path.read_bytes())
        except (CheckpointCorruptError, OSError):
            return False
        return True

    def _rotate(self) -> None:
        refs = self.checkpoints()
        protected = {r.path for r in refs[-self.keep :]}
        for ref in reversed(refs):  # dernier sain : le plus récent sain ET intègre
            try:
                healthy = bool(read_meta(ref.path).get("healthy"))
            except (CheckpointCorruptError, OSError, ValueError):
                continue
            if healthy and (ref.path in protected or self._verified(ref)):
                protected.add(ref.path)
                break
        for ref in refs:
            if ref.path not in protected:
                ref.path.unlink()

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        """Charge un checkpoint : somme de contrôle vérifiée d'abord, puis `torch.load(weights_only=True)`."""
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            raise CheckpointCorruptError(f"lecture impossible : {exc}") from exc
        _, payload = _read_header(data)
        try:
            raw = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CheckpointCorruptError(f"désérialisation refusée : {exc}") from exc
        out: dict[str, Any] = decode(raw)
        return out

    def load_latest(self, *, healthy_only: bool = False) -> tuple[Path, dict[str, Any]] | None:
        """Le plus récent checkpoint intègre (et sain si demandé) ; les fichiers corrompus sont sautés, jamais
        modifiés. None si aucun."""
        for ref in reversed(self.checkpoints()):
            if healthy_only:
                try:
                    if not read_meta(ref.path).get("healthy"):
                        continue
                except (CheckpointCorruptError, OSError, ValueError):
                    continue
            try:
                return ref.path, self.load(ref.path)
            except CheckpointCorruptError:
                continue
        return None
