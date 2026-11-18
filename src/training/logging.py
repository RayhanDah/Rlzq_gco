"""Journalisation des scalaires par itération : JSONL (stdlib) derrière une petite interface, TensorBoard optionnel.

`tensorboard` n'est pas installé dans le venv (vérifié : `import tensorboard` -> ModuleNotFoundError) ; le writer
TensorBoard est importé paresseusement à la construction et échoue clairement si le paquet manque. Dépendance à
ajouter si on le veut : `tensorboard` (groupe `learn`).
Aucune écriture hors des chemins passés en paramètre.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from training.ppo import IterationResult, Learner
from training.rollout import RolloutBatch


class ScalarWriter(Protocol):
    def write(self, step: int, scalars: Mapping[str, float]) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class JsonlWriter:
    """Une ligne JSON par appel (`{"step": n, "t": unix, ...scalaires}`), ajoutée et vidée à chaque écriture.
    Les valeurs non finies sont écrites en chaînes ("nan", "inf") : le JSON reste valide et le NaN reste visible."""

    def __init__(self, path: str | Path, fsync: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")
        self._fsync = fsync

    def write(self, step: int, scalars: Mapping[str, float]) -> None:
        row: dict[str, Any] = {"step": int(step), "t": time.time()}
        for k, v in scalars.items():
            row[k] = v if math.isfinite(v) else str(float(v))
        self._f.write(json.dumps(row, sort_keys=True) + "\n")
        self._f.flush()
        if self._fsync:
            os.fsync(self._f.fileno())

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


def append_jsonl(path: str | Path, obj: Mapping[str, Any], *, fsync: bool = True) -> None:
    """Ajoute une ligne JSON (clés triées, `default=str`) à `path`, vidée et, si `fsync`, synchronisée sur disque."""
    line = json.dumps(dict(obj), sort_keys=True, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        if fsync:
            os.fsync(f.fileno())


def try_append_jsonl(path: str | Path, obj: Mapping[str, Any], *, fsync: bool = True) -> Exception | None:
    """`append_jsonl` qui ne lève JAMAIS d'`Exception` : à utiliser dans un gestionnaire d'erreur, où une erreur
    d'écriture du journal (disque plein, fichier verrouillé, objet non sérialisable) masquerait l'exception d'origine.
    Rend l'exception d'écriture (None si la ligne est écrite) et la signale sur stderr ; à l'appelant de la rattacher
    (p. ex. `exc.add_note`) s'il le souhaite. `BaseException` (Ctrl+C) n'est pas avalée."""
    try:
        append_jsonl(path, obj, fsync=fsync)
    except Exception as err:
        try:
            print(f"[journal] écriture impossible dans {path} : {err!r}", file=sys.stderr, flush=True)
        except Exception:  # stderr fermé ou illisible : rien d'autre à faire
            pass
        return err
    return None


class TensorBoardWriter:
    """Writer TensorBoard (`torch.utils.tensorboard.SummaryWriter`), importé à la construction."""

    def __init__(self, logdir: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError("TensorBoard indisponible : installer le paquet `tensorboard`") from exc
        self._w = SummaryWriter(log_dir=str(logdir))

    def write(self, step: int, scalars: Mapping[str, float]) -> None:
        for k, v in scalars.items():
            self._w.add_scalar(k, v, step)

    def flush(self) -> None:
        self._w.flush()

    def close(self) -> None:
        self._w.close()


class MultiWriter:
    def __init__(self, writers: Sequence[ScalarWriter]) -> None:
        self._writers = list(writers)

    def write(self, step: int, scalars: Mapping[str, float]) -> None:
        for w in self._writers:
            w.write(step, scalars)

    def flush(self) -> None:
        for w in self._writers:
            w.flush()

    def close(self) -> None:
        for w in self._writers:
            w.close()


class ThroughputMeter:
    """Débit en transitions agents par seconde entre deux appels (horloge injectable pour les tests)."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._last: tuple[float, int] | None = None

    def tick(self, agent_steps: int) -> float:
        """Rend le débit depuis l'appel précédent ; 0.0 au premier appel (rien à mesurer)."""
        now = self._clock()
        prev, self._last = self._last, (now, agent_steps)
        if prev is None or now <= prev[0]:
            return 0.0
        return (agent_steps - prev[1]) / (now - prev[0])


def grad_norms(learner: Learner) -> dict[str, float]:
    """Normes L2 des gradients de π et de V tels que laissés par le dernier minibatch de l'itération (APRÈS clip global,
    donc réduites du même facteur ; `ppo/grad_norm` des stats est la norme globale AVANT clip). Lecture seule."""

    def norm(params: Iterable[torch.nn.Parameter]) -> float:
        sq = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
        return float(torch.stack(sq).sum().sqrt().item()) if sq else float("nan")

    return {
        "grad/policy_last_mb_postclip": norm(learner.policy.parameters()),
        "grad/value_last_mb_postclip": norm(learner.value.parameters()),
    }


def batch_scalars(batch: RolloutBatch, n_cars: int) -> dict[str, float]:
    """Buts et longueur d'épisode du lot appris. Buts = fins terminales / `n_cars` (une fin par agent de l'arène).
    Longueur d'épisode : estimation T·N / (fins terminales + troncatures) en transitions agents, biaisée par les
    épisodes enjambant le lot (estimé, non mesuré à l'échelle)."""
    n_term = int(batch.term.sum())
    n_trunc = int((batch.trunc & ~batch.term).sum())
    ends = n_term + n_trunc
    return {
        "env/goals": n_term / n_cars,
        "env/episode_ends": ends / n_cars,
        "env/episode_len_est": (batch.n_steps * batch.n_agents / ends) if ends else float("nan"),
        "env/reward_mean": float(np.mean(batch.rew)),
    }


def end_scalars(counts: Mapping[str, int]) -> dict[str, float]:
    """ADR-006 I3/I4 : fins d'épisode du lot collecté, par arène (pas par agent), depuis `Collector.end_counts`.
    Cause : but / no-touch / durée max (exclusives, somme = fins). Origine : kickoff / générique (somme = fins).
    `ends_touched` : fins avec au moins une touche (diagnostic). Rien ici n'entre dans la récompense ni la perte."""
    return {
        "env/ends_goal": float(counts["goal"]),
        "env/ends_no_touch": float(counts["no_touch"]),
        "env/ends_max_time": float(counts["max_time"]),
        "env/ends_kickoff": float(counts["kickoff"]),
        "env/ends_generic": float(counts["generic"]),
        "env/ends_touched": float(counts["touched"]),
        "env/goals_kickoff": float(counts["goals_kickoff"]),
        "env/goals_generic": float(counts["goals_generic"]),
    }


def iteration_scalars(
    result: IterationResult,
    *,
    p_kick: float | None = None,
    steps_per_s: float | None = None,
    batch: RolloutBatch | None = None,
    n_cars: int = 2,
    learner: Learner | None = None,
) -> dict[str, float]:
    """Scalaires d'une itération : stats PPO (`ppo/*`), journal SIL (`sil/*`, si activée), compteurs, débit,
    `p_kick`, staleness, buts et longueur d'épisode (si `batch`), normes de gradient par réseau (si `learner`)."""
    out: dict[str, float] = {f"ppo/{k}": float(v) for k, v in result.stats.as_dict().items()}
    out.update({f"sil/{k}": float(v) for k, v in result.stats.sil.items()})  # C4 ; vide sans self-imitation
    out["train/agent_steps"] = float(result.agent_steps)
    out["train/policy_version"] = float(result.policy_version)
    out["train/learned_batch_version"] = float(result.learned_batch_version)
    out["train/staleness"] = float(result.stats.staleness)
    if steps_per_s is not None:
        out["perf/agent_steps_per_s"] = float(steps_per_s)
    if p_kick is not None:
        out["env/p_kick"] = float(p_kick)
    if batch is not None:
        out.update(batch_scalars(batch, n_cars))
    if learner is not None:
        out.update(grad_norms(learner))
    return out
