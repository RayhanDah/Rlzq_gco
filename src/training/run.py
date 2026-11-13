"""Boucle de run : assemble simulateur, `BatchEnv`, réseaux, `Learner`, `Collector` et `OverlapTrainer`, puis itère
jusqu'au budget avec gardes, checkpoints, journal et reprise (T6, ADR-001 M4, ADR-003 §5–§7).

Garanties :
- aucun checkpoint, journal ou méta n'est supprimé ni écrasé : les checkpoints suivent `CheckpointManager` (noms
  jamais réutilisés), `run_meta*.json` et `segments.jsonl` sont créés en mode exclusif ou en ajout ; la seule réécriture
  est la troncature du journal de métriques à la reprise (les lignes retirées vont dans `metrics_dropped.jsonl`) ;
- les gardes (`training.guards`) sont évaluées à chaque itération et leurs décisions sont APPLIQUÉES : CONTINUE ;
  STOP (arrêt propre, checkpoint final marqué non sain, cause écrite) ; RESTORE (dernier checkpoint sain, restauration
  de l'env et du trainer, arrêt s'il n'y en a pas ou si `max_restores` est dépassé). Après une restauration, les
  générateurs de la collecte et du learner sont ré-ensemencés (seed, nombre de restaurations) : sans cela la trajectoire
  déterministe rejouerait exactement la divergence ;
- Ctrl+C (SIGINT) : l'itération en cours se termine, thread learner joint, checkpoint final, statut `interrupted` ;
  un second Ctrl+C lève `KeyboardInterrupt` immédiatement (pas de checkpoint, l'état serait incohérent) ;
- GCO strict : aucune récompense ajoutée, aucun signal d'exploration. L'état S1 n'est pas détecté ici : la garde
  d'entropie reste active (`s1_reached=False`) et la règle « EV < 0 après S1 » n'est jamais armée (hors sprint).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import random
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import torch

from env import BatchEnv, InitialStateSampler, RocketGPUSimulator, Simulator
from env.batch_env import ObsBuilder
from models import PolicyNet, ValueNet
from rewards import RewardFunction
from training.checkpoint import (
    CheckpointIdentity,
    CheckpointManager,
    capture,
    git_commit,
    restore,
)
from training.config import RunConfig
from training.guards import (
    Action,
    GuardDecision,
    GuardObservation,
    GuardState,
    evaluate,
    grads_finite,
    nonfinite_keys,
    params_finite,
)
from training.logging import JsonlWriter, ThroughputMeter, batch_scalars, iteration_scalars
from training.ppo import Learner, OptimizerLike, OverlapTrainer, make_optimizer, resolve_device
from training.rollout import Collector, RolloutBatch

METRICS_FILE = "metrics.jsonl"
EVENTS_FILE = "events.jsonl"
SEGMENTS_FILE = "segments.jsonl"
DROPPED_FILE = "metrics_dropped.jsonl"
RESTORE_RESEED_PRIME = 1_000_003


class RunError(RuntimeError):
    """Run impossible à démarrer ou à reprendre (dossier déjà utilisé, rien à reprendre, env incohérent)."""


EnvFactory = Callable[[RunConfig], BatchEnv]
OptimizerFactory = Callable[[list[torch.nn.Parameter], Any, torch.device], OptimizerLike]


@dataclass(frozen=True)
class RunSummary:
    """Résultat d'un appel à `run` (un « segment » : de `start_iteration` à `end_iteration`)."""

    status: str
    """`completed` (budget atteint), `paused` (`max_iterations` atteint), `interrupted` (Ctrl+C), `stopped` (garde)."""
    cause: str | None
    start_iteration: int
    end_iteration: int
    total_iterations: int
    trained_timesteps: int
    """Transitions agents apprises : `end_iteration` × transitions par lot."""
    collected_timesteps: int
    """Transitions agents collectées depuis le début du run (en recouvrement : un lot d'avance)."""
    segment_collected_timesteps: int
    wall_s: float
    loop_s: float
    """Durée des seules itérations (sans écriture de checkpoint)."""
    mean_steps_per_s: float
    """Transitions agents collectées pendant ce segment / `loop_s` (démarrage et checkpoints exclus)."""
    steps_per_s: list[float]
    """Débit de chaque itération du segment (la première itération d'un run neuf collecte deux lots)."""
    nan_detected: int
    restores: int
    decisions: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]
    goals_total: float
    """Buts (fins terminales / voitures) sur tous les lots collectés depuis le début du run."""
    p_kick_final: float
    out_dir: str
    checkpoint_dir: str
    resumed_from: str | None = None
    """Fichier de checkpoint repris (None pour un run neuf)."""
    exact_env: bool | None = None
    """Reprise : True si l'état du simulateur a été restauré bit à bit, False « modulo reset des arènes »."""

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class _Tally:
    """Compteurs cumulés sur les lots collectés, sauvegardés dans `extra` du checkpoint."""

    goals: float = 0.0
    episode_ends: float = 0.0
    restores: int = 0
    nan: int = 0
    last_scalars: dict[str, float] = field(default_factory=dict)


class _TalliedCollector(Collector):
    """`Collector` qui compte buts et fins d'épisode de chaque lot collecté (le premier lot compris)."""

    def __init__(
        self, env: BatchEnv, policy: PolicyNet, n_steps: int, device: torch.device, seed: int, tally: _Tally
    ) -> None:
        super().__init__(env, policy, n_steps, device, seed)
        self.tally = tally
        self._cars = env.n_cars

    def collect(self, policy_version: int) -> RolloutBatch:
        batch = super().collect(policy_version)
        sc = batch_scalars(batch, self._cars)
        self.tally.goals += sc["env/goals"]
        self.tally.episode_ends += sc["env/episode_ends"]
        self.tally.last_scalars = sc
        return batch


# -------------------------------------------------------------------------------------------------- assemblage


def make_env(
    cfg: RunConfig, sim: Simulator, *, build_obs: ObsBuilder | None = None, obs_dim: int | None = None
) -> BatchEnv:
    """`BatchEnv` de la config sur le simulateur `sim` (récompense de la variante, tirage seedé, `p_kick` de départ)."""
    reward = RewardFunction(cfg.reward.build())
    sampler = InitialStateSampler(cfg.env.n_cars, cfg.p_kick_at(0), cfg.seed)
    return BatchEnv(sim, reward, sampler, build_obs=build_obs, obs_dim=obs_dim)


def default_env_factory(cfg: RunConfig) -> BatchEnv:
    """RocketGPU CPU (variables `ROCKETGPU_PY_DIR`, `ROCKETGPU_COLLISION_MESH_DIR`, voir `env.sim_adapter`)."""
    sim = RocketGPUSimulator(cfg.env.n_envs, cfg.env.n_cars, cfg.env.tick_skip)
    return make_env(cfg, sim)


def _default_optimizer(params: list[torch.nn.Parameter], ppo: Any, device: torch.device) -> OptimizerLike:
    return make_optimizer(params, ppo, device)


def rocketgpu_commit() -> str:
    """SHA du build RocketGPU lu dans `build_info.json` (dossier parent de `ROCKETGPU_PY_DIR`), sinon "unknown"."""
    py_dir = os.environ.get("ROCKETGPU_PY_DIR", "")
    if not py_dir:
        return "unknown"
    info = Path(py_dir).parent / "build_info.json"
    try:
        sha = json.loads(info.read_text(encoding="utf-8")).get("sha")
    except (OSError, ValueError):
        return "unknown"
    return str(sha) if sha else "unknown"


# -------------------------------------------------------------------------------------------------- fichiers


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    line = json.dumps(obj, sort_keys=True, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _write_new_json(path: Path, obj: dict[str, Any]) -> Path:
    """Crée `path` en mode exclusif (jamais d'écrasement), au besoin suffixé `_001`... ; rend le chemin écrit."""
    candidate, n = path, 0
    while True:
        try:
            with open(candidate, "x", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, sort_keys=True, default=str)
                f.write("\n")
            return candidate
        except FileExistsError:
            n += 1
            candidate = path.with_name(f"{path.stem}_{n:03d}{path.suffix}")


def truncate_metrics(path: Path, keep_through_step: int, dropped_path: Path) -> int:
    """Retire de `path` les lignes de step > `keep_through_step` (journal d'un segment qui a dépassé le checkpoint
    repris). Les lignes retirées sont ajoutées à `dropped_path` : rien n'est perdu. Réécriture atomique (temporaire +
    remplacement) ; le fichier n'est jamais supprimé. Rend le nombre de lignes retirées."""
    if not path.exists():
        return 0
    kept: list[str] = []
    dropped: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            step = int(json.loads(line)["step"])
        except (ValueError, KeyError, TypeError):
            kept.append(line)  # ligne illisible : on ne la détruit pas
            continue
        (kept if step <= keep_through_step else dropped).append(line)
    if not dropped:
        return 0
    with open(dropped_path, "a", encoding="utf-8") as f:
        f.write("\n".join(dropped) + "\n")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    os.replace(tmp, path)
    return len(dropped)


# -------------------------------------------------------------------------------------------------- boucle


class _StopSignal:
    """Drapeau d'arrêt propre : posé par SIGINT (1er Ctrl+C) ou par `should_stop`."""

    def __init__(self, should_stop: Callable[[], bool] | None) -> None:
        self._event = threading.Event()
        self._should_stop = should_stop

    def request(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set() or (self._should_stop is not None and self._should_stop())

    def handler(self, _signum: int, _frame: FrameType | None) -> None:
        if self._event.is_set():
            raise KeyboardInterrupt  # second Ctrl+C : on sort sans attendre
        self._event.set()


def _meta(
    cfg: RunConfig,
    device: torch.device,
    reward_hash: str,
    start_iteration: int,
    command: list[str] | None,
    resume: bool,
    commit: str,
) -> dict[str, Any]:
    return {
        "name": cfg.name,
        "resume": resume,
        "start_iteration": start_iteration,
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": commit,
        "rocketgpu_commit": rocketgpu_commit(),
        "config_hash": cfg.trajectory_hash(),
        "config_full_hash": cfg.full_hash(),
        "reward_config_hash": reward_hash,
        "device": str(device),
        "device_requested": cfg.device,
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "seed": cfg.seed,
        "command": command if command is not None else list(sys.argv),
        "config": cfg.to_dict(),
    }


def run(
    config: RunConfig,
    out_dir: str | Path,
    *,
    resume: bool = False,
    max_iterations: int | None = None,
    checkpoint_dir: str | Path | None = None,
    env_factory: EnvFactory | None = None,
    optimizer_factory: OptimizerFactory | None = None,
    should_stop: Callable[[], bool] | None = None,
    handle_sigint: bool = True,
    command: list[str] | None = None,
) -> RunSummary:
    """Entraîne jusqu'au budget de `config`.

    `out_dir` : journaux (`metrics.jsonl`, `events.jsonl`, `segments.jsonl`, `run_meta*.json`) ; checkpoints dans
    `checkpoint_dir` (défaut `out_dir/checkpoints`). `resume=False` refuse un dossier déjà utilisé ; `resume=True`
    reprend du dernier checkpoint sain et refuse si l'identité ou le hash de config diffère
    (`CheckpointIncompatibleError`). `max_iterations` : plafond ABSOLU du compteur d'itérations (arrêt `paused`).
    `env_factory` et `optimizer_factory` sont injectables (tests sans RocketGPU ni `torch.optim`).
    """
    cfg = config
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck_dir = Path(checkpoint_dir) if checkpoint_dir is not None else out / "checkpoints"
    manager = CheckpointManager(ck_dir, keep=cfg.checkpoint.keep)
    metrics_path = out / METRICS_FILE
    existing = manager.checkpoints()
    if resume and not existing:
        raise RunError(f"reprise demandée mais aucun checkpoint dans {ck_dir}")
    if not resume and (existing or (metrics_path.exists() and metrics_path.stat().st_size > 0)):
        raise RunError(f"{out} ou {ck_dir} déjà utilisé : relancer avec resume=True (rien n'est jamais écrasé)")

    # --- assemblage
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)  # noqa: NPY002 (état global sauvegardé par les checkpoints)
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    env = (env_factory or default_env_factory)(cfg)
    if (env.n_envs, env.n_cars, env.tick_skip) != (cfg.env.n_envs, cfg.env.n_cars, cfg.env.tick_skip):
        raise RunError(
            f"env fourni {env.n_envs}x{env.n_cars} tick_skip {env.tick_skip} != config "
            f"{cfg.env.n_envs}x{cfg.env.n_cars} tick_skip {cfg.env.tick_skip}"
        )
    reward_hash = RewardFunction(cfg.reward.build()).config_hash
    identity = CheckpointIdentity.current(reward_config_hash=reward_hash, obs_dim=env.obs_dim, tick_skip=env.tick_skip)
    traj = cfg.trajectory_dict()

    m = cfg.model
    p_kw: dict[str, Any] = {} if m.policy_hidden is None else {"hidden": m.policy_hidden}
    v_kw: dict[str, Any] = {} if m.value_hidden is None else {"hidden": m.value_hidden}
    policy = PolicyNet(env.obs_dim, env.n_actions, **p_kw)
    value = ValueNet(env.obs_dim, **v_kw)
    actor = PolicyNet(env.obs_dim, env.n_actions, **p_kw).to(device)
    policy.to(device)  # avant l'optimiseur : il doit tenir les paramètres définitifs
    value.to(device)
    optimizer = (optimizer_factory or _default_optimizer)([*policy.parameters(), *value.parameters()], cfg.ppo, device)
    learner = Learner(policy, value, cfg.ppo, device, optimizer=optimizer, seed=cfg.seed)
    tally = _Tally()
    collector = _TalliedCollector(env, actor, cfg.collection.steps_per_batch, device, cfg.seed, tally)
    trainer = OverlapTrainer(learner, collector, overlap=cfg.collection.overlap)

    # --- état initial : neuf ou repris
    guard_state = GuardState()
    iteration = 0
    commit = git_commit(Path(__file__).resolve().parents[2])
    start_note = ""
    resumed_from: str | None = None
    exact_env: bool | None = None
    if resume:
        found = manager.load_latest(healthy_only=True)
        if found is None:
            raise RunError(f"aucun checkpoint sain et intègre à reprendre dans {ck_dir}")
        path, state = found
        info = restore(
            state, trainer, expected=identity, train_config=traj, env=env
        )  # refuse si hash/identité diffèrent
        iteration = info.iteration
        guard_state = GuardState(**info.extra["guard"])
        tally.goals, tally.episode_ends = float(info.extra["goals"]), float(info.extra["episode_ends"])
        tally.restores, tally.nan = int(info.extra["restores"]), int(info.extra["nan"])
        resumed_from, exact_env = path.name, info.exact_env
        n_dropped = truncate_metrics(metrics_path, iteration, out / DROPPED_FILE)
        start_note = (
            f"reprise depuis {path.name} (itération {iteration}, exact_env={info.exact_env}, "
            f"lignes retirées={n_dropped})"
        )
    meta_path = _write_new_json(
        out / "run_meta.json", _meta(cfg, device, reward_hash, iteration, command, resume, commit)
    )

    writer = JsonlWriter(metrics_path, fsync=cfg.logging.fsync)
    stop = _StopSignal(should_stop)
    previous_handler: Any = None
    sigint_installed = False
    if handle_sigint and threading.current_thread() is threading.main_thread():
        previous_handler = signal.signal(signal.SIGINT, stop.handler)
        sigint_installed = True

    total = cfg.total_iterations
    cap = total if max_iterations is None else min(total, max_iterations)
    start_iteration = iteration
    start_collected = collector.agent_steps
    t_start = time.perf_counter()
    loop_s = 0.0
    sps: list[float] = []
    decisions: list[dict[str, Any]] = []
    ckpts: list[dict[str, Any]] = []
    status, cause = "paused", None
    last_ckpt_iteration = iteration if resume else -1
    last_healthy = True
    meter = ThroughputMeter()

    def save(healthy: bool) -> None:
        nonlocal last_ckpt_iteration
        t0 = time.perf_counter()
        extra = {
            "p_kick": env.p_kick,
            "guard": dataclasses.asdict(guard_state),
            "goals": tally.goals,
            "episode_ends": tally.episode_ends,
            "restores": tally.restores,
            "nan": tally.nan,
        }
        state = capture(
            trainer,
            identity=identity,
            train_config=traj,
            iteration=iteration,
            env_state=env.state_dict(include_sim=cfg.checkpoint.include_sim_state),
            extra=extra,
            git_commit_id=commit,
            rocketgpu_commit=rocketgpu_commit(),
        )
        t1 = time.perf_counter()
        path = manager.save(state, healthy=healthy)
        t2 = time.perf_counter()
        last_ckpt_iteration = iteration
        rec = {
            "iteration": iteration,
            "file": path.name,
            "bytes": path.stat().st_size,
            "capture_s": t1 - t0,
            "write_s": t2 - t1,
            "healthy": healthy,
        }
        ckpts.append(rec)
        _append_jsonl(out / EVENTS_FILE, {"event": "checkpoint", "t": time.time(), **rec})

    def log_decision(decision: GuardDecision) -> None:
        rec = {
            "iteration": iteration,
            "action": decision.action.value,
            "rule": decision.rule,
            "reason": decision.reason,
        }
        decisions.append(rec)
        _append_jsonl(out / EVENTS_FILE, {"event": "guard", "t": time.time(), **rec})

    try:
        _append_jsonl(
            out / EVENTS_FILE,
            {"event": "start", "t": time.time(), "meta_file": meta_path.name, "note": start_note, "resume": resume},
        )
        meter.tick(collector.agent_steps)
        while iteration < cap:
            if stop.is_set():
                status, cause = "interrupted", "arrêt demandé (Ctrl+C ou should_stop)"
                break
            t_it = time.perf_counter()
            env.set_p_kick(cfg.p_kick_at(iteration))
            result = trainer.iterate()
            iteration += 1
            rate = meter.tick(collector.agent_steps)
            loop_s += time.perf_counter() - t_it
            sps.append(rate)

            stats = {k: float(v) for k, v in result.stats.as_dict().items()}
            if iteration % cfg.logging.every_iterations == 0:
                scalars = iteration_scalars(
                    result, p_kick=env.p_kick, steps_per_s=rate, n_cars=env.n_cars, learner=learner
                )
                scalars.update(tally.last_scalars)
                scalars["env/goals_total"] = tally.goals
                scalars["train/iteration"] = float(iteration)
                scalars["train/restores_total"] = float(tally.restores)
                scalars["train/epoch"] = float(tally.restores)
                scalars["train/trained_timesteps"] = float(iteration * cfg.batch_agent_steps)
                writer.write(iteration, scalars)

            p_ok, g_ok = params_finite(learner.params), grads_finite(learner.params)
            guard_state, decision = evaluate(
                cfg.kill, guard_state, GuardObservation(stats=stats, params_finite=p_ok, grads_finite=g_ok)
            )
            last_healthy = decision.healthy
            if nonfinite_keys(stats) or not p_ok or not g_ok:  # compté même si la décision est STOP (max_restores)
                tally.nan += 1
            if decision.action is Action.STOP:
                log_decision(decision)
                status, cause = "stopped", f"{decision.rule} : {decision.reason}"
                break
            if decision.action is Action.RESTORE:
                log_decision(decision)
                trainer.close()
                found = manager.load_latest(healthy_only=True)
                if found is None:
                    status, cause = (
                        "stopped",
                        f"{decision.rule} : {decision.reason} ; aucun checkpoint sain à restaurer",
                    )
                    last_healthy = False
                    break
                path, state = found
                tally.restores += 1
                info = restore(state, trainer, expected=identity, train_config=traj, env=env)
                iteration = info.iteration
                last_ckpt_iteration = iteration
                seed = cfg.seed + RESTORE_RESEED_PRIME * tally.restores
                collector.generator.manual_seed(seed)
                learner.generator.manual_seed(seed)
                _append_jsonl(
                    out / EVENTS_FILE,
                    {
                        "event": "restore",
                        "t": time.time(),
                        "from": path.name,
                        "to_iteration": iteration,
                        "reseed": seed,
                        "exact_env": info.exact_env,
                    },
                )
                meter.tick(collector.agent_steps)
                continue
            if iteration % cfg.checkpoint.every_iterations == 0 and iteration < cap:
                save(decision.healthy)
            meter.tick(collector.agent_steps)  # rebase : l'écriture du checkpoint n'entre pas dans le débit
        else:
            status = "completed" if iteration >= total else "paused"
        if status == "stopped" and last_ckpt_iteration != iteration:
            try:
                save(False)
            except Exception as exc:  # l'arrêt doit rester propre même si la capture échoue (état cassé)
                _append_jsonl(out / EVENTS_FILE, {"event": "final_checkpoint_failed", "error": repr(exc)})
        elif status != "stopped" and last_ckpt_iteration != iteration:
            save(last_healthy)
    finally:
        trainer.close()
        writer.close()
        if sigint_installed:
            signal.signal(signal.SIGINT, previous_handler)

    wall = time.perf_counter() - t_start
    seg_steps = collector.agent_steps - start_collected
    summary = RunSummary(
        status=status,
        cause=cause,
        start_iteration=start_iteration,
        end_iteration=iteration,
        total_iterations=total,
        trained_timesteps=iteration * cfg.batch_agent_steps,
        collected_timesteps=collector.agent_steps,
        segment_collected_timesteps=seg_steps,
        wall_s=wall,
        loop_s=loop_s,
        mean_steps_per_s=(seg_steps / loop_s) if loop_s > 0 else float("nan"),
        steps_per_s=sps,
        nan_detected=tally.nan,
        restores=tally.restores,
        decisions=decisions,
        checkpoints=ckpts,
        goals_total=tally.goals,
        p_kick_final=env.p_kick,
        out_dir=str(out),
        checkpoint_dir=str(ck_dir),
        resumed_from=resumed_from,
        exact_env=exact_env,
    )
    row = summary.as_dict()
    row["steps_per_s"] = [x if math.isfinite(x) else None for x in sps]
    _append_jsonl(out / SEGMENTS_FILE, {"t": time.time(), **row})
    return summary
