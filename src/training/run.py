"""Boucle de run : assemble simulateur, `BatchEnv`, réseaux, `Learner`, `Collector` et `OverlapTrainer`, puis itère
jusqu'au budget avec gardes, checkpoints, journal et reprise (T6, ADR-001 M4, ADR-003 §5–§7).

Garanties :
- `run` ne supprime ni n'écrase jamais lui-même un checkpoint, un journal ou une méta : les checkpoints sont écrits
  par `CheckpointManager` (noms jamais réutilisés, écriture atomique), `run_meta*.json` et `segments.jsonl` sont créés
  en mode exclusif ou en ajout. Deux exceptions, toutes deux sans perte :
  (1) la ROTATION de `CheckpointManager` supprime les plus anciens checkpoints au-delà de `checkpoint.keep`, hors
  dernier checkpoint sain intègre, qui est conservé ; (2) le journal `metrics.jsonl` est réécrit (remplacement
  atomique) pour retirer les lignes d'une trajectoire écartée (à la reprise ou après une restauration par une garde) :
  les lignes retirées sont ajoutées à `metrics_dropped.jsonl`, jamais supprimées ;
- un run neuf écrit un checkpoint sain à l'itération 0 (avant la première itération) : une divergence précoce se
  restaure au lieu de donner « aucun checkpoint sain » ;
- les gardes (`training.guards`) sont évaluées à chaque itération et leurs décisions sont APPLIQUÉES : CONTINUE ;
  STOP (arrêt propre, checkpoint final marqué non sain, cause écrite) ; RESTORE (dernier checkpoint sain, restauration
  de l'env et du trainer, arrêt s'il n'y en a pas ou si `max_restores` (cumulé) ou `max_consecutive_restores` est
  dépassé). Après une restauration : les générateurs de la collecte et du learner sont ré-ensemencés (seed, nombre de
  restaurations ; sans cela la trajectoire déterministe rejouerait la divergence), le journal et les compteurs de la
  trajectoire (buts, fins d'épisode) reviennent au checkpoint restauré, tandis que les compteurs de vie du run
  (restaurations, NaN, S1) ne reculent jamais ;
- une exception du learner ou de la collecte est tracée (événement `exception` dans `events.jsonl`, ligne de statut
  `error` dans `segments.jsonl`), le thread learner est joint, aucun checkpoint n'est écrit, puis l'exception remonte ;
- Ctrl+C (SIGINT) : l'itération en cours se termine, thread learner joint, checkpoint final, statut `interrupted` ;
  un second Ctrl+C lève `KeyboardInterrupt` immédiatement (pas de checkpoint, l'état serait incohérent) ;
- pool d'adversaires (section `pool`, ADR-005 D5, `training.selfplay`) : θ₀ entre dans le pool avant le checkpoint
  de l'itération 0, puis un instantané toutes les `pool.snapshot_every` itérations (avant le checkpoint de la même
  itération) ; l'état du pool voyage dans l'état du collecteur (checkpoint, reprise et restauration exactes). Le
  budget, `trained_timesteps` et `train/trained_timesteps` sont en timesteps APPRIS (agents gelés exclus, D3) ;
  `train/collected_timesteps` donne le collecté ;
- GCO strict : aucune récompense ajoutée, aucun signal d'exploration. L'état S1 vient d'un fournisseur injecté
  (`s1_provider`, défaut : jamais atteint, donc garde d'entropie active et règle « EV < 0 après S1 » non armée) ; son
  calcul (matchs d'évaluation) n'est pas fait ici.
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
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from env import BatchEnv, InitialStateSampler, RocketGPUSimulator, Simulator
from env.batch_env import ObsBuilder
from models import PolicyNet, ValueNet
from rewards import RewardFunction

if TYPE_CHECKING:
    from training.evalhook import EvalHook
import training.logging as training_logging
from training.checkpoint import (
    CheckpointIdentity,
    CheckpointManager,
    capture,
    git_commit,
    restore,
)
from training.config import P_KICK_SCHEDULE_VERSION, RunConfig
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
from training.logging import JsonlWriter, ThroughputMeter, batch_scalars, iteration_scalars, try_append_jsonl
from training.ppo import Learner, OptimizerLike, OverlapTrainer, make_optimizer, resolve_device
from training.rollout import Collector, RolloutBatch
from training.selfplay import PastOpponents, QualityPool, make_layout

METRICS_FILE = "metrics.jsonl"
EVENTS_FILE = "events.jsonl"
SEGMENTS_FILE = "segments.jsonl"
DROPPED_FILE = "metrics_dropped.jsonl"
RESTORE_RESEED_PRIME = 1_000_003
POOL_SEED_OFFSET = 7_919  # flux du tirage des adversaires distinct de celui de la collecte (même seed de run)


class RunError(RuntimeError):
    """Run impossible à démarrer ou à reprendre (dossier déjà utilisé, rien à reprendre, env incohérent)."""


EnvFactory = Callable[[RunConfig], BatchEnv]
OptimizerFactory = Callable[[list[torch.nn.Parameter], Any, torch.device], OptimizerLike]


@dataclass(frozen=True)
class RunSummary:
    """Résultat d'un appel à `run` (un « segment » : de `start_iteration` à `end_iteration`)."""

    status: str
    """`completed` (budget atteint), `paused` (`max_iterations` atteint), `interrupted` (Ctrl+C), `stopped` (garde),
    `error` (exception du learner ou de la collecte : écrit dans `segments.jsonl`, puis remontée par `run`)."""
    cause: str | None
    start_iteration: int
    end_iteration: int
    total_iterations: int
    trained_timesteps: int
    """Transitions agents APPRISES (entrées dans la perte, unité du budget, ADR-005 D3) : `end_iteration` ×
    `RunConfig.batch_learned_steps`. Avec le pool, inférieur au collecté (agents gelés exclus)."""
    collected_timesteps: int
    """Transitions agents collectées depuis le début du run (en recouvrement : un lot d'avance)."""
    segment_collected_timesteps: int
    """Transitions agents RÉELLEMENT collectées pendant ce segment, lots des trajectoires écartées par une restauration
    compris (somme des lots par itération ; `collected_timesteps` ne compte que la trajectoire retenue)."""
    wall_s: float
    loop_s: float
    """Durée des seules itérations (sans écriture de checkpoint ni restauration), y compris celles écartées."""
    mean_steps_per_s: float
    """`segment_collected_timesteps` / `loop_s` (démarrage et checkpoints exclus). Numérateur et dénominateur incluent
    les itérations écartées : le débit n'est pas biaisé par une restauration, mais le temps rejoué n'est pas du
    progrès (voir `discarded_timesteps`)."""
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
    discarded_timesteps: int = 0
    """Transitions agents collectées puis écartées par des restaurations dans ce segment."""
    s1_reached: bool = False
    s1_iteration: int | None = None
    """Itération (compteur de la trajectoire) à laquelle `s1_provider` a signalé S1 (None si jamais)."""
    error: dict[str, Any] | None = None
    """Statut `error` : type, message, itération et cause de l'exception remontée (None sinon)."""

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
        self,
        env: BatchEnv,
        policy: PolicyNet,
        n_steps: int,
        device: torch.device,
        seed: int,
        tally: _Tally,
        opponents: PastOpponents | None = None,
    ) -> None:
        super().__init__(env, policy, n_steps, device, seed, opponents=opponents)
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
    training_logging.append_jsonl(path, obj, fsync=True)


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
    s1_provider: Callable[[int], bool] | None = None,
    eval_hook: EvalHook | None = None,
) -> RunSummary:
    """Entraîne jusqu'au budget de `config`.

    `out_dir` : journaux (`metrics.jsonl`, `events.jsonl`, `segments.jsonl`, `run_meta*.json`) ; checkpoints dans
    `checkpoint_dir` (défaut `out_dir/checkpoints`). `resume=False` refuse un dossier déjà utilisé ; `resume=True`
    reprend du dernier checkpoint sain et refuse si l'identité ou le hash de config diffère
    (`CheckpointIncompatibleError`). `max_iterations` : plafond ABSOLU du compteur d'itérations (arrêt `paused`).
    `env_factory` et `optimizer_factory` sont injectables (tests sans RocketGPU ni `torch.optim`).

    `s1_provider(iteration) -> bool` : crochet du jalon S1 (`docs/PLAN.md` § kill criteria). Appelé une fois par
    itération (avec le compteur d'itérations après l'itération), tant que S1 n'est pas atteint ; le premier `True` fige
    l'état (monotone : il ne redevient jamais faux, même après une restauration vers un checkpoint antérieur), écrit un
    événement `s1_reached`, et l'état est sauvegardé dans les checkpoints (`extra["s1_reached"]`, `["s1_iteration"]`)
    puis restauré à la reprise. Il est transmis aux gardes : avant S1 la garde d'entropie est active, après S1 la règle
    d'explained variance s'arme. Défaut `None` : S1 jamais atteint. Une exception du fournisseur est traitée comme une
    exception de la boucle (tracée puis remontée).

    `eval_hook` (`training.evalhook.EvalHook`, A4) : évaluation en cours de run. Lié à la politique apprise, appelé à
    CHAQUE itération, y compris après S1 (journal D7, alerte K7) ; son verdict arme S1 comme `s1_provider` (les deux
    peuvent coexister, le premier vrai l'emporte). Son état voyage dans `extra["evalhook"]` des checkpoints (reprise et
    restauration par une garde le rechargent) ; chaque checkpoint écrit lui est notifié (`note_checkpoint`). Ses
    transitions par itération doivent valoir `batch_learned_steps` (pas d'évaluation en timesteps collectés).

    Double état S1 après une restauration (ou une reprise) vers un checkpoint antérieur au verrou : `s1_reached` de
    `run` est monotone et reste vrai (il gouverne les gardes), tandis que le hook, rechargé depuis le checkpoint, n'est
    plus armé ; il refait ses évaluations et peut se réarmer, d'où un événement `s1_reached` possible deux fois dans
    `evals.jsonl` (un par trajectoire), contre un seul dans `events.jsonl`. Le verdict officiel (S1-seed, hors ligne)
    part des instantanés de la trajectoire retenue.

    Alertes de garde (`GuardDecision.alerts`, K2 sous 50 % de ln 90) : événement `alert` en passant sous le seuil,
    `alert_cleared` à la sortie, rien entre les deux. Le calendrier de `p_kick` (version `P_KICK_SCHEDULE_VERSION`) est
    écrit dans les checkpoints ; une reprise d'un checkpoint d'une autre version est refusée.
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
    if eval_hook is not None and eval_hook.learned_steps_per_iteration != cfg.batch_learned_steps:
        raise RunError(
            f"eval_hook : {eval_hook.learned_steps_per_iteration} transitions par itération != "
            f"batch_learned_steps {cfg.batch_learned_steps} (le pas d'évaluation compte en timesteps appris)"
        )

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
    # pool d'adversaires (ADR-005 D5) : None sans section `pool` (self-play contre la version courante seule)
    pool: PastOpponents | None = None
    if cfg.pool is not None:
        pool = PastOpponents(
            make_layout(cfg.env.n_envs, cfg.env.n_cars, cfg.pool.past_fraction),
            QualityPool(cfg.pool.max_size, cfg.pool.quality_lr, cfg.seed + POOL_SEED_OFFSET),
            lambda: PolicyNet(env.obs_dim, env.n_actions, **p_kw),
            device,
        )
    collector = _TalliedCollector(env, actor, cfg.collection.steps_per_batch, device, cfg.seed, tally, pool)
    trainer = OverlapTrainer(learner, collector, overlap=cfg.collection.overlap)
    if eval_hook is not None:
        eval_hook.bind_policy(learner.policy)  # par noms de paramètres ; copie au moment de chaque évaluation

    # --- état initial : neuf ou repris
    guard_state = GuardState()
    iteration = 0
    commit = git_commit(Path(__file__).resolve().parents[2])
    start_note = ""
    resumed_from: str | None = None
    exact_env: bool | None = None
    s1_reached = False
    s1_iteration: int | None = None
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
        # checkpoints d'avant A1 : pas de `restores_total` ni d'état S1 dans les extras (valeurs par défaut)
        guard_state = dataclasses.replace(guard_state, restores_total=max(guard_state.restores_total, tally.restores))
        s1_reached = bool(info.extra.get("s1_reached", False))
        s1_iteration = info.extra.get("s1_iteration")
        saved_schedule = info.extra.get("p_kick_schedule_version")
        if saved_schedule is not None and saved_schedule != P_KICK_SCHEDULE_VERSION:
            raise RunError(
                f"calendrier de p_kick du checkpoint (version {saved_schedule}) != code ({P_KICK_SCHEDULE_VERSION})"
            )
        if eval_hook is not None and "evalhook" in info.extra:
            eval_hook.load_state_dict(info.extra["evalhook"])
            eval_hook.note_checkpoint(iteration, True)
        elif eval_hook is not None:
            raise RunError(
                f"checkpoint {path.name} sans état d'évaluation (run lancé sans hook) : reprise avec évaluation "
                f"refusée, θ₀ serait pris sur les poids de l'itération {iteration} et fausserait S1 (ADR-005 D2)"
            )
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
    raw_collected = 0  # transitions réellement collectées dans ce segment (lots écartés compris)
    discarded = 0  # dont transitions des trajectoires écartées par une restauration
    current_iteration = iteration + 1  # itération (1-based) en cours, pour tracer une exception
    failure: Exception | None = None
    error_info: dict[str, Any] | None = None
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
            "s1_reached": s1_reached,
            "s1_iteration": s1_iteration,
            "p_kick_schedule_version": P_KICK_SCHEDULE_VERSION,
        }
        if eval_hook is not None:
            extra["evalhook"] = eval_hook.state_dict()
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
        if eval_hook is not None:
            eval_hook.note_checkpoint(iteration, healthy)
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

    active_alerts: dict[str, str] = {}

    def log_alerts(decision: GuardDecision) -> None:
        """Une ligne à l'entrée dans une alerte et une à la sortie, pas une par itération (revue A4, M2)."""
        now = {a.split(" :", 1)[0]: a for a in decision.alerts}
        for rule, text in now.items():
            if rule not in active_alerts:
                _append_jsonl(
                    out / EVENTS_FILE,
                    {"event": "alert", "t": time.time(), "iteration": iteration, "rule": rule, "detail": text},
                )
        for rule in [r for r in active_alerts if r not in now]:
            _append_jsonl(
                out / EVENTS_FILE, {"event": "alert_cleared", "t": time.time(), "iteration": iteration, "rule": rule}
            )
        active_alerts.clear()
        active_alerts.update(now)

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
        if not resume:
            if pool is not None:
                pool.add_snapshot(learner.policy, 0)  # θ₀ : premier adversaire passé, avant toute collecte
            if eval_hook is not None:
                eval_hook.write_theta0()  # avant le checkpoint 0 : il contient θ₀ (restauration vers 0, reprise)
            save(True)  # checkpoint sain à l'itération 0 : une divergence précoce reste restaurable
        meter.tick(collector.agent_steps)
        while iteration < cap:
            if stop.is_set():
                status, cause = "interrupted", "arrêt demandé (Ctrl+C ou should_stop)"
                break
            t_it = time.perf_counter()
            current_iteration = iteration + 1
            env.set_p_kick(cfg.p_kick_at(iteration))
            steps_before = collector.agent_steps
            result = trainer.iterate()
            iteration += 1
            raw_collected += collector.agent_steps - steps_before
            rate = meter.tick(collector.agent_steps)
            loop_s += time.perf_counter() - t_it
            sps.append(rate)

            hook_s1 = eval_hook(iteration) if eval_hook is not None else False  # à chaque itération (D7, K7)
            if not s1_reached and (hook_s1 or (s1_provider is not None and bool(s1_provider(iteration)))):
                s1_reached, s1_iteration = True, iteration  # monotone : plus jamais remis à faux
                _append_jsonl(out / EVENTS_FILE, {"event": "s1_reached", "t": time.time(), "iteration": iteration})

            stats = {k: float(v) for k, v in result.stats.as_dict().items()}
            if iteration % cfg.logging.every_iterations == 0:
                scalars = iteration_scalars(
                    result, p_kick=env.p_kick, steps_per_s=rate, n_cars=env.n_cars, learner=learner
                )
                scalars.update(tally.last_scalars)
                scalars["env/goals_total"] = tally.goals
                scalars["train/iteration"] = float(iteration)
                scalars["train/restores_total"] = float(tally.restores)
                scalars["train/s1_reached"] = float(s1_reached)
                scalars["train/epoch"] = float(tally.restores)
                scalars["train/trained_timesteps"] = float(iteration * cfg.batch_learned_steps)  # appris (D3)
                scalars["train/collected_timesteps"] = float(collector.agent_steps)
                if pool is not None:
                    scalars.update(pool.scalars())
                writer.write(iteration, scalars)

            p_ok, g_ok = params_finite(learner.params), grads_finite(learner.params)
            guard_state, decision = evaluate(
                cfg.kill,
                guard_state,
                GuardObservation(stats=stats, params_finite=p_ok, grads_finite=g_ok, s1_reached=s1_reached),
            )
            last_healthy = decision.healthy
            log_alerts(decision)
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
                steps_before_restore = collector.agent_steps
                info = restore(state, trainer, expected=identity, train_config=traj, env=env)
                discarded += steps_before_restore - collector.agent_steps
                iteration = info.iteration
                last_ckpt_iteration = iteration
                # compteurs de la TRAJECTOIRE : retour à la valeur du checkpoint ; compteurs de VIE du run
                # (restaurations, NaN, S1) : conservés, ils ne reculent jamais
                tally.goals, tally.episode_ends = float(info.extra["goals"]), float(info.extra["episode_ends"])
                if eval_hook is not None and "evalhook" in info.extra:
                    eval_hook.load_state_dict(info.extra["evalhook"])  # évaluations écartées avec la trajectoire
                    eval_hook.note_checkpoint(iteration, True)
                # le journal ne garde qu'une ligne par step de la trajectoire retenue ; les autres vont dans
                # `metrics_dropped.jsonl`. Le fichier est fermé le temps du remplacement atomique (Windows).
                writer.close()
                n_dropped = truncate_metrics(metrics_path, iteration, out / DROPPED_FILE)
                writer = JsonlWriter(metrics_path, fsync=cfg.logging.fsync)
                seed = cfg.seed + RESTORE_RESEED_PRIME * tally.restores
                collector.generator.manual_seed(seed)
                learner.generator.manual_seed(seed)
                if pool is not None:
                    pool.reseed(seed + POOL_SEED_OFFSET)
                _append_jsonl(
                    out / EVENTS_FILE,
                    {
                        "event": "restore",
                        "t": time.time(),
                        "from": path.name,
                        "to_iteration": iteration,
                        "reseed": seed,
                        "exact_env": info.exact_env,
                        "dropped_rows": n_dropped,
                        "discarded_timesteps": steps_before_restore - collector.agent_steps,
                    },
                )
                meter.tick(collector.agent_steps)
                continue
            if (
                pool is not None
                and cfg.pool is not None
                and iteration % cfg.pool.snapshot_every == 0
                and decision.healthy  # une itération non saine (KL haut) n'entre pas dans le pool à q = max
            ):
                pool.add_snapshot(learner.policy, iteration)  # avant le checkpoint de la même itération
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
    except Exception as exc:  # learner, collecte, fournisseur S1, E/S : tracer puis remonter, sans checkpoint
        failure = exc
        trainer.close()  # joint le thread learner avant toute écriture
        cause_exc = exc.__cause__
        error_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "iteration": current_iteration,
            "cause_type": type(cause_exc).__name__ if cause_exc is not None else None,
            "cause_message": str(cause_exc) if cause_exc is not None else None,
        }
        status, cause = "error", f"{type(exc).__name__} : {exc}"
        # jamais d'exception ici : une erreur d'écriture masquerait l'exception d'origine (revue A4, M2)
        try_append_jsonl(out / EVENTS_FILE, {"event": "exception", "t": time.time(), **error_info})
    finally:
        trainer.close()
        writer.close()
        if sigint_installed:
            signal.signal(signal.SIGINT, previous_handler)

    wall = time.perf_counter() - t_start
    seg_steps = raw_collected
    summary = RunSummary(
        status=status,
        cause=cause,
        start_iteration=start_iteration,
        end_iteration=iteration,
        total_iterations=total,
        trained_timesteps=iteration * cfg.batch_learned_steps,
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
        discarded_timesteps=discarded,
        s1_reached=s1_reached,
        s1_iteration=s1_iteration,
        error=error_info,
    )
    row = summary.as_dict()
    row["steps_per_s"] = [x if math.isfinite(x) else None for x in sps]
    if failure is not None:
        try_append_jsonl(out / SEGMENTS_FILE, {"t": time.time(), **row})
        raise failure
    _append_jsonl(out / SEGMENTS_FILE, {"t": time.time(), **row})
    return summary
