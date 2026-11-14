"""Évaluation en cours de run (A4) : « S1-léger » de l'ADR-005 D2, qui alimente `s1_reached` de `training.run`.

Toutes les `every_timesteps` transitions apprises (0,05B par défaut, ≈ 20 évaluations par run de 1B), la politique
courante joue, sur un simulateur DÉDIÉ (l'env d'entraînement n'est ni utilisé ni modifié) :
- `episodes_random` épisodes kickoff contre l'agent aléatoire (uniforme sur les 90 actions) ;
- `episodes_frozen` épisodes kickoff contre elle-même gelée `frozen_lag_timesteps` plus tôt (θ₀, la politique
  initiale, tant que ce passé n'existe pas), côtés échangés par paires (`eval.match.play_match`).
Une évaluation RÉUSSIT si (a) contre l'aléatoire, la borne basse de Wilson unilatérale (1 - `wilson_alpha`) de
victoires / parties décisives dépasse 0,5 avec au moins `min_decisive` décisives (`eval.decisive_win_test` ;
« indéterminé » = échec) ET (b) la borne basse de Garwood (bilatérale par défaut, niveau 1 - `garwood_alpha`) des buts
marqués par minute contre le gelé dépasse `r0_goals_per_min` (R0, provisoire 0,032 but/min par camp).
S1 est atteint quand `consecutive` (3) évaluations de suite réussissent ; il est alors irréversible pour la trajectoire.

Ne pas compter une évaluation faite sur un état ensuite écarté (revue A1 vers A4) :
- RETOUR EN ARRIÈRE : un appel avec une itération <= celle d'une évaluation retenue signifie que la trajectoire a été
  ramenée en arrière (restauration par une garde, ou reprise depuis un checkpoint antérieur) ; les évaluations faites
  à partir de cette itération sont retirées de l'historique (et le verrou S1 s'il est postérieur), un événement
  `rollback` est journalisé. Aucune ligne n'est effacée des journaux.
- CONFIRMATION : la 3e réussite n'arme S1 qu'une fois couverte par un checkpoint SAIN d'itération >= la sienne. Une
  restauration repart toujours du dernier checkpoint sain : une évaluation ainsi couverte ne peut plus être écartée.
  Source de l'itération du dernier checkpoint sain : `healthy_probe` (lecture des en-têtes du dossier de checkpoints,
  pont actuel, voir `latest_healthy_iteration`) ou `note_checkpoint` (appelé par `run` après A3). Sans l'un ni
  l'autre, S1 n'est jamais armé. Délai : au plus `checkpoint.every_iterations` itérations.

État (`state_dict` / `load_state_dict`, JSON pur) : config, historique des évaluations retenues, verrou S1, θ₀,
compteur de retours en arrière. Il est AJOUTÉ à `evalhook_state.jsonl` à chaque changement (jamais réécrit ; la
dernière ligne fait foi) ; `run` pourra aussi le mettre dans `extra` des checkpoints (après A3).
Instantanés d'évaluation : poids de politique seuls, dans `snapshot_dir`, au format `CheckpointManager` (relisibles par
`scripts/eval.py --a <fichier>`), sans rotation (jamais supprimés). Journal par évaluation (ADR-005 D7) :
`evals.jsonl` (score contre l'aléatoire avec Wilson, buts/min pour et contre le gelé avec Garwood, R0 rappelé, durées).

Aucun générateur global n'est consommé (politiques, kickoffs et agent aléatoire ont leurs générateurs dédiés) : la
trajectoire d'entraînement est identique avec ou sans évaluation (testé sur simulateur factice). GCO strict : rien de
ce module n'entre dans la perte ni dans la récompense.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from env import Simulator
from env.batch_env import ObsBuilder
from eval.match import TICKS_PER_MINUTE, MatchResult, NetPolicy, RandomPolicy, play_match
from eval.rating import garwood_interval
from eval.sprt import DecisiveVerdict, decisive_win_test
from models import PolicyNet
from rewards import RewardConfig
from training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointCorruptError,
    CheckpointIdentity,
    CheckpointManager,
    read_meta,
)
from training.logging import append_jsonl

STATE_VERSION = 1
EVALS_FILE = "evals.jsonl"
STATE_FILE = "evalhook_state.jsonl"
SNAPSHOT_KEEP = 1_000_000_000
"""`keep` du gestionnaire d'instantanés : aucune rotation (les instantanés d'évaluation ne sont jamais supprimés)."""

_ROLE_RANDOM_MATCH, _ROLE_RANDOM_AGENT, _ROLE_CURRENT_VS_RANDOM = 0, 1, 2
_ROLE_FROZEN_MATCH, _ROLE_CURRENT_VS_FROZEN, _ROLE_FROZEN_AGENT = 3, 4, 5


class EvalHookError(RuntimeError):
    """Hook mal configuré, état incompatible ou instantané introuvable."""


@dataclass(frozen=True)
class EvalHookConfig:
    """Paramètres de S1-léger. Valeurs par défaut : ADR-005 D2 (accepté le 2026-11-13)."""

    every_timesteps: int = 50_000_000
    """Une évaluation chaque fois que les timesteps appris franchissent un multiple de cette valeur."""
    frozen_lag_timesteps: int = 100_000_000
    """Retard du gelé ; multiple de `every_timesteps` (le gelé est l'instantané de l'évaluation k - lag/every)."""
    episodes_random: int = 512
    episodes_frozen: int = 512
    min_decisive: int = 20
    wilson_alpha: float = 0.05
    """Wilson UNILATÉRAL (borne basse de niveau 1 - alpha), comme `eval.decisive_win_test`."""
    garwood_alpha: float = 0.05
    garwood_two_sided: bool = True
    """« Garwood 95 % » : IC bilatéral par défaut (borne basse au quantile alpha/2, plus prudente que l'unilatérale)."""
    r0_goals_per_min: float = 0.032
    """R0 : borne haute des buts/min PAR CAMP de l'aléatoire contre l'aléatoire (ADR-005 P0, provisoire)."""
    consecutive: int = 3
    k7_consecutive: int = 3
    """K7 (alerte) : après S1, borne haute de Wilson (décisives contre l'aléatoire) < 0,5 sur autant d'évaluations."""
    n_envs: int = 256
    """Arènes du simulateur d'évaluation (le kickoff dépend de l'arène : changer cette valeur change l'échantillon).
    256 : mesuré le plus rapide pour 512 + 512 épisodes (28,7 s contre 40,9 s à 64 et 46,1 s à 512 arènes ; politique
    initiale 3x256, CPU, RocketGPU @f0508a1, laptop Core Ultra 5 125H, 2026-11-14, arbre fd2ec74 + modifications A4)."""
    mode: str = "sample"
    """"sample" (comme à l'entraînement) ou "argmax"."""
    seed: int | None = None
    """Graine de l'évaluation ; None = graine du run."""

    def __post_init__(self) -> None:
        bad: list[str] = []
        if self.every_timesteps < 1:
            bad.append("every_timesteps >= 1")
        if self.frozen_lag_timesteps < self.every_timesteps or self.frozen_lag_timesteps % max(self.every_timesteps, 1):
            bad.append("frozen_lag_timesteps multiple (>= 1 fois) de every_timesteps")
        if min(self.episodes_random, self.episodes_frozen) < 2:
            bad.append("episodes_* >= 2 (au moins une paire)")
        if self.min_decisive < 1 or self.consecutive < 1 or self.k7_consecutive < 1:
            bad.append("min_decisive, consecutive, k7_consecutive >= 1")
        if not (0.0 < self.wilson_alpha < 1.0 and 0.0 < self.garwood_alpha < 1.0):
            bad.append("alphas dans ]0, 1[")
        if not (self.r0_goals_per_min >= 0.0 and math.isfinite(self.r0_goals_per_min)):
            bad.append("r0_goals_per_min fini >= 0")
        if self.n_envs < 1:
            bad.append("n_envs >= 1")
        if self.mode not in ("sample", "argmax"):
            bad.append("mode : 'sample' ou 'argmax'")
        if self.seed is not None and self.seed < 0:
            bad.append("seed >= 0")
        if bad:
            raise ValueError(f"EvalHookConfig invalide : {', '.join(bad)}")

    @property
    def lag_evals(self) -> int:
        return self.frozen_lag_timesteps // self.every_timesteps

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalHookConfig:
        """Clés inconnues et types faux refusés (un booléen n'est pas un entier)."""
        if not isinstance(data, Mapping):
            raise ValueError("configuration d'évaluation : mapping attendu")
        known = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ValueError(f"évaluation : clés inconnues {unknown} (connues : {sorted(known)})")
        ints = {"every_timesteps", "frozen_lag_timesteps", "episodes_random", "episodes_frozen", "min_decisive"}
        ints |= {"consecutive", "k7_consecutive", "n_envs"}
        floats = {"wilson_alpha", "garwood_alpha", "r0_goals_per_min"}
        for k, v in data.items():
            ok = True
            if k in ints or (k == "seed" and v is not None):
                ok = isinstance(v, int) and not isinstance(v, bool)
            elif k in floats:
                ok = isinstance(v, int | float) and not isinstance(v, bool)
            elif k == "garwood_two_sided":
                ok = isinstance(v, bool)
            elif k == "mode":
                ok = isinstance(v, str)
            if not ok:
                raise ValueError(f"évaluation.{k} : type invalide ({type(v).__name__})")
        kwargs: dict[str, Any] = {k: (float(v) if k in floats else v) for k, v in data.items()}
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalHookConfig:
        import yaml  # PyYAML : déjà exigé par `RunConfig.from_yaml`

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data or {})


def latest_healthy_iteration(checkpoint_dir: str | Path) -> int | None:
    """Itération du plus récent checkpoint marqué sain dans `checkpoint_dir` (en-têtes seuls, sans charger les poids ;
    fichiers illisibles sautés), None s'il n'y en a pas. Lecture seule. Pont en attendant `note_checkpoint` dans
    `run` : `run` restaure toujours le plus récent checkpoint sain, donc rien d'antérieur à cette itération n'est
    plus jamais écarté."""
    d = Path(checkpoint_dir)
    if not d.is_dir():
        return None
    for ref in reversed(CheckpointManager(d, keep=SNAPSHOT_KEEP).checkpoints()):
        try:
            if read_meta(ref.path).get("healthy"):
                return ref.iteration
        except (CheckpointCorruptError, OSError, ValueError):
            continue
    return None


def load_latest_state(out_dir: str | Path) -> dict[str, Any] | None:
    """Dernière ligne lisible de `out_dir/evalhook_state.jsonl` (None si absent ou vide)."""
    import json

    p = Path(out_dir) / STATE_FILE
    if not p.exists():
        return None
    last: dict[str, Any] | None = None
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue  # ligne tronquée (arrêt brutal pendant l'écriture) : la précédente fait foi
        if isinstance(obj, dict):
            last = obj
    return last


class EvalHook:
    """Fournisseur S1 pour `training.run(..., s1_provider=hook)` : `hook(iteration) -> bool`.

    Cycle : `bind_params(params)` (paramètres VIVANTS de la politique apprise, ordre de `PolicyNet.parameters()`),
    puis un appel par itération. Paramètres :
    - `batch_agent_steps` : transitions apprises par itération (`RunConfig.batch_agent_steps`) ;
    - `policy_factory` : construit un `PolicyNet` de même architecture (CPU) ;
    - `sim_factory(n_envs)` : simulateur 1v1 DÉDIÉ, construit au premier besoin puis réutilisé ;
    - `identity`, `train_config_hash`, `git_commit_id` : provenance écrite dans les instantanés ;
    - `healthy_probe` : itération du dernier checkpoint sain (voir le docstring du module) ;
    - `clock` : horloge injectable (tests).
    """

    def __init__(
        self,
        cfg: EvalHookConfig,
        *,
        batch_agent_steps: int,
        run_seed: int,
        out_dir: str | Path,
        snapshot_dir: str | Path,
        policy_factory: Callable[[], PolicyNet],
        sim_factory: Callable[[int], Simulator],
        identity: CheckpointIdentity,
        reward_config: RewardConfig | None = None,
        build_obs: ObsBuilder | None = None,
        obs_dim: int | None = None,
        healthy_probe: Callable[[], int | None] | None = None,
        train_config_hash: str = "unknown",
        git_commit_id: str = "unknown",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if batch_agent_steps < 1:
            raise ValueError("batch_agent_steps >= 1")
        self.cfg = cfg
        self.batch_agent_steps = batch_agent_steps
        self.seed = cfg.seed if cfg.seed is not None else run_seed
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = Path(snapshot_dir)
        self._snapshots = CheckpointManager(self.snapshot_dir, keep=SNAPSHOT_KEEP)
        self._policy_factory = policy_factory
        self._sim_factory = sim_factory
        self._sim: Simulator | None = None
        self.identity = identity
        self.reward_config = reward_config or RewardConfig()
        self._build_obs = build_obs
        self._obs_dim = obs_dim
        self.healthy_probe = healthy_probe
        self.train_config_hash = train_config_hash
        self.git_commit_id = git_commit_id
        self._clock = clock
        self._net = policy_factory()
        self._live: list[torch.Tensor] | None = None
        self._theta0_pending: dict[str, torch.Tensor] | None = None
        # --- état sérialisable
        self.history: list[dict[str, Any]] = []
        self.theta0: dict[str, Any] | None = None
        self.s1_iteration: int | None = None
        self.s1_eval_iteration: int | None = None
        self.rollbacks = 0
        self._healthy_iteration: int | None = None
        self.eval_seconds: list[float] = []
        """Durées murales des évaluations de CE processus (non sérialisées ; mesure du coût)."""

    # ------------------------------------------------------------------------------------------- propriétés

    @property
    def s1_reached(self) -> bool:
        return self.s1_iteration is not None

    def eval_index(self, iteration: int) -> int:
        """Indice k de la dernière évaluation due au plus tard à l'itération `iteration` (0 avant la première)."""
        return (iteration * self.batch_agent_steps) // self.cfg.every_timesteps

    def is_due(self, iteration: int) -> bool:
        return iteration >= 1 and self.eval_index(iteration) > self.eval_index(iteration - 1)

    def streak(self) -> int:
        """Réussites consécutives en fin d'historique."""
        n = 0
        for rec in reversed(self.history):
            if not rec["passed"]:
                break
            n += 1
        return n

    # ------------------------------------------------------------------------------------------- liaison

    def bind_params(self, params: Sequence[torch.Tensor]) -> None:
        """Lie le hook aux paramètres VIVANTS de la politique apprise (les `len(PolicyNet.parameters())` premiers de
        `params`, formes vérifiées) ; les copie au moment de l'évaluation. Au premier appel sans θ₀ dans l'état, retient
        leur valeur courante comme θ₀ (écrit au premier appel du hook). Rappelable (nouveau run, même processus)."""
        mine = list(self._net.parameters())
        given = list(params)
        if len(given) < len(mine):
            raise EvalHookError(f"{len(given)} paramètres fournis, {len(mine)} attendus pour la politique")
        for i, (a, b) in enumerate(zip(mine, given, strict=False)):
            if tuple(a.shape) != tuple(b.shape):
                raise EvalHookError(f"paramètre {i} : forme {tuple(b.shape)} != {tuple(a.shape)} (politique)")
        self._live = given[: len(mine)]
        if self.theta0 is None and self._theta0_pending is None:
            self._copy_live()
            self._theta0_pending = {k: v.detach().clone() for k, v in self._net.state_dict().items()}

    def wrap_optimizer_factory(self, factory: Callable[..., Any]) -> Callable[..., Any]:
        """Enveloppe d'`optimizer_factory` de `run` : lie le hook aux paramètres reçus (politique en tête, voir
        `run` : `[*policy.parameters(), *value.parameters()]`), puis délègue à `factory`."""

        def wrapped(params: list[torch.nn.Parameter], ppo: Any, device: torch.device) -> Any:
            self.bind_params(params)
            return factory(params, ppo, device)

        return wrapped

    def note_checkpoint(self, iteration: int, healthy: bool) -> None:
        """À appeler par `run` après chaque checkpoint écrit (remplace `healthy_probe`)."""
        if healthy:
            self._healthy_iteration = int(iteration)

    # ------------------------------------------------------------------------------------------- appel par itération

    def __call__(self, iteration: int) -> bool:
        return self.on_iteration(iteration)

    def on_iteration(self, iteration: int) -> bool:
        """Retour en arrière éventuel, θ₀, évaluation si due, confirmation de S1. Rend `s1_reached`."""
        self._sync(iteration)
        if self._theta0_pending is not None:
            self._write_theta0()
        if self.is_due(iteration):
            self._evaluate(iteration)
        if not self.s1_reached:
            self._try_latch(iteration)
        return self.s1_reached

    # ------------------------------------------------------------------------------------------- état

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "config": self.cfg.to_dict(),
            "seed": self.seed,
            "batch_agent_steps": self.batch_agent_steps,
            "history": [dict(r) for r in self.history],
            "theta0": dict(self.theta0) if self.theta0 is not None else None,
            "s1_iteration": self.s1_iteration,
            "s1_eval_iteration": self.s1_eval_iteration,
            "rollbacks": self.rollbacks,
            "healthy_iteration": self._healthy_iteration,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restaure l'état (reprise ou restauration). Refuse une config, une graine ou une taille de lot différente :
        les évaluations ne seraient plus comparables. Le compteur de retours en arrière ne recule jamais."""
        if int(state.get("version", -1)) != STATE_VERSION:
            raise EvalHookError(f"version d'état {state.get('version')} != {STATE_VERSION}")
        if dict(state["config"]) != self.cfg.to_dict():
            raise EvalHookError(f"config d'évaluation différente de l'état sauvegardé : {state['config']}")
        if int(state["seed"]) != self.seed or int(state["batch_agent_steps"]) != self.batch_agent_steps:
            raise EvalHookError("graine ou transitions par itération différentes de l'état sauvegardé")
        self.history = [dict(r) for r in state["history"]]
        self.theta0 = dict(state["theta0"]) if state["theta0"] is not None else None
        if self.theta0 is not None:
            self._theta0_pending = None
        self.s1_iteration = state["s1_iteration"]
        self.s1_eval_iteration = state["s1_eval_iteration"]
        self.rollbacks = max(self.rollbacks, int(state["rollbacks"]))
        self._healthy_iteration = state.get("healthy_iteration")

    def _persist(self) -> None:
        append_jsonl(self.out_dir / STATE_FILE, {"t": time.time(), **self.state_dict()})

    def _log(self, obj: Mapping[str, Any]) -> None:
        append_jsonl(self.out_dir / EVALS_FILE, {"t": time.time(), **obj})

    # ------------------------------------------------------------------------------------------- retour en arrière

    def _sync(self, iteration: int) -> None:
        dropped = [r for r in self.history if r["iteration"] >= iteration]
        unlatch = self.s1_iteration is not None and self.s1_iteration >= iteration
        if not dropped and not unlatch:
            return
        self.history = [r for r in self.history if r["iteration"] < iteration]
        if unlatch:
            self.s1_iteration, self.s1_eval_iteration = None, None
        if self._healthy_iteration is not None and self._healthy_iteration >= iteration:
            self._healthy_iteration = iteration - 1
        self.rollbacks += 1
        self._log(
            {
                "event": "rollback",
                "iteration": iteration,
                "dropped_eval_iterations": [r["iteration"] for r in dropped],
                "s1_unlatched": unlatch,
                "rollbacks": self.rollbacks,
            }
        )
        self._persist()

    def _confirmed_through(self) -> int | None:
        if self.healthy_probe is not None:
            probed = self.healthy_probe()
            if probed is not None:
                return probed
        return self._healthy_iteration

    def _try_latch(self, iteration: int) -> None:
        candidates: list[dict[str, Any]] = []
        n = 0
        for rec in self.history:
            n = n + 1 if rec["passed"] else 0
            if n >= self.cfg.consecutive:
                candidates.append(rec)
        if not candidates:
            return
        confirmed = self._confirmed_through()
        if confirmed is None:
            return
        for rec in candidates:
            if rec["iteration"] <= confirmed:
                self.s1_iteration, self.s1_eval_iteration = iteration, int(rec["iteration"])
                self._log(
                    {
                        "event": "s1_reached",
                        "iteration": iteration,
                        "eval_iteration": rec["iteration"],
                        "eval_index": rec["k"],
                        "healthy_checkpoint_iteration": confirmed,
                        "series": [r["iteration"] for r in self.history if r["iteration"] <= rec["iteration"]][
                            -self.cfg.consecutive :
                        ],
                    }
                )
                self._persist()
                return

    # ------------------------------------------------------------------------------------------- instantanés

    def _copy_live(self) -> None:
        if self._live is None:
            raise EvalHookError("hook non lié : appeler bind_params (ou wrap_optimizer_factory) avant le run")
        with torch.no_grad():
            for dst, src in zip(self._net.parameters(), self._live, strict=True):
                dst.copy_(src.detach().to("cpu"))

    def _save_snapshot(self, weights: Mapping[str, torch.Tensor], iteration: int, k: int) -> str:
        state = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "kind": "eval_snapshot",
            "identity": dataclasses.asdict(self.identity),
            "iteration": int(iteration),
            "eval_index": int(k),
            "trained_timesteps": int(iteration) * self.batch_agent_steps,
            "train_config_hash": self.train_config_hash,
            "trainer": {"learner": {"policy": {n: t.detach().cpu().clone() for n, t in weights.items()}}},
            "versions": {"git_commit": self.git_commit_id, "torch": str(torch.__version__)},
        }
        return self._snapshots.save(state, healthy=True).name

    def _load_snapshot(self, name: str) -> dict[str, torch.Tensor]:
        weights: dict[str, torch.Tensor] = CheckpointManager.load(self.snapshot_dir / name)["trainer"]["learner"][
            "policy"
        ]
        return weights

    def _write_theta0(self) -> None:
        assert self._theta0_pending is not None
        name = self._save_snapshot(self._theta0_pending, 0, 0)
        self.theta0 = {"snapshot": name, "source": "bind_params"}
        self._theta0_pending = None
        self._persist()

    def _frozen_for(self, k: int) -> tuple[int, int, str]:
        """(indice, itération, fichier) de l'instantané gelé pour l'évaluation k : celui de l'évaluation
        k - lag retenue, sinon la plus récente retenue d'indice inférieur, sinon θ₀."""
        target = k - self.cfg.lag_evals
        best: dict[str, Any] | None = None
        for rec in self.history:
            if rec.get("snapshot") and rec["k"] <= target and (best is None or rec["k"] >= best["k"]):
                best = rec
        if best is not None:
            return int(best["k"]), int(best["iteration"]), str(best["snapshot"])
        if self.theta0 is None:
            raise EvalHookError("θ₀ absent : hook non lié avant la première itération")
        return 0, 0, str(self.theta0["snapshot"])

    # ------------------------------------------------------------------------------------------- évaluation

    def _seed_for(self, k: int, role: int) -> int:
        return int(np.random.SeedSequence([self.seed, k, role]).generate_state(1, dtype=np.uint32)[0])

    def _match(self, a: Any, b: Any, episodes: int, seed: int) -> MatchResult:
        if self._sim is None:
            self._sim = self._sim_factory(self.cfg.n_envs)
        return play_match(
            a,
            b,
            self._sim,
            episodes,
            seed=seed,
            reward_config=self.reward_config,
            build_obs=self._build_obs,
            obs_dim=self._obs_dim,
        )

    def _evaluate(self, iteration: int) -> None:
        cfg = self.cfg
        k = self.eval_index(iteration)
        t0 = self._clock()
        self._copy_live()
        rec: dict[str, Any] = {
            "event": "eval",
            "k": k,
            "iteration": iteration,
            "trained_timesteps": iteration * self.batch_agent_steps,
            "r0_goals_per_min": cfg.r0_goals_per_min,
            "rollbacks": self.rollbacks,
            "s1_reached_before": self.s1_reached,
        }
        finite = all(bool(torch.isfinite(p).all()) for p in self._net.parameters())
        if not finite:
            rec.update(passed=False, reason="paramètres non finis : aucune partie jouée", snapshot=None)
            self._finish(rec, t0, {})
            return
        rec["snapshot"] = self._save_snapshot(self._net.state_dict(), iteration, k)
        t_snap = self._clock()
        mode = cast(Literal["sample", "argmax"], cfg.mode)

        current = NetPolicy(self._net, mode, self._seed_for(k, _ROLE_CURRENT_VS_RANDOM))
        rnd = RandomPolicy(self._seed_for(k, _ROLE_RANDOM_AGENT))
        res_r = self._match(current, rnd, cfg.episodes_random, self._seed_for(k, _ROLE_RANDOM_MATCH))
        t_rand = self._clock()

        fk, fit, fname = self._frozen_for(k)
        frozen_net = self._policy_factory()
        frozen_net.load_state_dict(self._load_snapshot(fname))
        current = NetPolicy(self._net, mode, self._seed_for(k, _ROLE_CURRENT_VS_FROZEN))
        frozen = NetPolicy(frozen_net, mode, self._seed_for(k, _ROLE_FROZEN_AGENT))
        res_f = self._match(current, frozen, cfg.episodes_frozen, self._seed_for(k, _ROLE_FROZEN_MATCH))
        t_frz = self._clock()

        rand = _random_block(res_r, cfg)
        frz = _frozen_block(res_f, cfg)
        frz.update(eval_index=fk, iteration=fit, snapshot=fname, lag_target_index=max(k - cfg.lag_evals, 0))
        cond_a = rand["verdict"] == DecisiveVerdict.PASSED.value
        low = frz["goals_for_per_min"]["low"]
        cond_b = low is not None and low > cfg.r0_goals_per_min
        rec.update(
            random=rand,
            frozen=frz,
            condition_a=cond_a,
            condition_b=cond_b,
            passed=bool(cond_a and cond_b),
            reason="(a) et (b)"
            if cond_a and cond_b
            else f"(a)={'oui' if cond_a else 'non'} (b)={'oui' if cond_b else 'non'}",
        )
        wh = rand["wilson_high"]
        rec["k7_bad"] = bool(rand["decisive"] >= cfg.min_decisive and wh is not None and wh < 0.5)
        timing = {"snapshot_s": t_snap - t0, "random_s": t_rand - t_snap, "frozen_s": t_frz - t_rand}
        self._finish(rec, t0, timing)

    def _finish(self, rec: dict[str, Any], t0: float, timing: dict[str, float]) -> None:
        rec.setdefault("k7_bad", False)
        self.history.append(rec)
        rec["streak"] = self.streak()
        total = self._clock() - t0
        rec["timing"] = {**timing, "total_s": total}
        self.eval_seconds.append(total)
        self._log(rec)
        if self.s1_reached and self.s1_iteration is not None:
            after = [r for r in self.history if r["iteration"] > (self.s1_eval_iteration or 0)]
            n = 0
            for r in reversed(after):
                if not r["k7_bad"]:
                    break
                n += 1
            if n == self.cfg.k7_consecutive:
                self._log({"event": "k7_alert", "iteration": rec["iteration"], "consecutive": n})
        self._persist()


def _random_block(res: MatchResult, cfg: EvalHookConfig) -> dict[str, Any]:
    o = res.outcome
    w, d, lo = int((o == 1).sum()), int((o == 0).sum()), int((o == -1).sum())
    test = decisive_win_test(w, lo, min_decisive=cfg.min_decisive, alpha=cfg.wilson_alpha)
    return {
        "episodes": res.n_episodes,
        "wins": w,
        "draws": d,
        "losses": lo,
        "decisive": w + lo,
        "score": (w + 0.5 * d) / res.n_episodes,
        "verdict": test.verdict.value,
        "win_rate_decisive": test.win_rate,
        "wilson_low": test.wilson_low,
        "wilson_high": test.wilson_high,
        "wilson_alpha_one_sided": cfg.wilson_alpha,
        "minutes": float(res.ticks.sum()) / TICKS_PER_MINUTE,
    }


def _rate(count: int, minutes: float, cfg: EvalHookConfig) -> dict[str, float | int | None]:
    if minutes <= 0:
        return {"count": count, "value": None, "low": None, "high": None}
    lo, hi = garwood_interval(count, cfg.garwood_alpha, two_sided=cfg.garwood_two_sided)
    return {"count": count, "value": count / minutes, "low": lo / minutes, "high": hi / minutes}


def _frozen_block(res: MatchResult, cfg: EvalHookConfig) -> dict[str, Any]:
    o = res.outcome
    w, d, lo = int((o == 1).sum()), int((o == 0).sum()), int((o == -1).sum())
    minutes = float(res.ticks.sum()) / TICKS_PER_MINUTE
    return {
        "episodes": res.n_episodes,
        "wins": w,
        "draws": d,
        "losses": lo,
        "minutes": minutes,
        "goals_for_per_min": _rate(w, minutes, cfg),
        "goals_against_per_min": _rate(lo, minutes, cfg),
        "garwood_alpha": cfg.garwood_alpha,
        "garwood_two_sided": cfg.garwood_two_sided,
    }


__all__ = [
    "EVALS_FILE",
    "STATE_FILE",
    "EvalHook",
    "EvalHookConfig",
    "EvalHookError",
    "latest_healthy_iteration",
    "load_latest_state",
]
