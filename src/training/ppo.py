"""PPO-clip v0 (`docs/01_papers.md` §1.1–1.4, §2) et boucle collecte/apprentissage en recouvrement (ADR-001 §3).

Leviers mesurés repris du prototype (`scripts/bench_learner.py`, ADR-001 M3/M4) : `index_select` au lieu de `x[idx]`
(l'indexation avancée plante sur XPU), Adam `fused=True`, fp32, statistiques accumulées sur le device et lues une
seule fois par itération, apprentissage du lot k dans un thread pendant la collecte du lot k+1 avec une copie acteur
de π (staleness 1). Aucune récompense ni signal d'exploration n'est ajouté ici (GCO strict). Seule aide optionnelle :
la self-imitation (`training.sil`, ADR-005 D6 intervention (3)), désactivée par défaut ; désactivée, `Learner.update`
exécute exactement les mêmes opérations qu'avant son introduction (test de régression bit à bit).
"""

from __future__ import annotations

import functools
import math
import queue
import threading
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Protocol

import numpy as np
import torch

from models.policy import PolicyNet, log_prob_entropy
from models.value import ValueNet
from training.gae import bootstrap_values, gae
from training.rollout import Collector, RolloutBatch
from training.sil import SelfImitation, SILConfig

FUSED_ADAM_DEVICES = ("cpu", "cuda", "xpu")  # sous-ensemble de torch.utils._foreach_utils (torch 2.15.0)


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparamètres PPO v0. Valeurs par défaut = `docs/01_papers.md` §2 (source en commentaire).

    `total_iterations` : longueur du run, sur laquelle LR et coefficient d'entropie décroissent linéairement.
    γ reste constant ici : sa montée 0,995 → 0,997–0,998 (§1.2) est décidée par le trainer, pas par un calendrier fixe.
    """

    total_iterations: int
    gamma: float = 0.995  # §1.2, §2
    gae_lambda: float = 0.95  # §1.2, §2
    clip: float = 0.2  # §1.1, §2
    lr_start: float = 2e-4  # §2 : 2e-4 → 5e-5 linéaire
    lr_end: float = 5e-5
    ent_coef_start: float = 0.01  # §2 : 0,01 → 0,002
    ent_coef_end: float = 0.002
    vf_coef: float = 1.0  # réseaux séparés : « sans objet » (paper_logo §12) ; ne pèse que via le grad clip global
    max_grad_norm: float = 0.5  # §1.1, §2 : clip global
    epochs: int = 2  # §2
    minibatch_size: int = 25_000  # §2 : 12,5k–25k ; taille maximale, les minibatchs sont équilibrés
    target_batch_size: int = 50_000  # §2 : 50k–100k timesteps agents par itération
    adam_beta1: float = 0.9  # §1.1
    adam_beta2: float = 0.999
    adam_eps: float = 1e-5  # §1.1 (37 details)
    adv_norm_eps: float = 1e-8
    recompute_advantages: bool = True  # §1.1 et paper_logo §12 (WM §3.5) : avantages recalculés à chaque epoch
    max_staleness: int = 1  # §1.3 : staleness ≤ 1

    def __post_init__(self) -> None:
        checks = {
            "total_iterations ≥ 1": self.total_iterations >= 1,
            "0 ≤ γ ≤ 1": 0.0 <= self.gamma <= 1.0,
            "0 ≤ λ ≤ 1": 0.0 <= self.gae_lambda <= 1.0,
            "clip > 0": self.clip > 0.0,
            "lr ≥ 0": self.lr_start >= 0.0 and self.lr_end >= 0.0,
            "entropie ≥ 0": self.ent_coef_start >= 0.0 and self.ent_coef_end >= 0.0,
            "vf_coef > 0": self.vf_coef > 0.0,
            "max_grad_norm > 0": self.max_grad_norm > 0.0,
            "epochs ≥ 1": self.epochs >= 1,
            "minibatch_size ≥ 4": self.minibatch_size >= 4,  # garantit ≥ 2 échantillons par minibatch équilibré
            "target_batch_size ≥ 2": self.target_batch_size >= 2,
            "max_staleness ≥ 0": self.max_staleness >= 0,
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"PPOConfig invalide : {', '.join(bad)}")

    def progress(self, iteration: int) -> float:
        """Fraction du run dans [0, 1] (0 à la première itération, 1 à la dernière et au-delà)."""
        if self.total_iterations == 1:
            return 1.0
        return min(max(iteration, 0) / (self.total_iterations - 1), 1.0)

    def lr_at(self, iteration: int) -> float:
        f = self.progress(iteration)
        return self.lr_start + f * (self.lr_end - self.lr_start)

    def ent_coef_at(self, iteration: int) -> float:
        f = self.progress(iteration)
        return self.ent_coef_start + f * (self.ent_coef_end - self.ent_coef_start)

    def rollout_steps(self, n_agents: int) -> int:
        """T tel que T·N ≥ `target_batch_size` (plus petit entier)."""
        if n_agents <= 0:
            raise ValueError(f"n_agents={n_agents}")
        return max(1, math.ceil(self.target_batch_size / n_agents))


@dataclass(frozen=True)
class PPOStats:
    """Statistiques d'une itération (moyennes sur les minibatchs de toutes les epochs, sauf mention)."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float  # estimateur k3 : E[(r − 1) − log r] (37 details) ; alarme > 0,02 (§1.1)
    clip_frac: float  # fraction des échantillons avec |r − 1| > clip
    grad_norm: float  # norme globale avant clip
    explained_variance: float  # 1 − Var(R − V)/Var(R), avant la première epoch
    adv_mean: float  # avantages bruts (avant normalisation), première epoch
    adv_std: float
    value_mean: float
    return_mean: float
    lr: float
    ent_coef: float
    n_samples: int
    n_minibatches: int
    n_term: int
    n_trunc: int
    staleness: int
    policy_version: int  # version de π après cette mise à jour
    sil: Mapping[str, float] = field(default_factory=dict)
    """Journal C4 de la self-imitation (`SILStats.as_dict`), vide si elle est désactivée ; hors de `as_dict`."""

    def as_dict(self) -> dict[str, float | int]:
        """Statistiques PPO (sans `sil`, journalisé à part sous `sil/*`)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "sil"}


class OptimizerLike(Protocol):
    """Interface d'optimiseur utilisée par `Learner` (rend l'optimiseur injectable dans les tests)."""

    param_groups: list[dict[str, Any]]

    def zero_grad(self, set_to_none: bool = True) -> None: ...

    def step(self) -> object: ...


def resolve_device(name: str = "auto") -> torch.device:
    """`auto` : cuda, sinon xpu, sinon cpu ; sinon `torch.device(name)`."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def make_optimizer(params: Iterable[torch.nn.Parameter], cfg: PPOConfig, device: torch.device) -> torch.optim.Adam:
    """Adam (β 0,9/0,999, ε 1e-5), `fused=True` si le device le permet (+19 % mesuré sur XPU, ADR-001 M4)."""
    fused = device.type in FUSED_ADAM_DEVICES
    return torch.optim.Adam(
        list(params), lr=cfg.lr_start, betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_eps, fused=fused
    )


class Learner:
    """Mise à jour PPO-clip sur un `RolloutBatch`. Les valeurs (V(obs), V(last_obs), V(final_obs)) sont calculées ici.

    `version` = nombre de mises à jour appliquées à π ; un lot de version v n'est accepté que si
    0 ≤ version − v ≤ `cfg.max_staleness`. Les permutations des minibatchs viennent d'un générateur CPU propre au
    learner (indépendant du flux de la collecte et du device).

    `sil` : self-imitation (`training.sil`) si fournie et `enabled` ; sinon (défaut) aucun objet SIL n'existe et la
    mise à jour est celle du PPO seul. Activée : chaque lot entre dans le suivi des épisodes avant la mise à jour, puis
    un lot SIL s'ajoute à chaque minibatch PPO, même optimiseur (mode `joint`, ADR-009 D3 amendé, défaut), ou des pas
    SIL suivent les epochs PPO avec `sil_optimizer` (mode `separate`, Adam distinct par défaut).
    """

    def __init__(
        self,
        policy: PolicyNet,
        value: ValueNet,
        cfg: PPOConfig,
        device: torch.device,
        optimizer: OptimizerLike | None = None,
        seed: int = 0,
        sil: SILConfig | None = None,
        sil_optimizer: Any = None,
    ) -> None:
        self.cfg, self.device = cfg, device
        self.policy = policy.to(device)
        self.value = value.to(device)
        self.params = [*self.policy.parameters(), *self.value.parameters()]
        self.optimizer: OptimizerLike = optimizer if optimizer is not None else make_optimizer(self.params, cfg, device)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.version = 0
        self.policy_params = list(self.policy.parameters())
        self.sil: SelfImitation | None = None
        if sil is not None and sil.enabled:
            if sil.mode == "separate" and sil_optimizer is None:
                sil_optimizer = make_optimizer(self.params, cfg, device)  # Adam distinct (ADR-009 D3)
            opt = sil_optimizer if sil.mode == "separate" else None
            self.sil = SelfImitation(sil, cfg.gamma, policy.obs_dim, policy.n_actions, seed, opt)

    def state_dict(self) -> dict[str, Any]:
        """État complet du learner pour la reprise (T5) : poids π et V, optimiseur, générateur des permutations,
        version et, si la SIL est activée, son état (clé `sil` : buffer, anneau des épisodes en cours, tirage).
        Les tenseurs et tableaux sont ceux du learner (non copiés) ; l'optimiseur doit exposer `state_dict()`."""
        state = {
            "version": self.version,
            "generator": self.generator.get_state(),
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "optimizer": self._stateful_optimizer().state_dict(),
        }
        if self.sil is not None:
            state["sil"] = self.sil.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (self.sil is not None) != ("sil" in state):
            raise ValueError("état SIL présent XOR learner avec SIL : checkpoint et configuration incohérents")
        if self.sil is not None:
            self.sil.load_state_dict(state["sil"])
        self.policy.load_state_dict(state["policy"])
        self.value.load_state_dict(state["value"])
        self._stateful_optimizer().load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"].cpu())
        self.version = int(state["version"])

    def _stateful_optimizer(self) -> Any:
        if not (hasattr(self.optimizer, "state_dict") and hasattr(self.optimizer, "load_state_dict")):
            raise TypeError("l'optimiseur n'expose pas state_dict()/load_state_dict() : reprise impossible")
        return self.optimizer

    def _values(self, obs: torch.Tensor) -> torch.Tensor:
        """V par morceaux de `minibatch_size` (borne la mémoire des activations 3×512)."""
        if obs.shape[0] == 0:
            return torch.zeros(0, device=self.device)
        chunks = torch.split(obs, self.cfg.minibatch_size)
        return torch.cat([self.value(c) for c in chunks])

    def update(self, batch: RolloutBatch) -> PPOStats:
        """Une mise à jour PPO sur `batch`. `grad_norm` : norme globale (π et V) avant clip du gradient appliqué par
        l'optimiseur PPO, moyenne des minibatchs ; en mode SIL `joint` avec un lot SIL, c'est celle du gradient TOTAL
        (PPO + SIL) ; en mode `separate`, celle de PPO seul. Les normes séparées sur π sont dans `sil` (`SILStats`)."""
        cfg, dev = self.cfg, self.device
        staleness = self.version - batch.policy_version
        if not 0 <= staleness <= cfg.max_staleness:
            raise ValueError(
                f"lot de version {batch.policy_version}, π en version {self.version} : staleness hors borne"
            )
        if batch.obs_dim != self.policy.obs_dim:
            raise ValueError(f"obs_dim du lot {batch.obs_dim} ≠ politique {self.policy.obs_dim}")
        lr, ent_coef = cfg.lr_at(self.version), cfg.ent_coef_at(self.version)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        t_len, n, d = batch.obs.shape
        obs = torch.from_numpy(batch.obs).to(dev).reshape(t_len * n, d)
        rew = torch.from_numpy(batch.rew).to(dev)
        term = torch.from_numpy(batch.term).to(dev)
        trunc = torch.from_numpy(batch.trunc).to(dev)
        last_obs = torch.from_numpy(batch.last_obs).to(dev)
        final_obs = torch.from_numpy(batch.final_obs).to(dev)
        final_t = torch.from_numpy(batch.final_t).to(dev)
        final_agent = torch.from_numpy(batch.final_agent).to(dev)
        cols = None if bool(batch.learn_mask.all()) else torch.from_numpy(np.flatnonzero(batch.learn_mask)).to(dev)

        def select(x: torch.Tensor) -> torch.Tensor:
            """(T, N, …) → (T·N_learn, …) sur les agents de `learn_mask`."""
            if cols is not None:
                x = x.index_select(1, cols)
            return x.reshape(-1, *x.shape[2:])

        obs_l = select(obs.view(t_len, n, d))
        act_l = select(torch.from_numpy(batch.act).to(dev))
        old_logp_l = select(torch.from_numpy(batch.logp).to(dev))
        n_samples = int(obs_l.shape[0])
        if n_samples < 2:
            raise ValueError("au moins 2 échantillons appris sont nécessaires (normalisation des avantages)")
        n_mb = max(1, math.ceil(n_samples / cfg.minibatch_size))
        sil_it = None
        if self.sil is not None:  # SIL désactivée : aucune opération de plus (bit à bit identique au PPO seul)
            sil_it = self.sil.begin(
                batch.obs,
                batch.act,
                batch.rew,
                batch.term,
                batch.trunc,
                batch.learn_mask,
                iteration=self.version,
                learned_per_iteration=n_samples,
                device=dev,
                n_ppo_steps=cfg.epochs * n_mb,
                generic=batch.ep_generic,
                policy_params=self.policy_params,
                ppo_optimizer=self.optimizer,
            )

        def advantages() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                v = self._values(obs).view(t_len, n)
                v_next = bootstrap_values(v, self.value(last_obs), self._values(final_obs), final_t, final_agent, term)
                adv, ret = gae(rew, v, v_next, term, trunc, cfg.gamma, cfg.gae_lambda)
            return select(adv), select(ret), select(v)

        acc = torch.zeros(6, device=dev)
        first: torch.Tensor | None = None
        for epoch in range(cfg.epochs):
            if epoch == 0 or cfg.recompute_advantages:
                adv, ret, v_old = advantages()
                if epoch == 0:
                    first = torch.stack(
                        [
                            1.0 - (ret - v_old).var() / ret.var().clamp_min(1e-12),
                            adv.mean(),
                            adv.std(),
                            v_old.mean(),
                            ret.mean(),
                        ]
                    )
                    if sil_it is not None:
                        sil_it.note_raw_advantages(adv)
            perm = torch.randperm(n_samples, generator=self.generator).to(dev)
            for idx in perm.tensor_split(n_mb):
                # index_select plutôt que x[idx] : x[idx] lève UR_RESULT_ERROR_OUT_OF_RESOURCES sur XPU (ADR-001 M3).
                mb_adv = adv.index_select(0, idx)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + cfg.adv_norm_eps)  # par minibatch, §1.1
                mb_obs = obs_l.index_select(0, idx)
                logp, entropy = log_prob_entropy(self.policy(mb_obs), act_l.index_select(0, idx))
                logratio = logp - old_logp_l.index_select(0, idx)
                ratio = logratio.exp()
                pg_loss = -torch.min(ratio * mb_adv, ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * mb_adv).mean()
                v_loss = 0.5 * (self.value(mb_obs) - ret.index_select(0, idx)).pow(2).mean()  # sans value clipping
                ent = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - ent_coef * ent
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if sil_it is not None:  # journal C4 ; mode joint : gradient de c_sil·L_sil ajouté avant le clip
                    sil_it.note_ppo_grad()
                    sil_it.joint_step(self.policy, self.value)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.optimizer.step()
                if sil_it is not None:  # mode joint : moments fantômes, part de déplacement (ADR-009 D5)
                    sil_it.after_ppo_step(grad_norm, cfg.max_grad_norm)
                with torch.no_grad():
                    kl = ((ratio - 1.0) - logratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip).float().mean()
                    acc += torch.stack([pg_loss.detach(), v_loss.detach(), ent.detach(), kl, clip_frac, grad_norm])
        assert first is not None
        if sil_it is not None:  # mode separate : pas SIL après les epochs PPO, optimiseur SIL distinct
            sil_it.separate_steps(self.policy, self.value, self.params, lr)
        # Une seule lecture hôte par itération (ADR-001 M4).
        vals = torch.cat([acc / (cfg.epochs * n_mb), first]).tolist()
        self.version += 1
        return PPOStats(
            policy_loss=vals[0],
            value_loss=vals[1],
            entropy=vals[2],
            approx_kl=vals[3],
            clip_frac=vals[4],
            grad_norm=vals[5],
            explained_variance=vals[6],
            adv_mean=vals[7],
            adv_std=vals[8],
            value_mean=vals[9],
            return_mean=vals[10],
            lr=lr,
            ent_coef=ent_coef,
            n_samples=n_samples,
            n_minibatches=n_mb,
            n_term=int(batch.term.sum()),
            n_trunc=int((batch.trunc & ~batch.term).sum()),
            staleness=staleness,
            policy_version=self.version,
            sil={} if sil_it is None else sil_it.stats().as_dict(),
        )


class LearnerError(RuntimeError):
    """Exception levée dans le thread learner, propagée au thread principal (cause dans `__cause__`)."""


@dataclass(frozen=True)
class IterationResult:
    stats: PPOStats
    learned_batch_version: int  # version de π qui a collecté le lot appris
    policy_version: int  # version de π après la mise à jour (= version de l'acteur pour le lot suivant)
    agent_steps: int  # transitions agents collectées depuis le début (tous agents)


class _LearnerWorker:
    """Thread learner persistant : exécute une tâche à la fois, rend (résultat, exception)."""

    def __init__(self) -> None:
        self._tasks: queue.SimpleQueue[Callable[[], PPOStats] | None] = queue.SimpleQueue()
        self._results: queue.SimpleQueue[tuple[PPOStats | None, BaseException | None]] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._loop, name="ppo-learner", daemon=True)  # daemon : un trainer
        # jamais fermé (exception non rattrapée) ne bloque pas la sortie de l'interpréteur
        self._thread.start()

    def _loop(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            try:
                self._results.put((task(), None))
            except BaseException as exc:  # relevée dans le thread appelant
                self._results.put((None, exc))
            del task  # ne pas retenir le lot (ni le learner) pendant l'attente de la tâche suivante

    def submit(self, task: Callable[[], PPOStats]) -> None:
        self._tasks.put(task)

    def result(self) -> tuple[PPOStats | None, BaseException | None]:
        return self._results.get()

    def stop(self) -> None:
        """Laisse finir la tâche en cours (résultat ignoré), puis termine et joint le thread."""
        self.request_stop()
        self._thread.join()

    def request_stop(self) -> None:
        self._tasks.put(None)


class OverlapTrainer:
    """Itération PPO : apprentissage du lot k dans un thread pendant que le thread appelant collecte le lot k+1 avec
    une copie acteur de π (staleness 1, ADR-001 §3, ADR-003 §6). L'acteur est rafraîchi après `join`.

    `overlap=False` : apprentissage puis collecte en séquence (staleness 0), même interface.
    Un seul thread learner (« ppo-learner ») sert toutes les itérations : il est créé à la première, reste en attente
    entre deux `iterate` (aucune mise à jour ne déborde d'un `iterate`) et s'arrête dans `close()` ; il est recréé au
    besoin. Un thread neuf par itération faisait croître la mémoire privée de ≈ 2,3 Mo par mise à jour sur XPU
    (ressources par thread jamais rendues, ADR-001 M7). Une exception du learner est relevée en `LearnerError` dans
    l'appelant ; toute exception (learner ou collecte) arrête le thread et rend le trainer inutilisable (état
    partiellement mis à jour) : reprendre depuis un checkpoint.
    """

    def __init__(self, learner: Learner, collector: Collector, overlap: bool = True) -> None:
        if collector.policy is learner.policy:
            raise ValueError("le collecteur doit utiliser une copie acteur de la politique, pas le module du learner")
        self.learner, self.collector, self.overlap = learner, collector, overlap
        self._pending: RolloutBatch | None = None
        self._actor_version = -1
        self._worker: _LearnerWorker | None = None
        self._finalizer: weakref.finalize[Any, Any] | None = None
        self._busy = False  # une mise à jour est en cours dans le thread learner
        self._failed = False

    @property
    def policy_version(self) -> int:
        return self.learner.version

    @property
    def pending_batch(self) -> RolloutBatch | None:
        """Lot déjà collecté, à apprendre à la prochaine itération (à inclure dans un checkpoint, ADR-003 §7)."""
        return self._pending

    def state_dict(self) -> dict[str, Any]:
        """État pour la reprise exacte, à la frontière d'itération : learner, collecteur, lot en attente (k+1) et
        version de l'acteur. Refuse un trainer en échec ou dont le thread learner tourne."""
        if self._failed or self._busy:
            raise RuntimeError("state_dict impossible : trainer en échec ou thread learner actif")
        return {
            "learner": self.learner.state_dict(),
            "collector": self.collector.state_dict(),
            "pending": self._pending,
            "actor_version": self._actor_version,
        }

    def load_state_dict(self, state: Mapping[str, Any], obs: Any = None) -> None:
        """Inverse de `state_dict`. `obs` : observation courante de l'env si elle diffère de celle sauvegardée (reprise
        « modulo reset des arènes »). L'acteur est resynchronisé sur π (copie de π à la frontière d'itération)."""
        if self._busy:
            raise RuntimeError("load_state_dict impossible : thread learner actif")
        self.learner.load_state_dict(state["learner"])
        self.collector.load_state_dict(state["collector"], obs=obs)
        self._pending = state["pending"]
        self.sync_actor()
        self._actor_version = int(state["actor_version"])
        self._failed = False

    def sync_actor(self) -> None:
        self.collector.policy.load_state_dict(self.learner.policy.state_dict())
        self._actor_version = self.learner.version

    def iterate(self) -> IterationResult:
        if self._failed:
            raise RuntimeError("trainer en échec après une exception du learner")
        if self._pending is None:
            self.sync_actor()
            self._pending = self.collector.collect(self._actor_version)
        batch = self._pending
        if not self.overlap:
            stats = self._learn_or_fail(batch)
            self.sync_actor()
            nxt = self.collector.collect(self._actor_version)
        else:
            if self._worker is None:
                self._worker = _LearnerWorker()
                # trainer libéré sans close() : le thread en attente se termine (sans join dans un finaliseur)
                self._finalizer = weakref.finalize(self, self._worker.request_stop)
            self._busy = True
            self._worker.submit(functools.partial(self.learner.update, batch))
            try:
                nxt = self.collector.collect(self._actor_version)
                done, error = self._worker.result()
            except BaseException:  # collecte, ou interruption pendant l'attente du learner (revue)
                self._failed = True
                self.close()
                raise
            self._busy = False
            if error is not None or done is None:
                self._failed = True
                self.close()
                raise LearnerError("échec du thread learner") from error
            stats = done
            self.sync_actor()
        self._pending = nxt
        return IterationResult(
            stats=stats,
            learned_batch_version=batch.policy_version,
            policy_version=self.learner.version,
            agent_steps=self.collector.agent_steps,
        )

    def _learn_or_fail(self, batch: RolloutBatch) -> PPOStats:
        try:
            return self.learner.update(batch)
        except Exception as exc:
            self._failed = True
            raise LearnerError("échec de la mise à jour PPO") from exc

    def close(self) -> None:
        """Attend la fin de la mise à jour en cours, puis arrête le thread learner (idempotent ; `iterate` le
        recrée)."""
        worker, self._worker = self._worker, None
        if worker is not None:
            if self._finalizer is not None:
                self._finalizer.detach()
            worker.stop()
        self._busy = False

    def __enter__(self) -> OverlapTrainer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
