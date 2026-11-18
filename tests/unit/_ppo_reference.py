"""Référence figée de `Learner.update` AVANT la self-imitation (copie verbatim de `src/training/ppo.py`, commit
f2218a6, lignes 212–317), pour le test de régression bit à bit « SIL désactivée = code d'avant » (ADR-009 D7-1).

Ne pas modifier : ce fichier est la définition du comportement attendu quand `sil` est absente ou `enabled: false`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from models.policy import log_prob_entropy
from training.gae import bootstrap_values, gae
from training.ppo import Learner, PPOStats
from training.rollout import RolloutBatch


class ReferenceLearner(Learner):
    """`Learner` dont `update` est celle d'avant la SIL."""

    def update(self, batch: RolloutBatch) -> PPOStats:
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

        def advantages() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                v = self._values(obs).view(t_len, n)
                v_next = bootstrap_values(v, self.value(last_obs), self._values(final_obs), final_t, final_agent, term)
                adv, ret = gae(rew, v, v_next, term, trunc, cfg.gamma, cfg.gae_lambda)
            return select(adv), select(ret), select(v)

        n_mb = max(1, math.ceil(n_samples / cfg.minibatch_size))
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
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    kl = ((ratio - 1.0) - logratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip).float().mean()
                    acc += torch.stack([pg_loss.detach(), v_loss.detach(), ent.detach(), kl, clip_frac, grad_norm])
        assert first is not None
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
        )
