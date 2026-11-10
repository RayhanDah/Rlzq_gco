"""Tests du micro-benchmark learner : formes des réseaux et itération PPO valide sur un mini-lot."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bench_learner", ROOT / "scripts" / "bench_learner.py")
assert spec and spec.loader
bl = importlib.util.module_from_spec(spec)
sys.modules["bench_learner"] = bl
spec.loader.exec_module(bl)


def test_nets_follow_ppo_v0_config() -> None:
    policy, value = bl.make_nets(17, torch.device("cpu"))
    x = torch.randn(5, 17)
    assert policy(x).shape == (5, bl.N_ACTIONS)
    assert value(x).shape == (5, 1)
    widths = [m.out_features for m in policy if isinstance(m, torch.nn.Linear)]
    assert widths == [256, 256, 256, bl.N_ACTIONS]
    assert [m.out_features for m in value if isinstance(m, torch.nn.Linear)] == [512, 512, 512, 1]


def test_ppo_update_changes_params_and_stays_finite() -> None:
    torch.manual_seed(0)
    policy, value = bl.make_nets(8, torch.device("cpu"))
    opt = torch.optim.Adam([*policy.parameters(), *value.parameters()], lr=1e-3)
    obs = torch.randn(64, 8)
    with torch.no_grad():
        dist = torch.distributions.Categorical(logits=policy(obs))
        act = dist.sample()
        data = {"obs": obs, "act": act, "logp": dist.log_prob(act), "adv": torch.randn(64), "ret": torch.randn(64)}
    before = [p.detach().clone() for p in policy.parameters()]
    stats = bl.ppo_update(policy, value, opt, data, epochs=2, minibatch=16)
    assert math.isfinite(stats["loss"]) and math.isfinite(stats["kl"])
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters(), strict=True))
