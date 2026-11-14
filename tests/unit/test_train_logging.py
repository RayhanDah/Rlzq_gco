"""Journalisation (T5) : JSONL stdlib, débit, scalaires d'itération, TensorBoard optionnel."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_ppo import Corridor, _ManualAdam, _small_cfg  # noqa: E402

from models import PolicyNet, ValueNet  # noqa: E402
from training import (  # noqa: E402
    Collector,
    JsonlWriter,
    Learner,
    MultiWriter,
    OverlapTrainer,
    TensorBoardWriter,
    ThroughputMeter,
    grad_norms,
    iteration_scalars,
)


def test_jsonl_writer_appends_lines_and_keeps_nonfinite_visible(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "m.jsonl"
    w = JsonlWriter(path)
    w.write(1, {"a": 1.5, "b": float("nan")})
    w.write(2, {"a": float("inf")})
    w.close()
    w.close()  # idempotent
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in rows] == [1, 2] and rows[0]["a"] == 1.5
    assert rows[0]["b"] == "nan" and rows[1]["a"] == "inf"
    JsonlWriter(path).write(3, {"a": 0.0})  # ajout, sans écraser
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_multi_writer_fans_out(tmp_path: Path) -> None:
    a, b = JsonlWriter(tmp_path / "a.jsonl"), JsonlWriter(tmp_path / "b.jsonl")
    m = MultiWriter([a, b])
    m.write(0, {"x": 1.0})
    m.flush()
    m.close()
    assert len((tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len((tmp_path / "b.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_throughput_meter_with_fake_clock() -> None:
    t = iter([0.0, 2.0, 3.0])
    m = ThroughputMeter(clock=lambda: next(t))
    assert m.tick(0) == 0.0
    assert m.tick(1000) == 500.0
    assert m.tick(1000) == 0.0


def test_iteration_scalars_full_set() -> None:
    cfg = _small_cfg(lr_start=3e-3, lr_end=3e-3)
    env = Corridor(16)
    pol, val = PolicyNet(7, n_actions=3, hidden=(16,)), ValueNet(7, hidden=(16,))
    learner = Learner(
        pol, val, cfg, torch.device("cpu"), optimizer=_ManualAdam([*pol.parameters(), *val.parameters()], 3e-3)
    )
    trainer = OverlapTrainer(
        learner, Collector(env, PolicyNet(7, n_actions=3, hidden=(16,)), 64, torch.device("cpu"), 0)
    )
    trainer.iterate()
    batch = trainer.pending_batch
    res = trainer.iterate()
    s = iteration_scalars(res, p_kick=0.7, steps_per_s=123.0, batch=batch, n_cars=1, learner=learner)
    for key in (
        "ppo/approx_kl",
        "ppo/entropy",
        "ppo/grad_norm",
        "ppo/explained_variance",
        "train/staleness",
        "perf/agent_steps_per_s",
        "env/p_kick",
        "env/goals",
        "env/episode_len_est",
        "grad/policy_last_mb_postclip",
        "grad/value_last_mb_postclip",
    ):
        assert key in s and math.isfinite(s[key]), key
    assert s["train/staleness"] == 1.0 and s["env/p_kick"] == 0.7
    # norme globale post-clip = racine de la somme des carrés par réseau, et ≤ max_grad_norm (+ marge flottante)
    g = grad_norms(learner)
    total = math.hypot(*g.values())
    assert total <= cfg.max_grad_norm * (1 + 1e-4)
    trainer.close()


@pytest.mark.skipif(importlib.util.find_spec("tensorboard") is not None, reason="tensorboard installé")
def test_tensorboard_writer_fails_clearly_when_package_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="tensorboard"):
        TensorBoardWriter(tmp_path)


def test_try_append_jsonl_ne_masque_jamais_l_exception_d_origine(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from training.logging import append_jsonl, try_append_jsonl

    # Arrange : un chemin impossible (dossier inexistant) ; une exception « d'origine » en cours de traitement
    bad = tmp_path / "absent" / "events.jsonl"
    good = tmp_path / "events.jsonl"
    # Act
    with pytest.raises(ValueError, match="origine"):
        try:
            raise ValueError("origine")
        except ValueError:
            err = try_append_jsonl(bad, {"event": "exception"})
            raise
    # Assert : l'erreur d'écriture est rendue et signalée, l'exception d'origine remonte intacte
    assert isinstance(err, OSError) and "écriture impossible" in capsys.readouterr().err
    assert try_append_jsonl(good, {"b": 1, "a": float("nan")}) is None
    append_jsonl(good, {"c": 2}, fsync=False)
    assert [json.loads(x) for x in good.read_text(encoding="utf-8").splitlines()][1] == {"c": 2}
