"""Configuration de run (T6) : chargement YAML, validation stricte, dérivés, hash, `configs/smoke.yaml`."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from training import ConfigError, RunConfig

ROOT = Path(__file__).resolve().parents[2]

BASE: dict[str, Any] = {
    "name": "t",
    "seed": 3,
    "device": "cpu",
    "env": {"n_envs": 4, "n_cars": 2, "tick_skip": 8},
    "reward": {"variant": "GCO-std", "no_touch_timeout_s": 20.0, "max_episode_s": 120.0},
    "initial_states": {"p_kick_start": 0.2, "p_kick_end": 0.8, "anneal_timesteps": 400},
    "ppo": {"epochs": 1, "minibatch_size": 16, "target_batch_size": 32},
    "collection": {"steps_per_batch": 4},
    "budget": {"total_timesteps": 320},
}


def _cfg(**over: Any) -> dict[str, Any]:
    d = copy.deepcopy(BASE)
    for k, v in over.items():
        d[k] = v
    return d


def test_config_valide_et_derives() -> None:
    # Arrange / Act
    cfg = RunConfig.from_dict(BASE)
    # Assert : 4 arènes x 2 agents x 4 steps = 32 par lot ; 320 / 32 = 10 itérations
    assert cfg.n_agents == 8 and cfg.batch_agent_steps == 32
    assert cfg.total_iterations == 10 and cfg.ppo.total_iterations == 10
    assert cfg.anneal_iterations == 13  # ceil(400 / 32)


def test_budget_arrondi_au_lot_superieur() -> None:
    cfg = RunConfig.from_dict(_cfg(budget={"total_timesteps": 321}))
    assert cfg.total_iterations == 11


def test_calendrier_p_kick_lineaire_puis_constant() -> None:
    cfg = RunConfig.from_dict(BASE)
    assert cfg.p_kick_at(0) == pytest.approx(0.2)
    assert cfg.p_kick_at(cfg.anneal_iterations) == pytest.approx(0.8)
    assert cfg.p_kick_at(10 * cfg.anneal_iterations) == pytest.approx(0.8)
    # revue T6 (A1.5) : la dernière itération de l'annealing (indice n - 1) utilise déjà la cible
    assert cfg.p_kick_at(cfg.anneal_iterations - 1) == pytest.approx(0.8)
    mid = cfg.p_kick_at(cfg.anneal_iterations // 2)
    assert 0.2 < mid < 0.8
    deltas = [cfg.p_kick_at(i + 1) - cfg.p_kick_at(i) for i in range(cfg.anneal_iterations - 1)]
    assert max(deltas) - min(deltas) < 1e-12  # linéaire


def test_annealing_d_une_seule_iteration_utilise_la_cible() -> None:
    # Une rampe d'une itération n'a pas de pas intermédiaire : la cible est utilisée dès l'itération 0.
    cfg = RunConfig.from_dict(_cfg(initial_states={"p_kick_start": 0.2, "p_kick_end": 0.8, "anneal_timesteps": 32}))
    assert cfg.anneal_iterations == 1 and cfg.p_kick_at(0) == pytest.approx(0.8)


def test_annealing_nul_donne_p_kick_final_des_le_depart() -> None:
    cfg = RunConfig.from_dict(_cfg(initial_states={"p_kick_start": 0.2, "p_kick_end": 0.8, "anneal_timesteps": 0}))
    assert cfg.p_kick_at(0) == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda d: d.update(bogus=1), "clés inconnues"),
        (lambda d: d["env"].update(n_env=4), "env"),
        (lambda d: d["ppo"].update(clipp=0.2), "ppo"),
        (lambda d: d["ppo"].update(total_iterations=5), "dérivée"),
        (lambda d: d["env"].update(n_envs="4"), "env.n_envs"),
        (lambda d: d["env"].update(n_envs=True), "env.n_envs"),
        (lambda d: d["env"].update(n_envs=4.0), "env.n_envs"),
        (lambda d: d["ppo"].update(lr_start="2e-4"), "ppo.lr_start"),
        (lambda d: d["collection"].update(overlap="oui"), "collection.overlap"),
        (lambda d: d.update(seed=-1), "seed"),
        (lambda d: d.update(device="tpu"), "device"),
        (lambda d: d.update(name="a/b"), "name"),
        (lambda d: d["reward"].update(variant="GCO-x"), "reward"),
        (lambda d: d["reward"].update(max_episode_s=0.001), "reward"),
        (lambda d: d["initial_states"].update(p_kick_end=1.5), "p_kick_end"),
        (lambda d: d["ppo"].update(epochs=0), "PPOConfig"),
        (lambda d: d.update(kill={"kl_threshold": 0.05, "oops": 1}), "kill"),
        (lambda d: d.update(kill={"kl_patience": 0}), "kill"),
        (lambda d: d.update(model={"policy_hidden": [16, "a"]}), r"model\.policy_hidden\[1\]"),
        (lambda d: d.update(model={"policy_hidden": 16}), "model.policy_hidden"),
        (lambda d: d.update(checkpoint={"every_iterations": 0}), "checkpoint"),
        (lambda d: d.pop("budget"), "budget"),
        (lambda d: d["collection"].pop("steps_per_batch"), "obligatoires"),
        (lambda d: d["initial_states"].pop("anneal_timesteps"), "obligatoires"),
        (lambda d: d.update(smoke={"resume_at_fraction": 1.0}), "smoke"),
    ],
)
def test_configuration_invalide_refusee_avec_message_explicite(mutate: Any, needle: str) -> None:
    d = copy.deepcopy(BASE)
    mutate(d)
    with pytest.raises(ConfigError, match=needle):
        RunConfig.from_dict(d)


def test_gco_s_refuse_les_etats_initiaux_generiques() -> None:
    """Spec §3 : GCO-S = kickoffs uniquement."""
    d = _cfg(reward={"variant": "GCO-S"})
    with pytest.raises(ConfigError, match="génériques interdits"):
        RunConfig.from_dict(d)
    d["initial_states"] = {"p_kick_start": 1.0, "p_kick_end": 1.0, "anneal_timesteps": 0}
    assert RunConfig.from_dict(d).reward.variant == "GCO-S"


def test_yaml_invalide_ou_non_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1, 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML invalide"):
        RunConfig.from_yaml(bad)
    lst = tmp_path / "list.yaml"
    lst.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        RunConfig.from_yaml(lst)


def test_notation_scientifique_yaml_sans_point_refusee(tmp_path: Path) -> None:
    """PyYAML lit `2e-4` comme une chaîne : le validateur la refuse au lieu de la convertir en silence."""
    p = tmp_path / "c.yaml"
    p.write_text(
        "name: t\nppo: {lr_start: 2e-4}\ncollection: {steps_per_batch: 1}\nbudget: {total_timesteps: 10}\n"
        "initial_states: {anneal_timesteps: 0}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"ppo\.lr_start"):
        RunConfig.from_yaml(p)


@pytest.mark.parametrize(
    ("raw", "section", "key", "fix"),
    [("2e-4", "ppo", "lr_start", "2.0e-4"), ("1e6", "budget", "total_timesteps", "1000000")],
)
def test_message_notation_scientifique_explicite(raw: str, section: str, key: str, fix: str) -> None:
    # Revue T6 (A1.8) : PyYAML lit `2e-4` comme une chaîne ; le message doit dire pourquoi et comment corriger.
    d = copy.deepcopy(BASE)
    d[section][key] = raw  # ce que yaml.safe_load rend pour « 2e-4 » ou « 1e6 »
    with pytest.raises(ConfigError, match="notation scientifique") as err:
        RunConfig.from_dict(d)
    assert f"{section}.{key}" in str(err.value) and "PyYAML" in str(err.value) and fix in str(err.value)


def test_message_chaine_non_numerique_inchange() -> None:
    d = copy.deepcopy(BASE)
    d["ppo"]["lr_start"] = "vite"
    with pytest.raises(ConfigError) as err:
        RunConfig.from_dict(d)
    assert "notation scientifique" not in str(err.value)


def test_hash_stable_sensible_et_sections_operationnelles_exclues() -> None:
    a = RunConfig.from_dict(BASE)
    b = RunConfig.from_dict(copy.deepcopy(BASE))
    assert a.trajectory_hash() == b.trajectory_hash() and a.full_hash() == b.full_hash()
    assert len(a.trajectory_hash()) == 64
    # une section d'entraînement change le hash de trajectoire
    for mutated in (
        _cfg(seed=4),
        _cfg(ppo={**BASE["ppo"], "epochs": 2}),
        _cfg(budget={"total_timesteps": 640}),
        _cfg(reward={"variant": "GCO-std", "no_touch_timeout_s": 10.0}),
    ):
        assert RunConfig.from_dict(mutated).trajectory_hash() != a.trajectory_hash()
    # checkpoints, gardes, journalisation, smoke : pas de changement de trajectoire, mais le hash complet change
    ops = RunConfig.from_dict(_cfg(checkpoint={"every_iterations": 7}, kill={"max_restores": 9}, smoke={}))
    assert ops.trajectory_hash() == a.trajectory_hash()
    assert ops.full_hash() != a.full_hash()


def test_smoke_yaml_se_charge_et_totalise_au_moins_1m_timesteps() -> None:
    cfg = RunConfig.from_yaml(ROOT / "configs" / "smoke.yaml")
    assert cfg.env.n_envs == 4096 and cfg.env.n_cars == 2 and cfg.collection.steps_per_batch == 7
    assert cfg.collection.overlap is True and cfg.device == "auto" and cfg.reward.variant == "GCO-std"
    assert cfg.batch_agent_steps == 4096 * 2 * 7 == 57_344
    assert cfg.total_iterations * cfg.batch_agent_steps >= 1_000_000
    assert (cfg.total_iterations - 1) * cfg.batch_agent_steps < 1_000_000  # pas d'itération de trop
    assert cfg.smoke is not None and cfg.smoke.require_resume and cfg.smoke.max_nan == 0
    # checkpoints assez fréquents pour exercer la rotation, et un checkpoint avant le milieu du run
    assert cfg.checkpoint.every_iterations * cfg.checkpoint.keep < cfg.total_iterations
    assert cfg.p_kick_at(0) == pytest.approx(0.2) and cfg.p_kick_at(cfg.total_iterations) == pytest.approx(0.8)
    # revue T6 (A1.5) : la dernière itération du run tourne avec la cible (avant : 0,767)
    assert cfg.p_kick_at(cfg.total_iterations - 1) == pytest.approx(cfg.initial_states.p_kick_end)
    # section kill complète et conforme à docs/PLAN.md
    k = cfg.kill
    assert (k.kl_threshold, k.kl_patience, k.entropy_floor_frac, k.ev_patience) == (0.05, 5, 0.10, 50)
    assert k.max_restores == 3 and k.max_consecutive_restores == 2


def test_smoke_yaml_contient_une_section_kill_textuelle() -> None:
    """`scripts/gate_check.py` cherche la chaîne `kill` dans `configs/long_run.yaml` : smoke.yaml en sert de modèle."""
    text = (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8")
    assert "\nkill:" in text
