"""Évaluation en cours de run (`training.evalhook`, A4, ADR-005 D2) sur simulateur factice.

Logique S1 (compteur de 3 réussites, remise à zéro, irréversibilité, confirmation par checkpoint sain, retour en
arrière, état sérialisable) : `play_match` est remplacé par un script qui rend des résultats choisis (réussite ou
échec) ; boucle réelle (`training.run`) : simulateur factice de `test_run`, sans RocketGPU.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")


from eval.match import MatchResult, RandomPolicy, load_checkpoint_policy  # noqa: E402
from models import PolicyNet  # noqa: E402
from training.checkpoint import CheckpointIdentity  # noqa: E402
from training.evalhook import (  # noqa: E402
    EVALS_FILE,
    EvalHook,
    EvalHookConfig,
    EvalHookError,
    load_latest_state,
)

hook_mod = importlib.import_module("training.evalhook")
OBS_D = 3
TICKS_1MIN = 120 * 60


def _result(wins: int, losses: int, n: int = 64, ticks: int = TICKS_1MIN) -> MatchResult:
    flat = np.zeros(n, dtype=np.int8)
    flat[:wins] = 1
    flat[wins : wins + losses] = -1
    return MatchResult(
        outcome=flat.reshape(-1, 2),
        ticks=np.full((n // 2, 2), ticks, dtype=np.int64),
        seed=0,
        n_pairs=n // 2,
        n_envs=4,
        tick_skip=8,
        env_steps=0,
        wall_s=0.0,
    )


# réussite : 30 V / 0 D contre l'aléatoire (Wilson bas ≈ 0,91) ; 40 buts en 64 min contre le gelé (Garwood bas ≈ 0,45)
PASS_RANDOM, PASS_FROZEN = _result(30, 0), _result(40, 0)
# échec (a) : 10 V / 10 D contre l'aléatoire (Wilson bas < 0,5)
FAIL_RANDOM = _result(10, 10)


class Script:
    """Remplace `play_match` : rend, pour la n-ième évaluation, réussite ou échec selon `plan` ('P' / 'F')."""

    def __init__(self, plan: str) -> None:
        self.plan = plan
        self.calls: list[tuple[str, int]] = []  # (rôle, graine)
        self.n_eval = 0

    def __call__(self, a: Any, b: Any, sim: Any, n: int, *, seed: int, **kw: Any) -> MatchResult:
        role = "random" if isinstance(b, RandomPolicy) else "frozen"
        self.calls.append((role, seed))
        if role == "random":
            ok = self.plan[self.n_eval] == "P" if self.n_eval < len(self.plan) else False
            self.n_eval += 1
            return PASS_RANDOM if ok else FAIL_RANDOM
        return PASS_FROZEN


class Healthy:
    """Itération du dernier checkpoint sain (sonde) ; par défaut, un checkpoint sain à chaque itération précédente."""

    def __init__(self) -> None:
        self.value: int | None = None

    def __call__(self) -> int | None:
        return self.value


def make_hook(tmp: Path, cfg: EvalHookConfig | None = None, probe: Any = None, **kw: Any) -> EvalHook:
    ident = CheckpointIdentity.current(reward_config_hash="r", obs_dim=OBS_D, tick_skip=8)
    hook = EvalHook(
        cfg or EvalHookConfig(every_timesteps=2, frozen_lag_timesteps=4, episodes_random=64, episodes_frozen=64),
        learned_steps_per_iteration=1,
        run_seed=3,
        out_dir=tmp / "out",
        snapshot_dir=tmp / "snap",
        policy_factory=lambda: PolicyNet(OBS_D, 90, (4,)),
        sim_factory=lambda n: object(),
        identity=ident,
        healthy_probe=probe,
        **kw,
    )
    return hook


def bound(hook: EvalHook, seed: int = 0) -> PolicyNet:
    torch.manual_seed(seed)
    live = PolicyNet(OBS_D, 90, (4,))
    hook.bind_policy(live)
    return live


def drive(hook: EvalHook, probe: Healthy, iterations: range) -> list[bool]:
    """Appelle le hook comme `run` : un checkpoint sain à chaque itération (sonde = itération précédente)."""
    out = []
    for it in iterations:
        probe.value = it - 1
        out.append(hook(it))
    return out


def evals(tmp: Path) -> list[dict[str, Any]]:
    p = tmp / "out" / EVALS_FILE
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------------------------------------ S1


def test_s1_apres_exactement_trois_reussites_consecutives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : une évaluation toutes les 2 itérations (2, 4, 6) ; trois réussites
    monkeypatch.setattr(hook_mod, "play_match", Script("PPP"))
    probe = Healthy()
    hook = make_hook(tmp_path, probe=probe)
    bound(hook)
    # Act
    got = drive(hook, probe, range(1, 9))
    # Assert : la 3e réussite (it. 6) n'arme S1 qu'une fois couverte par un checkpoint sain (it. 7 : sain = 6)
    assert got == [False] * 6 + [True, True]
    assert hook.s1_iteration == 7 and hook.s1_eval_iteration == 6
    s1 = [e for e in evals(tmp_path) if e["event"] == "s1_reached"]
    assert len(s1) == 1 and s1[0]["series"] == [2, 4, 6] and s1[0]["healthy_checkpoint_iteration"] == 6


def test_deux_reussites_ne_suffisent_pas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hook_mod, "play_match", Script("PPFPP"))
    probe = Healthy()
    hook = make_hook(tmp_path, probe=probe)
    bound(hook)
    got = drive(hook, probe, range(1, 14))  # évaluations 2, 4, 6 (échec), 8, 10, puis 12 : échec (hors plan)
    assert not any(got) and hook.streak() == 0
    assert [e["streak"] for e in evals(tmp_path) if e["event"] == "eval"] == [1, 2, 0, 1, 2, 0]


def test_echec_remet_le_compteur_a_zero_puis_trois_de_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hook_mod, "play_match", Script("PPFPPP"))
    probe = Healthy()
    hook = make_hook(tmp_path, probe=probe)
    bound(hook)
    got = drive(hook, probe, range(1, 14))
    assert got.index(True) + 1 == 13 and hook.s1_eval_iteration == 12


def test_s1_irreversible_malgre_les_echecs_suivants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hook_mod, "play_match", Script("PPPFFFF"))
    probe = Healthy()
    hook = make_hook(tmp_path, probe=probe)
    bound(hook)
    got = drive(hook, probe, range(1, 16))
    assert got[6:] == [True] * 9 and hook.streak() == 0 and hook.s1_iteration == 7


def test_sans_checkpoint_sain_couvrant_s1_n_est_pas_arme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : checkpoints non sains après l'itération 5 (série KL) : la 3e réussite (it. 6) n'est jamais couverte
    monkeypatch.setattr(hook_mod, "play_match", Script("PPP"))
    probe = Healthy()
    hook = make_hook(tmp_path, probe=probe)
    bound(hook)
    # Act
    got = [hook(it) for it in range(1, 7)]
    probe.value = 5
    got += [hook(it) for it in range(7, 9)]
    # Assert
    assert not any(got)
    # sans sonde ni note_checkpoint : jamais armé
    hook2 = make_hook(tmp_path / "b")
    bound(hook2)
    monkeypatch.setattr(hook_mod, "play_match", Script("PPP"))
    assert not any(hook2(it) for it in range(1, 9))
    hook2.note_checkpoint(8, healthy=False)
    assert hook2(9) is False
    hook2.note_checkpoint(9, healthy=True)
    assert hook2(10) is True


def test_iteration_divergente_ecartee_par_retour_en_arriere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : réussites à 2, 4, 6 ; l'itération 6 diverge, la garde restaure le checkpoint de l'itération 4 :
    # l'appel suivant porte l'itération 5. L'évaluation de 6 est retirée ; la nouvelle évaluation de 6 échoue.
    monkeypatch.setattr(hook_mod, "play_match", Script("PPPF"))
    probe = Healthy()
    hook = make_hook(tmp_path, probe=probe)
    bound(hook)
    got = drive(hook, probe, range(1, 7))
    # Act : restauration vers 4 (dernier sain = 4)
    probe.value = 4
    got += [hook(5)]
    got += drive(hook, probe, range(6, 10))
    # Assert
    assert not any(got)
    assert [r["iteration"] for r in hook.history] == [2, 4, 6, 8] and [r["passed"] for r in hook.history] == [
        True,
        True,
        False,
        False,
    ]
    rb = [e for e in evals(tmp_path) if e["event"] == "rollback"]
    assert len(rb) == 1 and rb[0]["iteration"] == 5 and rb[0]["dropped_eval_iterations"] == [6]
    assert hook.rollbacks == 1


def test_retour_avant_le_verrou_le_leve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # reprise depuis un checkpoint antérieur au verrou (arrêt brutal avant le checkpoint suivant)
    monkeypatch.setattr(hook_mod, "play_match", Script("PPP"))
    probe = Healthy()
    hook = make_hook(tmp_path, probe=probe)
    bound(hook)
    drive(hook, probe, range(1, 8))
    assert hook.s1_reached
    probe.value = 6
    assert hook(7) is True  # même itération que le verrou après retour à 6 : réarmé aussitôt (6 couvert)
    probe.value = 4
    assert hook(5) is False and hook.s1_iteration is None and [r["iteration"] for r in hook.history] == [2, 4]


def test_etat_restaure_depuis_state_dict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : deux réussites, état passé par JSON (comme dans un checkpoint ou evalhook_state.jsonl)
    monkeypatch.setattr(hook_mod, "play_match", Script("PP"))
    probe = Healthy()
    a = make_hook(tmp_path, probe=probe)
    bound(a)
    drive(a, probe, range(1, 6))
    state = json.loads(json.dumps(a.state_dict()))
    # Act : nouveau hook (nouveau processus), même config ; une réussite de plus
    monkeypatch.setattr(hook_mod, "play_match", Script("P"))
    b = make_hook(tmp_path, probe=probe)
    b.load_state_dict(state)
    bound(b, seed=99)  # θ₀ vient de l'état, pas de ces paramètres
    got = drive(b, probe, range(6, 8))
    # Assert
    assert got == [False, True] and b.s1_eval_iteration == 6 and b.streak() == 3
    assert b.theta0 == a.theta0
    assert load_latest_state(tmp_path / "out") == {**b.state_dict(), "t": load_latest_state(tmp_path / "out")["t"]}  # type: ignore[index]


def test_etat_incompatible_refuse(tmp_path: Path) -> None:
    a = make_hook(tmp_path)
    other = make_hook(tmp_path, cfg=EvalHookConfig(every_timesteps=2, frozen_lag_timesteps=4, r0_goals_per_min=0.05))
    with pytest.raises(EvalHookError, match="config"):
        other.load_state_dict(a.state_dict())
    with pytest.raises(EvalHookError, match="version"):
        a.load_state_dict({**a.state_dict(), "version": 0})


# ------------------------------------------------------------------------------------------------ instantanés, gelé


def test_gele_theta0_puis_instantane_retarde_et_instantanes_relisibles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script("FFFF")
    monkeypatch.setattr(hook_mod, "play_match", script)
    probe = Healthy()
    hook = make_hook(tmp_path, probe=probe)
    live = bound(hook)
    theta0 = {k: v.clone() for k, v in live.state_dict().items()}
    for it in range(1, 9):
        with torch.no_grad():
            for p in live.parameters():
                p.add_(0.01)  # « apprentissage »
        probe.value = it - 1
        hook(it)
    recs = [e for e in evals(tmp_path) if e["event"] == "eval"]
    # Assert : k = 1, 2 contre θ₀ ; k = 3 contre l'instantané de k = 1 ; k = 4 contre k = 2
    assert [r["frozen"]["eval_index"] for r in recs] == [0, 0, 1, 2]
    assert [r["frozen"]["iteration"] for r in recs] == [0, 0, 2, 4]
    snap = tmp_path / "snap"
    files = sorted(p.name for p in snap.iterdir())
    assert len(files) == 5  # θ₀ + 4 évaluations, aucun supprimé
    pol, info = load_checkpoint_policy(
        snap / hook.theta0["snapshot"], mode="argmax", seed=0, obs_dim=OBS_D, tick_skip=8
    )  # type: ignore[index]
    assert info.iteration == 0
    for k, v in theta0.items():
        assert torch.equal(pol.net.state_dict()[k], v)
    pol8, info8 = load_checkpoint_policy(snap / recs[-1]["snapshot"], mode="argmax", seed=0, obs_dim=OBS_D, tick_skip=8)
    assert info8.iteration == 8
    for k, v in live.state_dict().items():
        assert torch.equal(pol8.net.state_dict()[k], v)
    # graines : distinctes par rôle et par évaluation
    seeds = [s for _, s in script.calls]
    assert len(set(seeds)) == len(seeds)


def test_parametres_non_finis_evaluation_echouee_sans_partie(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Script("PPP")
    monkeypatch.setattr(hook_mod, "play_match", script)
    hook = make_hook(tmp_path)
    live = bound(hook)
    hook(1)
    with torch.no_grad():
        next(live.parameters()).fill_(float("nan"))
    hook(2)
    rec = evals(tmp_path)[-1]
    assert rec["passed"] is False and rec["snapshot"] is None and "non finis" in rec["reason"]
    assert script.calls == []


def test_journal_d7_wilson_garwood_r0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hook_mod, "play_match", Script("P"))
    hook = make_hook(tmp_path)
    bound(hook)
    hook(1)
    hook(2)
    rec = evals(tmp_path)[-1]
    r, f = rec["random"], rec["frozen"]
    assert rec["r0_goals_per_min"] == 0.032 and rec["passed"] is True
    assert r["wins"] == 30 and r["decisive"] == 30 and r["verdict"] == "passed" and r["wilson_low"] > 0.5
    assert f["minutes"] == pytest.approx(64.0) and f["goals_for_per_min"]["value"] == pytest.approx(40 / 64)
    # Garwood bilatéral 95 % de 40 : [28,58 ; 54,47] (tables), ramené à 64 minutes
    assert f["goals_for_per_min"]["low"] == pytest.approx(28.58 / 64, rel=1e-3)
    assert f["goals_against_per_min"]["value"] == 0.0 and f["goals_against_per_min"]["low"] == 0.0
    assert set(rec["timing"]) == {"snapshot_s", "random_s", "frozen_s", "total_s"}


def test_condition_b_seule_ne_suffit_pas_et_indetermine_echoue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 5 V / 0 D contre l'aléatoire : moins de 20 décisives = indéterminé = échec, même si (b) est vraie
    def few(a: Any, b: Any, sim: Any, n: int, **kw: Any) -> MatchResult:
        return _result(5, 0) if isinstance(b, RandomPolicy) else PASS_FROZEN

    monkeypatch.setattr(hook_mod, "play_match", few)
    hook = make_hook(tmp_path)
    bound(hook)
    hook(1)
    hook(2)
    rec = evals(tmp_path)[-1]
    assert rec["random"]["verdict"] == "indeterminate" and rec["condition_b"] is True and rec["passed"] is False


def test_alerte_k7_apres_s1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # après S1, borne haute de Wilson < 0,5 trois fois de suite (0 V / 30 D)
    script = Script("PPP")
    monkeypatch.setattr(hook_mod, "play_match", script)
    probe = Healthy()
    hook = make_hook(tmp_path, probe=probe)
    bound(hook)
    drive(hook, probe, range(1, 8))
    monkeypatch.setattr(
        hook_mod,
        "play_match",
        lambda a, b, s, n, **kw: _result(0, 30) if isinstance(b, RandomPolicy) else PASS_FROZEN,
    )
    drive(hook, probe, range(8, 15))
    k7 = [e for e in evals(tmp_path) if e["event"] == "k7_alert"]
    assert len(k7) == 1 and k7[0]["iteration"] == 12


def test_erreur_d_evaluation_ignoree_par_k7(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : après S1 (évaluations 2, 4, 6), mauvaise à 8, ERREUR à 10, mauvaises à 12 et 14
    monkeypatch.setattr(hook_mod, "play_match", Script("PPP"))
    probe = Healthy()
    hook = make_hook(tmp_path, probe=probe)
    bound(hook)
    drive(hook, probe, range(1, 8))

    def bad(a: Any, b: Any, s: Any, n: int, **kw: Any) -> MatchResult:
        if kw["seed"] == hook._seed_for(5, 0):
            raise RuntimeError("panne")
        return _result(0, 30) if isinstance(b, RandomPolicy) else PASS_FROZEN

    monkeypatch.setattr(hook_mod, "play_match", bad)
    # Act
    drive(hook, probe, range(8, 15))
    # Assert : l'erreur ne remet pas la série à zéro et ne compte pas : alerte à la 3e mauvaise (14), pas avant
    log = evals(tmp_path)
    assert [e["iteration"] for e in log if e["event"] == "eval_error"] == [10]
    assert [e["iteration"] for e in log if e["event"] == "k7_alert"] == [14]


def test_config_validee() -> None:
    assert EvalHookConfig().lag_evals == 2
    d = EvalHookConfig().to_dict()
    assert EvalHookConfig.from_dict(d) == EvalHookConfig()
    with pytest.raises(ValueError, match="inconnues"):
        EvalHookConfig.from_dict({"every": 1})
    with pytest.raises(ValueError, match="type"):
        EvalHookConfig.from_dict({"episodes_random": True})
    with pytest.raises(ValueError, match="multiple"):
        EvalHookConfig(every_timesteps=3, frozen_lag_timesteps=4)
    with pytest.raises(ValueError, match="mode"):
        EvalHookConfig(mode="greedy")
    assert EvalHookConfig.from_dict({"r0_goals_per_min": 1}).r0_goals_per_min == 1.0


def test_parametres_de_forme_differente_refuses(tmp_path: Path) -> None:
    hook = make_hook(tmp_path)
    with pytest.raises(EvalHookError, match="incompatible"):
        hook.bind_policy(PolicyNet(OBS_D, 90, (5,)))


# ----------------------------------------------------------------------------------------- protocole S1 v2 (ADR-006)

V2 = EvalHookConfig(
    protocol="s1_v2",
    every_timesteps=2,
    frozen_lag_timesteps=4,
    episodes_random=64,
    episodes_frozen=64,
    consecutive=2,
    k7_consecutive=2,
    r0_goals_per_min=0.027,
)


def _with_touches(res: MatchResult, a_touch: int, b_touch: int) -> MatchResult:
    t = np.zeros((res.n_pairs, 2, 2), dtype=np.int64)
    t[:, :, 0], t[:, :, 1] = a_touch, b_touch
    return dataclasses.replace(res, touches=t)


class ScriptV2:
    """`play_match` des évaluations v2 : plan 'P' / 'F' contre l'aléatoire ; contre θ₀, résultat `frozen` ; trace
    (rôle, graine, poids de B) pour vérifier QUEL adversaire est joué. 30 V / 0 D (en tête de l'ordre de jeu) : le SPRT
    +100 Elo accepte H1 après 12 victoires (12 x 0,2470 = 2,96 > 2,944)."""

    def __init__(self, plan: str, frozen: MatchResult = PASS_FROZEN) -> None:
        self.plan, self.frozen = plan, frozen
        self.n_eval = 0
        self.calls: list[tuple[str, int, Any]] = []

    def __call__(self, a: Any, b: Any, sim: Any, n: int, *, seed: int, **kw: Any) -> MatchResult:
        if isinstance(b, RandomPolicy):
            ok = self.n_eval < len(self.plan) and self.plan[self.n_eval] == "P"
            self.n_eval += 1
            self.calls.append(("random", seed, None))
            return _with_touches(PASS_RANDOM if ok else FAIL_RANDOM, 3, 1)
        self.calls.append(("frozen", seed, {k: v.clone() for k, v in b.net.state_dict().items()}))
        return _with_touches(self.frozen, 2, 0)


def test_v2_arme_au_deuxieme_point_reussi_pas_au_premier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : points aux itérations 2 et 4, réussis ; checkpoint sain = itération précédente
    monkeypatch.setattr(hook_mod, "play_match", ScriptV2("PP"))
    probe = Healthy()
    hook = make_hook(tmp_path, V2, probe=probe)
    bound(hook)
    # Act
    got = drive(hook, probe, range(1, 8))
    # Assert : rien au 1er point (it. 2, 3), ni à l'it. 4 ; armé au 2e, confirmé par le sain 4 à l'it. 5
    assert got == [False] * 4 + [True] * 3
    assert hook.s1_eval_iteration == 4 and hook.s1_iteration == 5
    s1 = [e for e in evals(tmp_path) if e["event"] == "s1_reached"]
    assert len(s1) == 1 and s1[0]["series"] == [2, 4]


def test_v2_un_echec_reinitialise_la_serie(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : points 2 (P), 4 (F), 6 (P), 8 (P)
    monkeypatch.setattr(hook_mod, "play_match", ScriptV2("PFPP"))
    probe = Healthy()
    hook = make_hook(tmp_path, V2, probe=probe)
    bound(hook)
    # Act
    got = drive(hook, probe, range(1, 12))
    # Assert : l'échec du point 4 remet la série à zéro : armé sur 6-8, pas sur 2-4-6
    assert got.index(True) + 1 == 9 and hook.s1_eval_iteration == 8
    assert [e["streak"] for e in evals(tmp_path) if e["event"] == "eval"][:4] == [1, 0, 1, 2]


def test_v2_confirmation_par_checkpoint_sain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hook_mod, "play_match", ScriptV2("PP"))
    hook = make_hook(tmp_path, V2)
    bound(hook)
    # sans sonde ni checkpoint noté : jamais armé, même après 2 points réussis
    assert not any(hook(it) for it in range(1, 9))
    hook.note_checkpoint(3, healthy=True)  # ne couvre pas le point 4
    assert hook(9) is False
    hook.note_checkpoint(4, healthy=True)
    assert hook(10) is True and hook.s1_eval_iteration == 4


def test_v2_retour_en_arriere_retire_le_point_ecarte(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : points réussis à 2 et 4 (non couverts) ; l'itération 4 diverge : restauration vers 3
    monkeypatch.setattr(hook_mod, "play_match", ScriptV2("PPF"))
    probe = Healthy()
    hook = make_hook(tmp_path, V2, probe=probe)
    bound(hook)
    probe.value = 1
    got = [hook(it) for it in range(1, 5)]
    assert [r["iteration"] for r in hook.history] == [2, 4]
    # Act : le point 4 est écarté ; il est rejoué et échoue
    probe.value = 3
    got += [hook(3), hook(4)]
    # Assert : un seul point réussi retenu, pas d'armement
    assert not any(got) and [r["iteration"] for r in hook.history] == [2, 4]
    assert [r["passed"] for r in hook.history] == [True, False] and hook.rollbacks == 1


def test_v2_critere_b_contre_theta0_fixe_et_non_gele_glissant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : la politique apprend (poids qui changent) ; 4 points
    script = ScriptV2("PPPP")
    monkeypatch.setattr(hook_mod, "play_match", script)
    probe = Healthy()
    hook = make_hook(tmp_path, V2, probe=probe)
    live = bound(hook)
    theta0 = {k: v.clone() for k, v in live.state_dict().items()}
    for it in range(1, 9):
        with torch.no_grad():
            for p in live.parameters():
                p.add_(0.01)
        probe.value = it - 1
        hook(it)
    # Assert : à chaque point, l'adversaire de (b) a exactement les poids de θ₀ (le glissant aurait ceux du point k - 2)
    frozen = [c for c in script.calls if c[0] == "frozen"]
    assert len(frozen) == 4
    for _, _, w in frozen:
        assert all(torch.equal(w[k], theta0[k]) for k in theta0)
    recs = [e for e in evals(tmp_path) if e["event"] == "eval"]
    assert [r["frozen"]["eval_index"] for r in recs] == [0, 0, 0, 0] and {r["frozen"]["iteration"] for r in recs} == {0}
    assert {r["frozen"]["snapshot"] for r in recs} == {hook.theta0["snapshot"]}  # type: ignore[index]


def test_v2_les_deux_criteres_sont_necessaires(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # (a) vrai, (b) faux : 0 but contre θ₀ ; puis (a) faux, (b) vrai
    monkeypatch.setattr(hook_mod, "play_match", ScriptV2("P", frozen=_result(0, 0)))
    hook = make_hook(tmp_path / "x", V2)
    bound(hook)
    hook(1), hook(2)
    r = evals(tmp_path / "x")[-1]
    assert r["condition_a"] is True and r["condition_b"] is False and r["passed"] is False
    monkeypatch.setattr(hook_mod, "play_match", ScriptV2("F"))
    hook2 = make_hook(tmp_path / "y", V2)
    bound(hook2)
    hook2(1), hook2(2)
    r = evals(tmp_path / "y")[-1]
    assert r["condition_a"] is False and r["condition_b"] is True and r["passed"] is False


def test_v2_aucun_plancher_de_decisives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 12 V / 0 D seulement (< min_decisive = 20) : le SPRT accepte H1 (ADR-006 D2 : pas de plancher)
    few = _result(12, 0)
    monkeypatch.setattr(
        hook_mod, "play_match", lambda a, b, s, n, **kw: few if isinstance(b, RandomPolicy) else PASS_FROZEN
    )
    hook = make_hook(tmp_path, V2)
    bound(hook)
    hook(1), hook(2)
    rec = evals(tmp_path)[-1]
    assert rec["random"]["sprt"]["decision"] == "accept_h1" and rec["passed"] is True


def test_v2_journal_i1_et_touches_i2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hook_mod, "play_match", ScriptV2("P"))
    hook = make_hook(tmp_path, V2)
    bound(hook)
    hook(1), hook(2)
    rec = evals(tmp_path)[-1]
    r, f = rec["random"], rec["frozen"]
    assert rec["protocol"] == "s1_v2" and rec["passed"] is True and rec["s1_reached_before"] is False
    assert "s1_reached" not in rec  # l'armement est l'événement `s1_reached`, pas un champ du point
    assert (r["wins"], r["draws"], r["losses"]) == (30, 34, 0) and r["episodes"] == 64
    assert set(r["elo_decisive"]) == {"value", "low", "high"} and r["elo_decisive"]["value"] > 0
    sp = r["sprt"]
    assert sp["decision"] == "accept_h1" and sp["n_episodes"] == 12 and sp["n_decisive"] == 12
    assert sp["llr"] >= sp["upper"] and sp["elo1"] == 100.0
    assert f["goals_for_per_min"]["value"] == pytest.approx(40 / 64)
    assert f["goals_for_per_min"]["low"] == pytest.approx(28.58 / 64, rel=1e-3)  # Garwood bilatéral 95 % de 40
    assert f["goals_against_per_min"]["value"] == 0.0 and rec["condition_b"] is True and f["reference"] == "theta0"
    assert rec["k7_bad"] is False and "total_s" in rec["timing"]
    # I2 : touches contre l'aléatoire et contre θ₀
    assert (
        r["touches"]["a_mean"] == 3.0 and r["touches"]["b_touch_rate"] == 1.0 and r["touches"]["no_touch_rate"] == 0.0
    )
    assert f["touches"]["a_touch_rate"] == 1.0 and f["touches"]["b_touch_rate"] == 0.0


def test_v1_journalise_aussi_les_touches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def match(a: Any, b: Any, sim: Any, n: int, **kw: Any) -> MatchResult:
        return _with_touches(PASS_RANDOM if isinstance(b, RandomPolicy) else PASS_FROZEN, 2, 5)

    monkeypatch.setattr(hook_mod, "play_match", match)
    hook = make_hook(tmp_path)
    bound(hook)
    hook(1), hook(2)
    rec = evals(tmp_path)[-1]
    assert rec["random"]["touches"]["a_mean"] == 2.0 and rec["frozen"]["touches"]["b_mean"] == 5.0


def test_v2_k7_apres_deux_points_consecutifs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hook_mod, "play_match", ScriptV2("PP"))
    probe = Healthy()
    hook = make_hook(tmp_path, V2, probe=probe)
    bound(hook)
    drive(hook, probe, range(1, 8))
    monkeypatch.setattr(
        hook_mod,
        "play_match",
        lambda a, b, s, n, **kw: _result(0, 30) if isinstance(b, RandomPolicy) else PASS_FROZEN,
    )
    drive(hook, probe, range(8, 15))
    k7 = [e for e in evals(tmp_path) if e["event"] == "k7_alert"]
    assert [e["iteration"] for e in k7] == [10] and k7[0]["consecutive"] == 2


def test_v2_config_et_etats_v1_rechargeables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # protocole par défaut = S1-léger ; la config s'écrit sans le nouveau champ
    assert EvalHookConfig().protocol == "s1_light"
    with pytest.raises(ValueError, match="protocol"):
        EvalHookConfig(protocol="s1_v3")
    with pytest.raises(ValueError, match="pair"):
        EvalHookConfig(protocol="s1_v2", episodes_random=63)
    assert EvalHookConfig.from_dict(V2.to_dict()) == V2
    # état v1 sauvegardé AVANT l'ajout des champs de protocole : rechargé sans erreur
    monkeypatch.setattr(hook_mod, "play_match", Script("PP"))
    probe = Healthy()
    a = make_hook(tmp_path / "a", probe=probe)
    bound(a)
    drive(a, probe, range(1, 6))
    old = json.loads(json.dumps(a.state_dict()))
    for key in ("protocol", "sprt_elo1", "sprt_alpha", "sprt_beta"):
        old["config"].pop(key)
    b = make_hook(tmp_path / "b", probe=probe)
    b.load_state_dict(old)
    assert [r["iteration"] for r in b.history] == [2, 4] and b.streak() == 2
    # un état v1 n'est pas rechargé par un hook v2 (les points ne sont pas comparables)
    with pytest.raises(EvalHookError, match="config"):
        make_hook(tmp_path / "c", V2).load_state_dict(old)


def _flat(outcomes: list[int]) -> MatchResult:
    """Résultat dont l'ordre de jeu (paire, côté) est exactement `outcomes` (longueur paire)."""
    n = len(outcomes)
    return MatchResult(
        outcome=np.asarray(outcomes, dtype=np.int8).reshape(-1, 2),
        ticks=np.full((n // 2, 2), TICKS_1MIN, dtype=np.int64),
        seed=0,
        n_pairs=n // 2,
        n_envs=4,
        tick_skip=8,
        env_steps=0,
        wall_s=0.0,
    )


def _v2_single(tmp: Path, monkeypatch: pytest.MonkeyPatch, res: MatchResult) -> dict[str, Any]:
    monkeypatch.setattr(
        hook_mod, "play_match", lambda a, b, s, n, **kw: res if isinstance(b, RandomPolicy) else PASS_FROZEN
    )
    hook = make_hook(tmp, V2)
    bound(hook)
    hook(1), hook(2)
    return evals(tmp)[-1]


def test_v2_plafond_du_sprt_egal_aux_episodes_et_echec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : 5 V / 3 D parmi 64 épisodes (LLR final ≈ 0,25) : aucune borne franchie ; (b) vraie (PASS_FROZEN)
    rec = _v2_single(tmp_path, monkeypatch, _result(5, 3))
    # Assert : le plafond (= episodes_random) est atteint sans décision : échec, pas « continue »
    sp = rec["random"]["sprt"]
    assert sp["decision"] == "cap_reached" and sp["n_episodes"] == V2.episodes_random == 64
    assert rec["condition_a"] is False and rec["condition_b"] is True and rec["passed"] is False


def test_v2_borne_franchie_a_mi_paire_ne_compte_qu_a_la_frontiere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : nul, puis 12 V (la borne haute est franchie à l'épisode 13, 1er de la paire 7), puis une D (2e de la
    # paire 7) : à la frontière de paire, LLR = 2,964 - 0,329 = 2,635 < 2,944 ; le reste, des nuls
    flat = [0] + [1] * 12 + [-1] + [0] * 50
    rec = _v2_single(tmp_path, monkeypatch, _flat(flat))
    # Assert : pas d'acceptation à mi-paire
    sp = rec["random"]["sprt"]
    assert sp["decision"] == "cap_reached" and sp["n_episodes"] == 64 and rec["passed"] is False


def test_v2_theta0_reste_l_adversaire_apres_reprise_et_retour_en_arriere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange : un 1er processus évalue (points 2 et 4) ; sa politique a appris
    sa = ScriptV2("PPPPPP")
    monkeypatch.setattr(hook_mod, "play_match", sa)
    probe = Healthy()
    a = make_hook(tmp_path, V2, probe=probe)
    live = bound(a)
    theta0 = {k: v.clone() for k, v in live.state_dict().items()}
    for it in range(1, 5):
        with torch.no_grad():
            for p in live.parameters():
                p.add_(0.05)
        probe.value = it - 1
        a(it)
    state = json.loads(json.dumps(a.state_dict()))
    # Act : nouveau processus (même dossier d'instantanés), état rechargé, politique vivante de poids DIFFÉRENTS
    sb = ScriptV2("PPPPPP")
    monkeypatch.setattr(hook_mod, "play_match", sb)
    b = make_hook(tmp_path, V2, probe=probe)
    b.load_state_dict(state)
    bound(b, seed=99)
    for it in range(5, 7):
        probe.value = it - 1
        b(it)
    # retour en arrière (restauration vers 3) puis rejeu du point 4
    probe.value = 3
    b(4)
    b(5)
    # Assert : dans tous les cas, l'adversaire de (b) a les poids de θ₀
    seen = [w for role, _, w in sa.calls + sb.calls if role == "frozen"]
    assert len(seen) >= 4 and b.rollbacks == 1
    for w in seen:
        assert all(torch.equal(w[k], theta0[k]) for k in theta0)
