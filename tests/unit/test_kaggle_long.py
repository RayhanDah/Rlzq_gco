"""Relais jour/nuit du run long (ADR-014 D6) : `scripts/kaggle_long.py` et `kaggle/train/long_kernel.py`.
Aucun appel réseau : le CLI kaggle n'est jamais lancé (mode à blanc ou fonctions pures)."""

from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from training import CheckpointManager

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kl = _load(ROOT / "scripts" / "kaggle_long.py", "kaggle_long")
RUN = kl.RUN


def _ckpt_state(it: int) -> dict[str, Any]:
    return {"iteration": it, "identity": {"reward_config_hash": "r"}, "train_config_hash": "h", "w": [it] * 4}


def _make_run(root: Path, its: tuple[int, ...] = (100, 200), segments: int = 1) -> tuple[Path, Path, Path]:
    """Faux run en pause : journaux, 2 checkpoints, un instantané d'éval."""
    runs_root, ck_root = root / "runs", root / "checkpoints"
    rd, cd = runs_root / RUN, ck_root / RUN
    rd.mkdir(parents=True)
    (rd / "metrics.jsonl").write_text("".join(json.dumps({"step": i}) + "\n" for i in range(1, 4)), encoding="utf-8")
    (rd / "events.jsonl").write_text(json.dumps({"event": "start"}) + "\n", encoding="utf-8")
    (rd / "segments.jsonl").write_text(
        "".join(json.dumps({"status": "interrupted"}) + "\n" for _ in range(segments)), encoding="utf-8"
    )
    mgr = CheckpointManager(cd, keep=2)
    for it in its:
        mgr.save(_ckpt_state(it))
    snaps = CheckpointManager(cd / "eval_snapshots", keep=5)
    snaps.save(_ckpt_state(0))
    return runs_root, ck_root, root / "build"


def _kaggle_output(out: Path, record: dict[str, Any], new_it: int, extra_metrics: int = 2) -> Path:
    """Sortie Kaggle simulée : le bundle du pack, un checkpoint de plus et des lignes de journal en plus."""
    work = out / "work"
    with tarfile.open(kl.BUILD / "data" / kl.BUNDLE_TAR) as tf:  # BUILD remplacé par la fixture `packed`
        tf.extractall(work, filter="data")
    rd, cd = work / "runs" / RUN, work / "checkpoints" / RUN
    with (rd / "metrics.jsonl").open("a", encoding="utf-8") as f:
        for i in range(extra_metrics):
            f.write(json.dumps({"step": 4 + i}) + "\n")
    (rd / "run_meta_001.json").write_text("{}", encoding="utf-8")
    CheckpointManager(cd, keep=2).save(_ckpt_state(new_it))
    with tarfile.open(out / kl.BUNDLE_TAR, "w") as tf:
        tf.add(rd, arcname=f"runs/{RUN}")
        tf.add(cd, arcname=f"checkpoints/{RUN}")
    rep = {
        "status": "paused",
        "start_sha256": record["sha256"],
        "bundle": {"file": kl.BUNDLE_TAR, "sha256": kl.kr.sha256_file(out / kl.BUNDLE_TAR)},
    }
    (out / "relay-out.json").write_text(json.dumps(rep), encoding="utf-8")
    return out


@pytest.fixture
def packed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict[str, Any], Path]:
    runs_root, ck_root, build = _make_run(tmp_path)
    monkeypatch.setattr(kl, "BUILD", build)
    record = kl.pack(runs_root, ck_root, build / "data", "alice", "c0ffee", 0.0)
    return runs_root, ck_root, record, build


# --------------------------------------------------------------------------------------------------- pack


def test_pack_contient_le_dernier_checkpoint_sain_les_instantanes_et_les_journaux(packed: Any) -> None:
    # Arrange
    _, _, record, build = packed
    data = build / "data"
    # Act
    with tarfile.open(data / kl.BUNDLE_TAR) as tf:
        names = set(tf.getnames())
    manifest = json.loads((data / kl.BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    meta = json.loads((data / "dataset-metadata.json").read_text(encoding="utf-8"))
    # Assert : seul le dernier checkpoint de reprise voyage (moitié moins à téléverser)
    assert f"checkpoints/{RUN}/ckpt_000001_it00000200.pt" in names
    assert f"checkpoints/{RUN}/ckpt_000000_it00000100.pt" not in names
    assert f"checkpoints/{RUN}/eval_snapshots/ckpt_000000_it00000000.pt" in names
    assert f"runs/{RUN}/metrics.jsonl" in names
    assert manifest["iteration"] == record["iteration"] == 200 and manifest["run"] == RUN
    assert manifest["sha256"] == kl.kr.sha256_file(data / kl.BUNDLE_TAR)
    assert meta == {"id": "alice/rlgco-long-s0", "title": "rlgco-long-s0", "licenses": [{"name": "other"}]}
    assert len(record["local_fingerprint"]) == 6  # 3 journaux + 2 checkpoints + 1 instantané


def test_pack_refuse_une_session_en_cours(tmp_path: Path) -> None:
    runs_root, ck_root, build = _make_run(tmp_path, segments=0)  # un start sans segment de fin
    with pytest.raises(kl.LongRelayError, match="en cours"):
        kl.pack(runs_root, ck_root, build / "data", "alice", "c", 0.0)


# ------------------------------------------------------------------------------------------------- kernel


def test_kernel_rendu_compile_sans_secret_et_porte_la_config() -> None:
    # Arrange
    cfg = kl.kernel_config("https://github.com/o/r.git", "abc1234", False, "f" * 64)
    # Act
    src = kl.render_kernel(cfg)
    ns: dict[str, Any] = {"__name__": "rendu"}
    exec(compile(src, "long_kernel.py", "exec"), ns)  # modèle du dépôt, rendu localement
    # Assert
    assert ns["CONFIG"] == cfg
    assert cfg["session_s"] == 12 * 3600 - 600 and cfg["margin_s"] == 45 * 60 and cfg["run"] == "long_s0"
    assert not kl.kr.SECRET_RE.search(src)


def test_config_de_kernel_refuse_un_depot_non_github_et_un_run_sans_pack() -> None:
    with pytest.raises(kl.LongRelayError, match="GitHub"):
        kl.kernel_config("git@github.com:o/r.git", "abc", False, "f" * 64)
    with pytest.raises(kl.LongRelayError, match="pack"):
        kl.kernel_config("https://github.com/o/r.git", "abc", False, None)
    smoke = kl.kernel_config("https://github.com/o/r.git", "abc", True, None)
    assert smoke["smoke"] is True and smoke["session_s"] == 30 * 60


def test_metadonnees_du_kernel() -> None:
    run, smoke = kl.kernel_metadata("alice", False), kl.kernel_metadata("alice", True)
    assert run["id"] == "alice/rlgco-long-s0-t4" and run["title"] == "rlgco-long-s0-t4"
    assert run["dataset_sources"] == ["alice/rlgco-long-s0"] and smoke["dataset_sources"] == []
    for m in (run, smoke):
        assert m["machine_shape"] == "NvidiaTeslaT4" and m["enable_gpu"] == "true"
        assert m["is_private"] == "true" and m["enable_internet"] == "true" and m["code_file"] == "long_kernel.py"


def test_start_a_blanc_ne_lance_aucune_commande(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange : CLI à blanc (executor None) : la commande est affichée, jamais exécutée
    shown: list[str] = []
    cli = kl.kr.KaggleCli(None, echo=shown.append)
    folder = kl.build_kernel(
        tmp_path / "k",
        kl.kernel_config("https://github.com/o/r.git", "abc", True, None),
        kl.kernel_metadata("alice", True),
    )
    # Act
    out = cli.kernels_push(folder, kl.MACHINE, 1800)
    # Assert
    assert out is None and len(shown) == 1 and shown[0].startswith("(à blanc) kaggle kernels push")
    assert (folder / "long_kernel.py").is_file() and (folder / "kernel-metadata.json").is_file()


# --------------------------------------------------------------------------------------------------- sync


def test_sync_a_blanc_n_ecrit_rien(packed: Any, tmp_path: Path) -> None:
    # Arrange
    runs_root, ck_root, record, build = packed
    out = _kaggle_output(kl.kr.fresh_dir(build / "out" / "x"), record, 300)
    before = kl.fingerprint(runs_root / RUN, ck_root / RUN)
    # Act
    summary = kl.sync(out, runs_root, ck_root, record, apply=False, stamp="T", echo=lambda _: None)
    # Assert
    assert summary["iteration_after"] == 300 and summary["applied"] is False
    assert kl.fingerprint(runs_root / RUN, ck_root / RUN) == before
    assert not (build / "backup").exists()


def test_sync_applique_sauvegarde_et_ne_supprime_rien(packed: Any) -> None:
    # Arrange
    runs_root, ck_root, record, build = packed
    out = _kaggle_output(kl.kr.fresh_dir(build / "out" / "x"), record, 300)
    old_metrics = (runs_root / RUN / "metrics.jsonl").read_text(encoding="utf-8")
    old_bytes = (runs_root / RUN / "metrics.jsonl").read_bytes()
    # Act
    summary = kl.sync(out, runs_root, ck_root, record, apply=True, stamp="T", echo=lambda _: None)
    # Assert : journaux remplacés par la version plus longue, nouveau checkpoint ajouté, anciens gardés, sauvegarde
    new_metrics = (runs_root / RUN / "metrics.jsonl").read_text(encoding="utf-8")
    assert new_metrics.startswith(old_metrics) and new_metrics.count("\n") == 5
    assert (runs_root / RUN / "run_meta_001.json").is_file()
    names = {p.name for p in (ck_root / RUN).glob("ckpt_*.pt")}
    assert {"ckpt_000000_it00000100.pt", "ckpt_000001_it00000200.pt", "ckpt_000002_it00000300.pt"} <= names
    latest = CheckpointManager(ck_root / RUN).load_latest(healthy_only=True)
    assert latest is not None and latest[1]["iteration"] == 300
    backup = Path(summary["backup"])
    with tarfile.open(backup) as tf:
        member = tf.extractfile(f"runs/{RUN}/metrics.jsonl")
        assert member is not None and member.read() == old_bytes


def test_sync_refuse_si_l_etat_local_a_change_depuis_le_pack(packed: Any) -> None:
    runs_root, ck_root, record, build = packed
    out = _kaggle_output(kl.kr.fresh_dir(build / "out" / "x"), record, 300)
    with (runs_root / RUN / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"step": 4}) + "\n")  # une session locale a tourné après le pack
    with pytest.raises(kl.LongRelayError, match="changé depuis `pack`"):
        kl.sync(out, runs_root, ck_root, record, apply=True, stamp="T", echo=lambda _: None)


def test_sync_refuse_un_run_qui_recule(packed: Any) -> None:
    runs_root, ck_root, record, build = packed
    out = _kaggle_output(kl.kr.fresh_dir(build / "out" / "x"), record, 150)
    # le checkpoint 150 a l'index le plus haut : c'est lui que la reprise choisirait
    with pytest.raises(kl.LongRelayError, match="reculé"):
        kl.sync(out, runs_root, ck_root, record, apply=True, stamp="T", echo=lambda _: None)


def test_sync_refuse_une_archive_alteree(packed: Any) -> None:
    runs_root, ck_root, record, build = packed
    out = _kaggle_output(kl.kr.fresh_dir(build / "out" / "x"), record, 300)
    with (out / kl.BUNDLE_TAR).open("ab") as f:
        f.write(b"\0")
    with pytest.raises(kl.LongRelayError, match="empreinte"):
        kl.sync(out, runs_root, ck_root, record, apply=True, stamp="T", echo=lambda _: None)


def test_sync_refuse_un_checkpoint_de_meme_nom_et_autre_contenu(packed: Any) -> None:
    # Arrange : la sortie contient un checkpoint qui porte le nom d'un checkpoint local mais pas son contenu
    runs_root, ck_root, record, build = packed
    out = kl.kr.fresh_dir(build / "out" / "x")
    work = out / "work"
    rd, cd = work / "runs" / RUN, work / "checkpoints" / RUN
    rd.mkdir(parents=True)
    (rd / "metrics.jsonl").write_text("", encoding="utf-8")
    CheckpointManager(cd, keep=2).save(_ckpt_state(999))  # ckpt_000000_it00000999.pt
    CheckpointManager(cd, keep=2).save(_ckpt_state(1000))
    (cd / "ckpt_000000_it00000999.pt").rename(cd / "ckpt_000000_it00000100.pt")  # même nom qu'en local
    with tarfile.open(out / kl.BUNDLE_TAR, "w") as tf:
        tf.add(rd, arcname=f"runs/{RUN}")
        tf.add(cd, arcname=f"checkpoints/{RUN}")
    rep = {
        "start_sha256": record["sha256"],
        "bundle": {"file": kl.BUNDLE_TAR, "sha256": kl.kr.sha256_file(out / kl.BUNDLE_TAR)},
    }
    (out / "relay-out.json").write_text(json.dumps(rep), encoding="utf-8")
    before = kl.fingerprint(runs_root / RUN, ck_root / RUN)
    # Act / Assert
    with pytest.raises(kl.LongRelayError, match="autre contenu"):
        kl.sync(out, runs_root, ck_root, record, apply=True, stamp="T", echo=lambda _: None)
    assert kl.fingerprint(runs_root / RUN, ck_root / RUN) == before


def test_sync_refuse_une_sortie_sans_archive(packed: Any) -> None:
    runs_root, ck_root, record, build = packed
    out = kl.kr.fresh_dir(build / "out" / "x")
    (out / "relay-out.json").write_text(json.dumps({"status": "error", "error": "build", "bundle": None}), "utf-8")
    with pytest.raises(kl.LongRelayError, match="aucune archive"):
        kl.sync(out, runs_root, ck_root, record, apply=True, stamp="T", echo=lambda _: None)


def test_sync_refuse_une_sortie_d_un_autre_pack(packed: Any) -> None:
    # Arrange : sortie valide, mais partie d'un autre bundle (ancienne session)
    runs_root, ck_root, record, build = packed
    out = _kaggle_output(kl.kr.fresh_dir(build / "out" / "x"), record, 300)
    rep = json.loads((out / "relay-out.json").read_text(encoding="utf-8"))
    rep["start_sha256"] = "0" * 64
    (out / "relay-out.json").write_text(json.dumps(rep), encoding="utf-8")
    # Act / Assert
    with pytest.raises(kl.LongRelayError, match="ne part pas du pack"):
        kl.sync(out, runs_root, ck_root, record, apply=True, stamp="T", echo=lambda _: None)


def test_pack_accepte_une_session_tuee_si_l_utilisateur_l_atteste(tmp_path: Path) -> None:
    runs_root, ck_root, build = _make_run(tmp_path, segments=0)
    record = kl.pack(runs_root, ck_root, build / "data", "alice", "c", 0.0, assume_stopped=True)
    assert record["iteration"] == 200


def test_config_d_essai_de_reprise_courte_et_plafonnee() -> None:
    cfg = kl.kernel_config("https://github.com/o/r.git", "abc", False, "f" * 64, max_iterations=23_594)
    assert cfg["max_iterations"] == 23_594 and cfg["session_s"] == 30 * 60 and cfg["margin_s"] == 5 * 60
    day = kl.kernel_config("https://github.com/o/r.git", "abc", False, "f" * 64)
    assert day["max_iterations"] is None and day["session_s"] == 12 * 3600 - 600


# ------------------------------------------------------------------------------------- restauration (kernel)


def _kernel(tmp_path: Path, record: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    k = _load(ROOT / "kaggle" / "train" / "long_kernel.py", "long_kernel_test")
    monkeypatch.setattr(k, "INPUT", tmp_path / "input")
    monkeypatch.setattr(k, "RELAY", tmp_path / "relay")
    monkeypatch.setattr(k, "WORK", tmp_path / "work")
    monkeypatch.setattr(k, "CONFIG", {"run": RUN, "bundle_sha256": record["sha256"]})
    return k


def _mount(build: Path, input_dir: Path, extracted: bool) -> Path:
    """Dataset monté : tel quel (bundle.tar) ou décompressé par Kaggle (dossier bundle/)."""
    ds = input_dir / "rlgco-long-s0"
    ds.mkdir(parents=True)
    (ds / kl.BUNDLE_MANIFEST).write_bytes((build / "data" / kl.BUNDLE_MANIFEST).read_bytes())
    if extracted:
        with tarfile.open(build / "data" / kl.BUNDLE_TAR) as tf:
            tf.extractall(ds / "bundle", filter="data")
    else:
        (ds / kl.BUNDLE_TAR).write_bytes((build / "data" / kl.BUNDLE_TAR).read_bytes())
    return ds


@pytest.mark.parametrize("extracted", [False, True], ids=["archive", "decompressee"])
def test_kernel_restaure_le_bundle_archive_ou_decompresse(
    packed: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extracted: bool
) -> None:
    # Arrange
    _, _, record, build = packed
    k = _kernel(tmp_path, record, monkeypatch)
    _mount(build, tmp_path / "input", extracted)
    # Act
    manifest = k.restore_bundle()
    # Assert
    assert manifest["iteration"] == 200 and k.REPORT["start_sha256"] == record["sha256"]
    assert (tmp_path / "relay" / "checkpoints" / RUN / "ckpt_000001_it00000200.pt").is_file()
    assert (tmp_path / "relay" / "runs" / RUN / "metrics.jsonl").is_file()
    latest = CheckpointManager(tmp_path / "relay" / "checkpoints" / RUN).load_latest(healthy_only=True)
    assert latest is not None and latest[1]["iteration"] == 200


def test_kernel_refuse_un_fichier_decompresse_altere(
    packed: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, record, build = packed
    k = _kernel(tmp_path, record, monkeypatch)
    ds = _mount(build, tmp_path / "input", True)
    with (ds / "bundle" / "runs" / RUN / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write("x" + chr(10))
    with pytest.raises(SystemExit, match="empreinte"):
        k.restore_bundle()


def test_kernel_refuse_un_manifest_d_un_autre_pack(
    packed: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, record, build = packed
    k = _kernel(tmp_path, record, monkeypatch)
    _mount(build, tmp_path / "input", False)
    monkeypatch.setattr(k, "CONFIG", {"run": RUN, "bundle_sha256": "0" * 64})
    with pytest.raises(SystemExit, match="périmée"):
        k.restore_bundle()
