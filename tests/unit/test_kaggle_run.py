"""Tests du lanceur de relais Kaggle (scripts/kaggle_run.py) et du kernel squelette (kaggle/train/).

Aucun réseau, aucun appel au CLI kaggle : un faux exécuteur simule le dataset de relais (dernière version visible) et
exécute le kernel rendu en local, avec /kaggle/input et /kaggle/working redirigés vers tmp_path.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kr = load(ROOT / "scripts" / "kaggle_run.py", "kaggle_run")

DATASET = "owner1/rlgco-relay-gco-a"
HEAD = "a" * 40
T0 = 1_800_000_000.0


class FakeGit:
    def __init__(self, dirty: str = "", head: str = HEAD, main: str = HEAD) -> None:
        self.dirty, self.head, self.main = dirty, head, main
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(args)
        if args[0] == "status":
            return self.dirty
        if args[0] == "fetch":
            return ""
        if args == ["rev-parse", "HEAD"]:
            return self.head
        if args == ["rev-parse", "origin/main"]:
            return self.main
        raise AssertionError(f"git inattendu : {args}")


class FakeKaggle:
    """Simule le CLI : un dataset (fichiers de la dernière version), des kernels poussés et leurs sorties."""

    def __init__(self, tmp: Path, status: str = "KernelWorkerStatus.COMPLETE", run_kernels: bool = True) -> None:
        self.tmp = tmp
        self.remote: dict[str, bytes] = {}
        self.versions: list[dict[str, bytes]] = []
        self.outputs: dict[str, Path] = {}
        self.status = status
        self.run_kernels = run_kernels
        self.calls: list[list[str]] = []
        self.on_version: Any = None  # crochet : simule l'écriture concurrente d'un autre membre
        self.on_full_download: Any = None  # crochet : un autre membre publie juste après notre lecture complète
        self.soft_error: tuple[str, str] | None = None  # refus serveur affiché avec le code 0 (CLI 2.2.5)

    def opt(self, cmd: list[str], flag: str) -> str | None:
        return cmd[cmd.index(flag) + 1] if flag in cmd else None

    def __call__(self, cmd: list[str]) -> Any:
        self.calls.append(cmd)
        group, sub = cmd[1], cmd[2]
        if self.soft_error == (group, sub):
            msg = {"push": "Kernel push error: refus", "version": "Dataset version creation error: refus"}[sub]
            return kr.CmdResult(0, msg + "\n")
        if (group, sub) == ("datasets", "create") or (group, sub) == ("datasets", "version"):
            folder = Path(str(self.opt(cmd, "-p")))
            meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
            assert meta["id"] == DATASET
            files = {p.name: p.read_bytes() for p in folder.iterdir() if p.name != "dataset-metadata.json"}
            self.remote = files
            self.versions.append(files)
            if self.on_version:
                self.on_version(self)
        elif (group, sub) == ("datasets", "download"):
            dest = Path(str(self.opt(cmd, "-p")))
            dest.mkdir(parents=True, exist_ok=True)
            wanted = self.opt(cmd, "-f")
            for name, data in self.remote.items():
                if wanted is None or name == wanted:
                    (dest / name).write_bytes(data)
            if wanted is None and self.on_full_download:
                self.on_full_download(self)
        elif (group, sub) == ("kernels", "push"):
            if self.run_kernels:
                self.run_kernel(Path(str(self.opt(cmd, "-p"))))
        elif (group, sub) == ("kernels", "status"):
            return kr.CmdResult(0, f'{cmd[3]} has status "{self.status}"\n')
        elif (group, sub) == ("kernels", "output"):
            dest = Path(str(self.opt(cmd, "-p")))
            for p in self.outputs[cmd[3]].iterdir():
                shutil.copyfile(p, dest / p.name)
        else:
            raise AssertionError(f"commande inattendue : {cmd}")
        return kr.CmdResult(0, "")

    def run_kernel(self, folder: Path) -> None:
        meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
        kid = meta["id"]
        mod = load(folder / meta["code_file"], f"kernel_{len(self.outputs)}_{len(self.calls)}")
        inp = self.tmp / "kin" / kid.replace("/", "_") / f"{len(self.calls)}"
        mount = inp / meta["dataset_sources"][0].split("/")[1]
        mount.mkdir(parents=True)
        for name, data in self.remote.items():  # montage de la dernière version visible
            (mount / name).write_bytes(data)
        work = self.tmp / "kwork" / kid.replace("/", "_") / f"{len(self.calls)}"
        mod.INPUT, mod.WORK = inp, work
        assert mod.main() == 0
        self.outputs[kid] = work


def run(fake: FakeKaggle | None, tmp: Path, *args: str, git: FakeGit | None = None, now: float = T0) -> int:
    return kr.main(
        list(args),
        executor=fake,
        git=git or FakeGit(),
        now=lambda: now,
        sleep=lambda s: None,
        work_root=tmp / "build",
    )


def common(cmd: str, user: str, *extra: str) -> list[str]:
    return [cmd, "--user", user, "--run", "gco-a", "--acc", "t4", "--dataset", DATASET, *extra]


DL = ["datasets", "download"]
VERSION = ["datasets", "version"]


class _AnyMsg(str):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and bool(other)

    __hash__ = str.__hash__


ANY_MSG = _AnyMsg("<message>")


def lease(user: str | None, expires: float, released: bool = False, nonce: str = "n0") -> dict[str, Any]:
    return {
        "schema": 1,
        "run_id": "gco-a",
        "user": user,
        "kernel": f"{user}/rlgco-train-gco-a-t4",
        "commit": HEAD,
        "accelerator": "NvidiaTeslaT4",
        "nonce": nonce,
        "expires_at": kr.iso(expires),
        "expires_at_unix": int(expires),
        "released": released,
    }


def seeded(tmp: Path, lease_obj: dict[str, Any] | None = None) -> FakeKaggle:
    fake = FakeKaggle(tmp)
    assert run(fake, tmp, *common("init", "owner1", "--push")) == 0
    if lease_obj is not None:
        fake.remote["lease.json"] = json.dumps(lease_obj).encode()
    fake.calls.clear()
    return fake


def remote_json(fake: FakeKaggle, name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(fake.remote[name])
    return data


# --- préconditions git ---------------------------------------------------------------------------------------------


def test_git_dirty_tree_refused() -> None:
    with pytest.raises(kr.RelayError, match="arbre git modifié"):
        kr.check_git(FakeGit(dirty=" M scripts/x.py"), fetch=False)


def test_git_head_must_equal_origin_main() -> None:
    with pytest.raises(kr.RelayError, match="origin/main"):
        kr.check_git(FakeGit(main="b" * 40), fetch=False)


def test_git_ok_returns_full_commit_and_fetches_only_when_asked() -> None:
    git = FakeGit()
    assert kr.check_git(git, fetch=False) == HEAD
    assert ["fetch", "--quiet", "origin", "main"] not in git.calls
    git = FakeGit()
    kr.check_git(git, fetch=True)
    assert git.calls[1] == ["fetch", "--quiet", "origin", "main"]


def test_start_refused_on_dirty_tree_before_any_kaggle_call(tmp_path: Path) -> None:
    fake = seeded(tmp_path)
    assert run(fake, tmp_path, *common("start", "alice", "--push"), git=FakeGit(dirty=" M a.py")) == 1
    assert fake.calls == []


# --- verrou --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lease_obj", "user", "expected"),
    [
        (None, "alice", "free"),
        (lease(None, T0 + 10, released=True), "alice", "free"),
        (lease("bob", T0 + 3600, released=True), "alice", "free"),
        (lease("bob", T0 + 3600), "alice", "other"),
        (lease("alice", T0 + 3600), "alice", "mine"),
        (lease("bob", T0 - 1), "alice", "expired"),
        (lease("alice", T0), "alice", "expired"),
    ],
)
def test_lease_state(lease_obj: dict[str, Any] | None, user: str, expected: str) -> None:
    assert kr.lease_state(lease_obj, user, T0) == expected


def test_new_lease_covers_queue_session_and_margin() -> None:
    t4 = kr.new_lease("gco-a", "alice", "alice/k", HEAD, "t4", 12 * 3600, T0, "n1")
    assert t4["expires_at_unix"] == int(T0 + 12 * 3600 + kr.LEASE_MARGIN_S)
    tpu = kr.new_lease("gco-a", "alice", "alice/k", HEAD, "tpu", 9 * 3600, T0, "n1")
    assert tpu["expires_at_unix"] == int(T0 + 40 * 3600 + 9 * 3600 + kr.LEASE_MARGIN_S)
    assert t4["commit"] == HEAD and t4["run_id"] == "gco-a" and t4["kernel"] == "alice/k"


def test_start_refused_when_other_member_holds_active_lease(tmp_path: Path) -> None:
    fake = seeded(tmp_path, lease("bob", T0 + 3600))
    assert run(fake, tmp_path, *common("start", "alice", "--push")) == 1
    assert [c[1:3] for c in fake.calls] == [["datasets", "download"]]  # lecture seule, ni version ni push


def test_start_refused_when_we_already_hold_lease(tmp_path: Path) -> None:
    fake = seeded(tmp_path, lease("alice", T0 + 3600))
    assert run(fake, tmp_path, *common("start", "alice", "--push")) == 1
    assert [c[1:3] for c in fake.calls] == [["datasets", "download"]]


def test_start_takes_over_expired_lease(tmp_path: Path) -> None:
    fake = seeded(tmp_path, lease("bob", T0 - 1))
    fake.run_kernels = False
    n_versions = len(fake.versions)
    assert run(fake, tmp_path, *common("start", "alice", "--push")) == 0
    got = remote_json(fake, "lease.json")
    session = json.loads((tmp_path / "build/gco-a/start/session.json").read_text(encoding="utf-8"))
    assert got["user"] == "alice" and not got["released"] and got["nonce"] == session["nonce"]
    assert got["expires_at_unix"] == int(T0 + 12 * 3600 + kr.LEASE_MARGIN_S)
    assert len(fake.versions) == n_versions + 1
    assert [c[1:3] for c in fake.calls][-1] == ["kernels", "push"]


def test_start_aborts_without_push_if_lease_stolen_between_write_and_read(tmp_path: Path) -> None:
    fake = seeded(tmp_path)

    def bob_writes_after_us(f: FakeKaggle) -> None:
        f.remote["lease.json"] = json.dumps(lease("bob", T0 + 3600, nonce="bob")).encode()
        f.on_version = None

    fake.on_version = bob_writes_after_us
    assert run(fake, tmp_path, *common("start", "alice", "--push")) == 1
    assert ["kernels", "push"] not in [c[1:3] for c in fake.calls]


def test_start_aborts_if_our_version_never_becomes_visible(tmp_path: Path) -> None:
    fake = seeded(tmp_path)
    before = dict(fake.remote)

    def not_yet_visible(f: FakeKaggle) -> None:
        f.remote = dict(before)

    fake.on_version = not_yet_visible
    assert run(fake, tmp_path, *common("start", "alice", "--push")) == 1
    checks = [c for c in fake.calls if c[-2:] == ["-f", "lease.json"] and c[5].endswith("dataset_lease_check")]
    assert len(checks) == kr.VERIFY_TRIES
    assert ["kernels", "push"] not in [c[1:3] for c in fake.calls]


def test_release_frees_our_lease_and_keeps_checkpoint_files(tmp_path: Path) -> None:
    fake = seeded(tmp_path, lease("alice", T0 + 3600))
    fake.remote["manifest.json"] = json.dumps({**kr.empty_manifest("gco-a"), "checkpoint": "c.json"}).encode()
    fake.remote["c.json"] = b"{}"
    manifest = remote_json(fake, "manifest.json")
    manifest["sha256"] = kr.hashlib.sha256(b"{}").hexdigest()
    fake.remote["manifest.json"] = json.dumps(manifest).encode()
    assert run(fake, tmp_path, *common("release", "alice", "--push")) == 0
    got = remote_json(fake, "lease.json")
    assert got["released"] and got["released_by"] == "alice"
    assert fake.remote["c.json"] == b"{}"


def test_release_refused_for_other_members_lease(tmp_path: Path) -> None:
    fake = seeded(tmp_path, lease("bob", T0 + 3600))
    assert run(fake, tmp_path, *common("release", "alice", "--push")) == 1
    assert ["datasets", "version"] not in [c[1:3] for c in fake.calls]


def test_finish_refused_when_lease_is_not_ours(tmp_path: Path) -> None:
    fake = seeded(tmp_path, lease("bob", T0 + 3600))
    assert run(fake, tmp_path, *common("finish", "alice", "--push")) == 1
    assert ["datasets", "version"] not in [c[1:3] for c in fake.calls]


# --- commandes produites et relais complet ---------------------------------------------------------------------------


def test_full_relay_two_members(tmp_path: Path) -> None:
    fake = seeded(tmp_path)
    git = FakeGit()
    assert run(fake, tmp_path, *common("start", "alice", "--push", "--train-seconds", "1"), git=git) == 0
    kdir = tmp_path / "build" / "gco-a" / "start" / "kernel"
    exe = kr.kaggle_exe()
    assert [c[1:3] + c[-2:] for c in fake.calls if c[2] == "download" and "-f" in c] == [
        ["datasets", "download", "-f", "lease.json"],
        ["datasets", "download", "-f", "manifest.json"],
    ] * 2
    assert [c[1:3] for c in fake.calls] == [DL, DL, DL, VERSION, DL, DL, ["kernels", "push"]]
    assert fake.calls[-1] == [exe, "kernels", "push", "-p", str(kdir), "--accelerator", "NvidiaTeslaT4", "-t", "43200"]
    din = str(tmp_path / "build" / "gco-a" / "start" / "dataset_in")
    assert fake.calls[0] == [exe, "datasets", "download", DATASET, "-p", din, "--unzip", "-o"]
    assert fake.calls[3] == [exe, "datasets", "version", "-p", str(kdir.parent / "dataset_lease"), "-m", ANY_MSG]
    assert ["fetch", "--quiet", "origin", "main"] in git.calls
    meta = json.loads((kdir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "alice/rlgco-train-gco-a-t4" and meta["dataset_sources"] == [DATASET]
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["enable_gpu"] == "true"

    fake.calls.clear()
    assert run(fake, tmp_path, *common("finish", "alice", "--push"), now=T0 + 600) == 0
    assert [c[1:3] for c in fake.calls] == [DL, ["kernels", "status"], ["kernels", "output"], DL, DL, VERSION, DL, DL]
    assert fake.calls[1] == [exe, "kernels", "status", "alice/rlgco-train-gco-a-t4"]
    assert fake.calls[2][3:] == ["alice/rlgco-train-gco-a-t4", "-p", str(tmp_path / "build/gco-a/finish/output"), "-o"]
    assert fake.calls[5][-2] == "-m"
    m1 = remote_json(fake, "manifest.json")
    assert len(m1["sessions"]) == 1 and m1["sessions"][0]["user"] == "alice" and m1["sessions"][0]["commit"] == HEAD
    assert m1["checkpoint"] in fake.remote and m1["steps"] > 0
    assert kr.hashlib.sha256(fake.remote[m1["checkpoint"]]).hexdigest() == m1["sha256"]
    assert remote_json(fake, "lease.json")["released"]

    # Bob reprend : son kernel doit lire et vérifier le checkpoint d'Alice.
    assert run(fake, tmp_path, *common("start", "bob", "--push", "--train-seconds", "1"), now=T0 + 700) == 0
    assert run(fake, tmp_path, *common("finish", "bob", "--push"), now=T0 + 1300) == 0
    m2 = remote_json(fake, "manifest.json")
    assert [s["user"] for s in m2["sessions"]] == ["alice", "bob"]
    assert m2["steps"] > m1["steps"]
    ckpt2 = json.loads(fake.remote[m2["checkpoint"]])
    assert ckpt2["parent_sha256"] == m1["sha256"] and ckpt2["session"] == 2
    assert m1["checkpoint"] not in fake.remote  # seul le dernier checkpoint est dans la version courante


def test_finish_refuses_running_kernel_and_needs_salvage_after_error(tmp_path: Path) -> None:
    fake = seeded(tmp_path)
    assert run(fake, tmp_path, *common("start", "alice", "--push", "--train-seconds", "1")) == 0
    fake.status = "KernelWorkerStatus.RUNNING"
    assert run(fake, tmp_path, *common("finish", "alice", "--push")) == 1
    fake.status = "KernelWorkerStatus.ERROR"
    assert run(fake, tmp_path, *common("finish", "alice", "--push")) == 1
    assert run(fake, tmp_path, *common("finish", "alice", "--push", "--salvage")) == 0
    assert remote_json(fake, "manifest.json")["sessions"][0]["kernel_status"] == "error"


def test_finish_rejects_output_with_foreign_nonce(tmp_path: Path) -> None:
    fake = seeded(tmp_path)
    assert run(fake, tmp_path, *common("start", "alice", "--push", "--train-seconds", "1")) == 0
    out = fake.outputs["alice/rlgco-train-gco-a-t4"] / "relay-out.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    data["nonce"] = "autre"
    out.write_text(json.dumps(data), encoding="utf-8")
    n_versions = len(fake.versions)
    assert run(fake, tmp_path, *common("finish", "alice", "--push")) == 1
    assert len(fake.versions) == n_versions


def test_finish_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    fake = seeded(tmp_path)
    assert run(fake, tmp_path, *common("start", "alice", "--push", "--train-seconds", "1")) == 0
    work = fake.outputs["alice/rlgco-train-gco-a-t4"]
    name = json.loads((work / "relay-out.json").read_text(encoding="utf-8"))["checkpoint"]
    (work / name).write_text("{}", encoding="utf-8")
    assert run(fake, tmp_path, *common("finish", "alice", "--push")) == 1


def test_status_queries_kernel_only_when_lease_is_ours(tmp_path: Path) -> None:
    fake = seeded(tmp_path, lease("alice", T0 + 3600))
    assert run(fake, tmp_path, *common("status", "alice", "--push")) == 0
    assert fake.calls[-1][1:4] == ["kernels", "status", "alice/rlgco-train-gco-a-t4"]
    fake.calls.clear()
    assert run(fake, tmp_path, *common("status", "bob", "--push")) == 0
    assert ["kernels", "status"] not in [c[1:3] for c in fake.calls]


def test_init_creates_private_dataset(tmp_path: Path) -> None:
    fake = FakeKaggle(tmp_path)
    assert run(fake, tmp_path, *common("init", "owner1", "--push")) == 0
    assert fake.calls[0][1:3] == ["datasets", "create"] and "-u" not in fake.calls[0]
    meta = json.loads((tmp_path / "build/gco-a/init/dataset_init/dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == DATASET and len(meta["licenses"]) == 1 and 6 <= len(meta["title"]) <= 50
    assert remote_json(fake, "lease.json")["released"]


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ('u/k has status "KernelWorkerStatus.COMPLETE"', "complete"),
        ('u/k has status "complete"', "complete"),
        ('u/k has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"\nFailure message: "x"', "cancel_acknowledged"),
        ('u/k has status "cancelRequested"', "cancel_requested"),
        ("rien", "unknown"),
        ('u/k has status "Bizarre"', "unknown"),
    ],
)
def test_parse_kernel_status(stdout: str, expected: str) -> None:
    assert kr.parse_kernel_status(stdout) == expected


# --- mode à blanc ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["init", "start", "status", "finish", "release"])
def test_dry_run_launches_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    def forbidden(*a: object, **k: object) -> None:
        raise AssertionError("sous-processus lancé en mode à blanc")

    def executor(cmd: list[str]) -> Any:
        raise AssertionError(f"exécuteur appelé en mode à blanc : {cmd}")

    monkeypatch.setattr(kr.subprocess, "run", forbidden)
    git = FakeGit()
    assert run(executor, tmp_path, *common(command, "alice"), git=git) == 0  # type: ignore[arg-type]
    assert ["fetch", "--quiet", "origin", "main"] not in git.calls


def test_dry_run_prints_planned_commands_and_builds_kernel(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(None, tmp_path, *common("start", "alice")) == 0
    out = capsys.readouterr().out
    assert "(à blanc)" in out and "kernels push" in out and "datasets version" in out
    assert (tmp_path / "build/gco-a/start/kernel/train_kernel.py").is_file()


# --- secrets ---------------------------------------------------------------------------------------------------------


def test_rendered_kernel_and_metadata_contain_no_token_pattern(tmp_path: Path) -> None:
    assert run(None, tmp_path, *common("start", "alice")) == 0
    for p in (tmp_path / "build/gco-a/start").rglob("*"):
        if p.is_file():
            assert not kr.SECRET_RE.search(p.read_text(encoding="utf-8")), p
    assert not kr.SECRET_RE.search(kr.TEMPLATE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("leak", ["KAGGLE" + "_API_TOKEN=x", "~/.kag" + "gle/acc" + "ess_token", "KG" + "AT_0123abcd"])
def test_secret_guard_blocks_rendering(leak: str) -> None:
    run_spec = {"run_id": "gco-a", "note": leak}
    with pytest.raises(kr.RelayError, match="motif de jeton"):
        kr.render_kernel(run_spec)


def test_invalid_identifiers_rejected() -> None:
    with pytest.raises(SystemExit):
        kr.parse_args(["start", "--user", "Alice", "--run", "gco-a", "--dataset", DATASET])
    with pytest.raises(SystemExit):
        kr.parse_args(["start", "--user", "alice", "--run", "../x", "--dataset", DATASET])
    with pytest.raises(SystemExit):
        kr.parse_args(["start", "--user", "alice", "--run", "gco-a", "--dataset", "pas-de-slash"])


# --- kernel squelette ------------------------------------------------------------------------------------------------


def test_kernel_refuses_stale_input_checkpoint(tmp_path: Path) -> None:
    src = kr.render_kernel(
        {
            "run_id": "gco-a",
            "session": 2,
            "user": "bob",
            "kernel": "bob/k",
            "commit": HEAD,
            "nonce": "n",
            "accelerator": "NvidiaTeslaT4",
            "expected_sha256": "0" * 64,
            "train_seconds": 1,
            "save_every_s": 1200,
        }
    )
    path = tmp_path / "k.py"
    path.write_text(src, encoding="utf-8")
    mod = load(path, "kernel_stale")
    mount = tmp_path / "input" / "ds"
    mount.mkdir(parents=True)
    (mount / "c.json").write_text("{}", encoding="utf-8")
    (mount / "manifest.json").write_text(json.dumps({"run_id": "gco-a", "checkpoint": "c.json"}), encoding="utf-8")
    mod.INPUT, mod.WORK = tmp_path / "input", tmp_path / "work"
    with pytest.raises(SystemExit, match="inattendu"):
        mod.main()
    assert not (tmp_path / "work" / "relay-out.json").exists()


def test_kernel_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    mod = load(kr.TEMPLATE, "kernel_template")
    target = tmp_path / "x.json"
    mod.atomic_write(target, b"abc")
    mod.atomic_write(target, b"def")
    assert target.read_bytes() == b"def"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


# --- constats de revue (2026-11-10) ----------------------------------------------------------------------------------


def test_soft_error_on_version_is_a_failure_not_a_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Le CLI 2.2.5 affiche « Dataset version creation error » avec le code 0 (kaggle_api_extended.py:5964).
    fake = seeded(tmp_path)
    assert run(fake, tmp_path, *common("start", "alice", "--push", "--train-seconds", "1")) == 0
    before = dict(fake.remote)
    fake.soft_error = ("datasets", "version")
    assert run(fake, tmp_path, *common("finish", "alice", "--push")) == 1
    assert fake.remote == before
    out = capsys.readouterr()
    assert "publiée" not in out.out and "Dataset version creation error" in out.err


def test_push_failure_releases_lease_automatically(tmp_path: Path) -> None:
    fake = seeded(tmp_path)
    fake.soft_error = ("kernels", "push")
    assert run(fake, tmp_path, *common("start", "alice", "--push")) == 1
    got = remote_json(fake, "lease.json")
    assert got["user"] == "alice" and got["released"] and got["release_reason"] == "push en échec"
    assert not (tmp_path / "build/gco-a/start/session.json").exists()


def test_start_refuses_to_overwrite_a_finish_published_after_our_read(tmp_path: Path) -> None:
    fake = seeded(tmp_path)

    def bob_finishes(f: FakeKaggle) -> None:
        manifest = json.loads(f.remote["manifest.json"])
        manifest["sessions"] = [{"session": 1, "user": "bob"}]
        f.remote["manifest.json"] = json.dumps(manifest).encode()
        f.on_full_download = None

    fake.on_full_download = bob_finishes
    n_versions = len(fake.versions)
    assert run(fake, tmp_path, *common("start", "alice", "--push")) == 1
    assert len(fake.versions) == n_versions
    assert ["kernels", "push"] not in [c[1:3] for c in fake.calls]


def test_finish_accepted_on_our_expired_lease_not_taken_over(tmp_path: Path) -> None:
    fake = seeded(tmp_path)
    assert run(fake, tmp_path, *common("start", "alice", "--push", "--train-seconds", "1")) == 0
    late = T0 + 20 * 3600  # au-delà de 12 h + 2 h de marge
    assert kr.lease_state(remote_json(fake, "lease.json"), "alice", late) == "expired"
    assert run(fake, tmp_path, *common("finish", "alice", "--push"), now=late) == 0
    assert remote_json(fake, "manifest.json")["sessions"][0]["user"] == "alice"


def test_finish_refused_after_takeover_of_our_expired_lease(tmp_path: Path) -> None:
    fake = seeded(tmp_path)
    assert run(fake, tmp_path, *common("start", "alice", "--push", "--train-seconds", "1")) == 0
    late = T0 + 20 * 3600
    fake.run_kernels = False
    assert run(fake, tmp_path, *common("start", "bob", "--push"), now=late) == 0
    n_versions = len(fake.versions)
    assert run(fake, tmp_path, *common("finish", "alice", "--push"), now=late + 60) == 1
    assert len(fake.versions) == n_versions


def test_release_of_our_expired_lease(tmp_path: Path) -> None:
    fake = seeded(tmp_path, lease("alice", T0 - 1))
    assert run(fake, tmp_path, *common("release", "alice", "--push")) == 0
    assert remote_json(fake, "lease.json")["released"]


def test_tpu_kernel_metadata_and_timeout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["start", "--user", "alice", "--run", "gco-a", "--acc", "tpu", "--dataset", DATASET]
    assert run(None, tmp_path, *args) == 0
    meta = json.loads((tmp_path / "build/gco-a/start/kernel/kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["enable_tpu"] == "true" and meta["enable_gpu"] == "false" and meta["machine_shape"] == "TpuV5E8"
    assert meta["id"] == "alice/rlgco-train-gco-a-tpu"
    assert "--accelerator TpuV5E8 -t 32400" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("init", ["datasets create"]),
        ("start", ["datasets download", "datasets version", "kernels push"]),
        ("status", ["datasets download", "kernels status"]),
        ("finish", ["datasets download", "kernels status", "kernels output", "datasets version"]),
        ("release", ["datasets download", "datasets version"]),
    ],
)
def test_dry_run_prints_planned_commands_in_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, expected: list[str]
) -> None:
    assert run(None, tmp_path, *common(command, "alice")) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("(à blanc) ")]
    seen = [" ".join(ln.split()[3:5]) for ln in lines]
    it = iter(seen)
    assert all(any(cmd == s for s in it) for cmd in expected), seen  # sous-suite ordonnée


def test_redact_masks_token_like_lines() -> None:
    text = "ok\n" + "KAGGLE" + "_KEY=abc\nfin"
    assert kr.redact(text) == "ok\n[ligne masquée]\nfin"


def test_kernel_refuses_unexpected_checkpoint_on_fresh_run(tmp_path: Path) -> None:
    src = kr.render_kernel(
        {
            "run_id": "gco-a",
            "session": 1,
            "user": "bob",
            "kernel": "bob/k",
            "commit": HEAD,
            "nonce": "n",
            "accelerator": "NvidiaTeslaT4",
            "expected_sha256": None,
            "train_seconds": 1,
            "save_every_s": 1200,
        }
    )
    path = tmp_path / "k.py"
    path.write_text(src, encoding="utf-8")
    mod = load(path, "kernel_fresh")
    mount = tmp_path / "input" / "ds"
    mount.mkdir(parents=True)
    (mount / "manifest.json").write_text(json.dumps({"run_id": "gco-a", "checkpoint": "c.json"}), encoding="utf-8")
    mod.INPUT, mod.WORK = tmp_path / "input", tmp_path / "work"
    with pytest.raises(SystemExit, match="démarre de zéro"):
        mod.main()


def test_kernel_checkpoint_names_unique_within_one_second(tmp_path: Path) -> None:
    src = kr.render_kernel({"run_id": "gco-a", "session": 1, "nonce": "n", "commit": HEAD})
    path = tmp_path / "k.py"
    path.write_text(src, encoding="utf-8")
    mod = load(path, "kernel_names")
    mod.WORK = tmp_path / "work"
    mod.WORK.mkdir()
    mod.KEEP_CKPTS = 10
    for _ in range(3):
        mod.save({"steps": 1}, None)
    names = sorted(p.name for p in mod.WORK.glob("ckpt-*.json"))
    assert len(names) == 3 and all(kr.CKPT_RE.fullmatch(n) for n in names)
    out = json.loads((mod.WORK / "relay-out.json").read_text(encoding="utf-8"))
    assert out["checkpoint"] == names[-1]
