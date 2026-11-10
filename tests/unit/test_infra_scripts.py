"""Tests des scripts d'infra Phase 2 : meshes, benchmark sim, sonde Kaggle."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(rel: str) -> ModuleType:
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


fetch = load("scripts/fetch_meshes.py")
bench = load("scripts/benchmark_sim.py")
probe = load("scripts/kaggle_probe.py")


def test_git_blob_sha_matches_git() -> None:
    # Valeurs de référence : `git hash-object` d'un fichier vide et de "hello\n".
    assert fetch.git_blob_sha(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert fetch.git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_aggregate_uses_slowest_worker() -> None:
    res = [
        {"env_steps": 100, "elapsed_s": 1.0, "goals": 1, "resets": 1, "min_car_z": 17.0},
        {"env_steps": 300, "elapsed_s": 2.0, "goals": 0, "resets": 0, "min_car_z": 15.0},
    ]
    row = bench.aggregate(res, tick_skip=8)
    assert row["env_steps"] == 400
    assert row["env_sps"] == 200.0
    assert row["agent_sps"] == 400.0
    assert row["ticks_per_s"] == 1600.0
    assert row["goals"] == 1
    assert row["min_car_z"] == 15.0


def test_aggregate_uses_union_window_when_workers_are_staggered() -> None:
    # Régression (revue 2026-11-10) : deux workers de 1 s décalés de 1 s ne tournent jamais en même temps ;
    # le débit concurrent doit diviser par la fenêtre réelle (2 s), pas par la durée d'un worker (1 s).
    res = [
        {"env_steps": 100, "elapsed_s": 1.0, "t_start": 10.0, "t_end": 11.0, "goals": 0, "min_car_z": 17.0},
        {"env_steps": 100, "elapsed_s": 1.0, "t_start": 11.0, "t_end": 12.0, "goals": 0, "min_car_z": 17.0},
    ]
    row = bench.aggregate(res, tick_skip=8)
    assert row["wall_s"] == 2.0
    assert row["env_sps"] == 100.0


@pytest.mark.skipif(not (ROOT / "vendor/collision_meshes/soccar").is_dir(), reason="meshes absents")
def test_worker_runs_and_cars_stay_above_floor() -> None:
    pytest.importorskip("RocketSim")
    for engine in ("step", "multi_step"):
        r = bench.worker((0, 0.5, 2, 8, 300.0, str(ROOT / "vendor/collision_meshes"), engine))
        assert r["env_steps"] > 0
        assert r["min_car_z"] > 0  # sous le sol ⇒ meshes non chargés


def test_worker_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError):
        bench.worker((0, 0.1, 1, 8, 300.0, "x", "gpu"))


def test_probe_metadata_t4_and_tpu() -> None:
    t4 = probe.metadata("alice-1", "t4")
    tpu = probe.metadata("alice-1", "tpu")
    assert t4["id"] == "alice-1/rlgco-probe-t4"
    assert t4["machine_shape"] == "NvidiaTeslaT4" and t4["enable_gpu"] == "true"
    assert tpu["machine_shape"] == "TpuV5E8" and tpu["enable_gpu"] == "false"
    assert tpu["enable_tpu"] == "true" and t4["enable_tpu"] == "false"
    assert t4["is_private"] == "true" and t4["enable_internet"] == "true"
    for bad in ("Alice", "a/b", "", "x" * 60):
        with pytest.raises(ValueError):
            probe.metadata(bad, "t4")


def test_probe_kernel_renders_and_compiles() -> None:
    src = probe.render_kernel("abc1234", "TpuV5E8")
    compile(src, "probe_kernel.py", "exec")
    assert '"abc1234"' in src and '"TpuV5E8"' in src
    line = next(s for s in src.splitlines() if s.startswith("EMBEDDED: dict[str, str] = "))
    embedded, _ = json.JSONDecoder().raw_decode(line.split(" = ", 1)[1])
    assert set(embedded) == set(probe.EMBED)
    assert "def worker" in embedded["scripts/benchmark_sim.py"]


def test_probe_kernel_has_no_secret_reference() -> None:
    src = probe.render_kernel("abc1234", "NvidiaTeslaT4").lower()
    for needle in ("access_token", "kaggle.json", "kaggle_api_token", "kaggle_key"):
        assert needle not in src


@pytest.mark.skipif(
    not (Path(bench.RG_PY_DIR).is_dir() and (ROOT / "vendor/collision_meshes/soccar").is_dir()),
    reason="build RocketGPU local ou meshes absents",
)
def test_rocketgpu_worker_runs_and_cars_stay_above_floor() -> None:
    r = bench.worker((0, 0.5, 8, 8, 300.0, str(ROOT / "vendor/collision_meshes"), "rocketgpu"))
    assert r["env_steps"] > 0 and r["env_steps"] % 8 == 0
    assert r["min_car_z"] > 0


build_rg = load("scripts/build_rocketgpu.py")


def test_rocketgpu_cmake_args_cpu_pgo_windows() -> None:
    args = build_rg.cmake_args(
        Path("src"), Path("b"), variant="cpu", pgo="USE", profile=Path("p/q"), portable=False, python="py", windows=True
    )
    assert "-DROCKETGPU_USE_CUDA=OFF" in args and "-DROCKETGPU_NATIVE_ARCH=ON" in args
    assert "-DCMAKE_CXX_COMPILER=g++" in args
    assert "-DROCKETGPU_PGO=USE" in args and "-DROCKETGPU_PGO_PROFILE=p/q" in args
    assert "-DROCKETGPU_BUILD_PYTHON=ON" in args and "-DPython_EXECUTABLE=py" in args


def test_rocketgpu_cmake_args_cuda_linux_portable() -> None:
    args = build_rg.cmake_args(
        Path("src"), Path("b"), variant="cuda", pgo="OFF", profile=None, portable=True, python="py", windows=False
    )
    assert "-DROCKETGPU_USE_CUDA=ON" in args and "-DROCKETGPU_NATIVE_ARCH=OFF" in args
    assert not any(a.startswith("-DROCKETGPU_PGO") for a in args)
    assert "-DCMAKE_CXX_COMPILER=g++" not in args


@pytest.mark.parametrize(
    ("variant", "pgo", "profile"),
    [("gpu", "OFF", None), ("cuda", "USE", Path("p")), ("cpu", "GENERATE", None), ("cpu", "ON", Path("p"))],
)
def test_rocketgpu_cmake_args_rejects_bad_combinations(variant: str, pgo: str, profile: Path | None) -> None:
    with pytest.raises(ValueError):
        build_rg.cmake_args(
            Path("s"), Path("b"), variant=variant, pgo=pgo, profile=profile, portable=False, python="py", windows=False
        )


def test_pinned_shas_are_full_length() -> None:
    assert len(build_rg.PINNED_SHA) == 40 and len(fetch.PINNED_SHA) == 40


def test_rocketgpu_cmake_args_cuda_compiler_only_for_cuda() -> None:
    kw = {
        "pgo": "OFF",
        "profile": None,
        "portable": False,
        "python": "py",
        "windows": False,
        "cuda_compiler": "/c/nvcc",
    }
    assert "-DCMAKE_CUDA_COMPILER=/c/nvcc" in build_rg.cmake_args(Path("s"), Path("b"), variant="cuda", **kw)
    assert not any("CUDA_COMPILER" in a for a in build_rg.cmake_args(Path("s"), Path("b"), variant="cpu", **kw))


def test_has_profile_data_requires_non_empty_file(tmp_path: Path) -> None:
    assert not build_rg.has_profile_data(tmp_path)
    (tmp_path / "a.gcda").write_bytes(b"")
    assert not build_rg.has_profile_data(tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.gcda").write_bytes(b"x")
    assert build_rg.has_profile_data(tmp_path)


@pytest.mark.skipif(not Path(bench.RG_PY_DIR).is_dir(), reason="build RocketGPU local absent")
def test_rocketgpu_controls_layout_matches_cuda_path() -> None:
    # Détecte un changement de layout CarControls en amont, dont dépend le chemin CUDA (vue float32/uint8).
    _, views = bench._import_rocketgpu()
    dt = views.state_dtype("controls")
    assert dt.itemsize == 24
    assert {n: dt.fields[n][1] for n in dt.names} == bench.CONTROL_OFFSETS


def test_probe_colab_variant_compiles_and_detects_work_dir() -> None:
    src = probe.render_kernel("abc1234", "colab")
    compile(src, "rlgco_probe.py", "exec")
    assert 'Path("/content")' in src and 'Path("/kaggle/working")' in src
    assert '"colab"' in src


def test_probe_parts_put_gpu_first_and_validate() -> None:
    ns: dict[str, object] = {"__name__": "probe_under_test"}
    exec(compile(probe.render_kernel("abc1234", "colab"), "rlgco_probe.py", "exec"), ns)
    parse = ns["parse_parts"]
    assert parse([], True) == (["rg_cuda", "rocketsim", "rg_cpu"], 40 * 60)  # type: ignore[operator]
    assert parse([], False) == (["rocketsim", "rg_cpu"], 40 * 60)  # type: ignore[operator]
    assert parse(["--parts", "rg_cuda", "--budget-min", "55"], True) == (["rg_cuda"], 55 * 60)  # type: ignore[operator]
    with pytest.raises(SystemExit):
        parse(["--parts", "gpu"], True)  # type: ignore[operator]


def test_probe_error_lines_keep_compiler_errors_before_traceback() -> None:
    ns: dict[str, object] = {"__name__": "probe_under_test"}
    exec(compile(probe.render_kernel("abc1234", "colab"), "rlgco_probe.py", "exec"), ns)
    log = "[1/9] Building CUDA object x.cu.o\nx.cu(12): error: identifier foo is undefined\nok\nCMake Error at y\n"
    log += "Traceback (most recent call last):\nsubprocess.CalledProcessError: exit status 1\n"
    lines = ns["error_lines"](log)  # type: ignore[operator]
    assert lines[0] == "x.cu(12): error: identifier foo is undefined"
    assert "CMake Error at y" in lines
