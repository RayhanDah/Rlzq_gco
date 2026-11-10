"""Tests des hooks du studio (stdlib, exécutables sans venv) :
python -m unittest discover -s tests/hooks -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / ".claude" / "hooks"


def run_hook(name: str, payload: dict, env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = {**os.environ, "CLAUDE_PROJECT_DIR": str(ROOT), **(env or {})}
    return subprocess.run(
        [sys.executable, str(HOOKS / name)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=full_env,
        timeout=30,
    )


def shell(cmd: str) -> int:
    return run_hook(
        "guard_shell.py", {"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": str(ROOT)}
    ).returncode


def write(path: str) -> int:
    return run_hook(
        "guard_paths.py", {"tool_name": "Write", "tool_input": {"file_path": path}, "cwd": str(ROOT)}
    ).returncode


class GuardShellTest(unittest.TestCase):
    DENY: ClassVar[list[str]] = [
        "rm -rf checkpoints",
        "rm -r runs/exp1",
        "rm checkpoints/latest.pt",
        "mv checkpoints/a.pt /tmp/",
        "Remove-Item -Recurse -Force data",
        "echo x > runs/log.txt",
        "rm -rf /",
        "rm -rf ~",
        "rm -rf .",
        "git push --force origin main",
        "git push -f",
        "git push origin +main",
        "git reset --hard HEAD~1",
        "git clean -fdx",
        'git commit -m "x" --no-verify',
        'git commit -n -m "x"',
        "git add -f checkpoints/a.pt",
        "git add repos",
        "git add threads/pulsar2.txt",
        "cat ~/.kaggle/access_token",
        "type C:\\Users\\me\\.kaggle\\kaggle.json",
        "echo $KAGGLE_API_TOKEN",
        "printenv",
        "env",
        "Get-ChildItem env:",
        "RLGCO_ALLOW_NO_TESTS=1 git commit -m x",
        "git config core.hooksPath /dev/null",
    ]
    ALLOW: ClassVar[list[str]] = [
        "ls checkpoints",
        "git status",
        'git commit -m "rm -rf checkpoints est interdit"',
        "git commit -m 'docs: --no-verify mentionné'",
        "git push origin main",
        "git push -u origin main",
        "git add docs src tests",
        "python scripts/train.py --out runs/exp1",
        "kaggle kernels push -p kaggle/probe",
        "kaggle kernels status user/probe",
        "env PYTHONHASHSEED=0 python -c 'print(1)'",
        "git config --get core.hooksPath",
        "git config core.hooksPath .githooks",
        "rm -rf build/tmp",
    ]

    def test_deny(self) -> None:
        for cmd in self.DENY:
            with self.subTest(cmd=cmd):
                self.assertEqual(shell(cmd), 2, f"devrait être refusé : {cmd}")

    def test_allow(self) -> None:
        for cmd in self.ALLOW:
            with self.subTest(cmd=cmd):
                self.assertEqual(shell(cmd), 0, f"devrait être accepté : {cmd}")

    def test_malformed_input_fails_closed(self) -> None:
        res = subprocess.run(
            [sys.executable, str(HOOKS / "guard_shell.py")], input="{pas du json", capture_output=True, text=True
        )
        self.assertEqual(res.returncode, 2)


class GuardPathsTest(unittest.TestCase):
    def test_deny(self) -> None:
        for p in [
            str(ROOT / "checkpoints" / "a.pt"),
            "runs/exp/log.txt",
            "data/meshes/x.bin",
            str(ROOT / "repos" / "pulsar2" / "CLAUDE.md"),
            "threads/disco.txt",
            str(Path.home() / ".kaggle" / "access_token"),
            "C:/Users/x/.kaggle/kaggle.json",
            ".env",
        ]:
            with self.subTest(path=p):
                self.assertEqual(write(p), 2)

    def test_allow(self) -> None:
        for p in [
            "src/rewards/gco.py",
            str(ROOT / "docs" / "adr" / "001-x.md"),
            "tests/unit/test_x.py",
            "configs/ppo.yaml",
        ]:
            with self.subTest(path=p):
                self.assertEqual(write(p), 0)


class LogRunsTest(unittest.TestCase):
    def test_logs_only_run_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {"CLAUDE_PROJECT_DIR": tmp}
            for cmd in ["git status", "python scripts/train.py --config configs/a.yaml", "kaggle kernels push -p k"]:
                res = run_hook(
                    "log_runs.py", {"tool_name": "Bash", "tool_input": {"command": cmd}, "session_id": "s"}, env
                )
                self.assertEqual(res.returncode, 0, res.stderr)
            lines = (Path(tmp) / "production" / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("scripts/train.py", json.loads(lines[0])["command"])


class StopHookTest(unittest.TestCase):
    def test_stop_hook_active_is_noop(self) -> None:
        self.assertEqual(run_hook("stop_tests.py", {"stop_hook_active": True}).returncode, 0)


class PrecommitPatternsTest(unittest.TestCase):
    def test_forbidden_files(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import precommit

        for f in [".kaggle/access_token", "x/kaggle.json", "model.onnx", "ck/latest.pt", ".env", ".env.local"]:
            with self.subTest(f=f):
                self.assertIsNotNone(precommit.FORBIDDEN_RE.search(f))
        for f in ["src/env/kaggle_runner.py", "docs/research/kaggle.md", "environment.md"]:
            with self.subTest(f=f):
                self.assertIsNone(precommit.FORBIDDEN_RE.search(f))


if __name__ == "__main__":
    unittest.main()
