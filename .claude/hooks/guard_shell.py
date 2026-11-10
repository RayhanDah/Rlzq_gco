"""PreToolUse Bash|PowerShell : gardes fail-closed sur les commandes shell.

Refuse :
- toute commande qui touche au jeton Kaggle ou dump l'environnement ;
- suppression / déplacement / écrasement dans checkpoints|runs|data|repos|threads ;
- suppression récursive de la racine, du home ou du projet ;
- git push --force, git reset --hard, git clean -f, --no-verify, git add -f, ajout de repos/ ou threads/ ;
- git commit si core.hooksPath n'est pas .githooks (le hook git lance scripts/precommit.py).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PROJECT_DIR, PROTECTED_DIRS, deny, read_input

PROT = "|".join(PROTECTED_DIRS)

SECRET_RE = re.compile(r"\.kaggle\b|kaggle\.json|access_token|KAGGLE_(API_)?(TOKEN|KEY)|RLGCO_ALLOW_NO_TESTS", re.I)
ENV_DUMP_RE = re.compile(
    r"(^|[;&|]\s*)(printenv|env|set|export\s+-p|Get-ChildItem\s+env:|gci\s+env:|dir\s+env:|ls\s+env:)\s*($|[;&|])",
    re.I,
)
DESTRUCTIVE_RE = re.compile(
    r"(^|[\s;&|(])(rm|del|erase|rmdir|rd|unlink|shred|truncate|mv|move|ren|rename|"
    r"Remove-Item|ri|Move-Item|mi|Rename-Item|rni|Clear-Content|clc|Set-Content|sc|Out-File|"
    r"git\s+rm|git\s+mv|robocopy)(?=[\s;&|)]|$)",
    re.I,
)
PROT_TOKEN_RE = re.compile(rf"(^|[\s\"'=(/\\])({PROT})([/\\\s\"')]|$)", re.I)
REDIRECT_TO_PROT_RE = re.compile(rf">{{1,2}}\s*[\"']?(\S*[/\\])?({PROT})[/\\]", re.I)
RM_ROOT_RE = re.compile(
    r"(^|[\s;&|])rm\s+-\w*[rR]\w*\s+[\"']?(/|~|\.|\.\.|\*|\$HOME|\$\{?CLAUDE_PROJECT_DIR\}?)[/\"']?(\s|$)"
)
PS_RECURSE_RE = re.compile(r"Remove-Item\b[^;|&]*-Recurse", re.I)

GIT_RULES = (
    (re.compile(r"\bgit\s+push\b[^;&|]*(\s--force\b|\s--force-with-lease\b|\s-f\b|\s\+\S)"), "git push forcé"),
    (re.compile(r"\bgit\s+reset\b[^;&|]*--hard\b"), "git reset --hard"),
    (re.compile(r"\bgit\s+clean\b[^;&|]*\s-\w*f"), "git clean -f"),
    (re.compile(r"--no-verify\b"), "--no-verify (contourne les hooks)"),
    (re.compile(r"\bgit\s+commit\b[^;&|]*\s-\w*n\w*\b"), "git commit -n (contourne les hooks)"),
    (re.compile(r"\bgit\s+add\b[^;&|]*(\s-f\b|\s--force\b)"), "git add -f (contourne .gitignore)"),
    (re.compile(r"\bgit\s+add\b[^;&|]*\s(\./)?(repos|threads)\b"), "versionner repos/ ou threads/ (données tierces)"),
    (
        re.compile(r"\bgit\s+config\b(?![^;&|]*--get)[^;&|]*core\.hooksPath\b(?![^;&|]*\.githooks)"),
        "modifier core.hooksPath",
    ),
)

COMMIT_RE = re.compile(r"\bgit\s+commit\b")


def strip_quotes(cmd: str) -> str:
    """Retire le contenu des chaînes quotées (messages de commit, etc.) pour l'analyse des drapeaux et des verbes."""
    return re.sub(r"\"(?:[^\"\\]|\\.)*\"|'[^']*'", '""', cmd)


def check(cmd: str) -> str | None:
    """Renvoie la raison du refus, ou None si la commande est acceptée."""
    bare = strip_quotes(cmd)
    if SECRET_RE.search(cmd):
        return "référence au jeton Kaggle ou à une variable de contournement (jeton géré par l'utilisateur)."
    if ENV_DUMP_RE.search(bare):
        return "dump de l'environnement (pourrait exposer des secrets)."
    if RM_ROOT_RE.search(bare) or (
        PS_RECURSE_RE.search(bare) and re.search(r"-Path\s+[\"']?(\.|\\|/|~|C:\\?)[\"']?(\s|$)", bare)
    ):
        return "suppression récursive d'une racine (/, ~, ., projet)."
    if DESTRUCTIVE_RE.search(bare) and PROT_TOKEN_RE.search(cmd):
        return "suppression, déplacement ou écrasement dans un dossier protégé (checkpoints|runs|data|repos|threads)."
    if REDIRECT_TO_PROT_RE.search(bare):
        return "redirection vers un dossier protégé."
    for rule, label in GIT_RULES:
        if rule.search(bare):
            return label
    if COMMIT_RE.search(bare):
        try:
            hooks_path = subprocess.run(
                ["git", "-C", str(PROJECT_DIR), "config", "--get", "core.hooksPath"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return "impossible de vérifier core.hooksPath (fail-closed)."
        if hooks_path != ".githooks":
            return "core.hooksPath doit valoir .githooks. Lancer : git config core.hooksPath .githooks"
    return None


def main() -> None:
    data = read_input()
    cmd = (data.get("tool_input") or {}).get("command") or ""
    if not cmd:
        return
    reason = check(cmd)
    if reason:
        deny(reason)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # fail-closed : une garde qui plante bloque
        deny(f"erreur interne du hook ({type(exc).__name__}: {exc})")
