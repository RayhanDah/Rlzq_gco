# ADR-000 — Structure du studio Claude Code et garde-fous

- Statut : accepté
- Date : 2026-11-10
- Décideurs : tech-director (session principale), d'après la mission validée par l'utilisateur au G1

## Contexte
La mission impose plusieurs éléments :
- une hiérarchie de 17 agents ;
- 10 skills ;
- des hooks (lint, format et types avant commit, blocage des commits sans tests, journal des runs, chargement de l'état,
  protection des checkpoints et des données) ;
- des rules path-scopées, une mémoire `production/state.md` et un protocole de coordination.

S'y ajoutent des contraintes du poste et du projet :
- le poste tourne sous Windows 11, avec Git Bash et Python 3.12 système ;
- les formats Claude Code ont été vérifiés dans la doc officielle (`docs/research/claude_code_configs.md` §c) ;
- l'entraînement se fera sur Kaggle (décision G1), ce qui ajoute un secret à protéger : le jeton API.

## Options
| Option | Pour | Contre |
|---|---|---|
| A. Reprendre CCGS tel quel (49 agents, hooks shell en fail-open) | complet | coût de contexte, rôles de jeu inutiles, une garde qui plante ne bloque rien |
| B. Studio minimal (5–7 agents) | contexte léger | ne respecte pas la hiérarchie demandée |
| **C. 17 agents à descriptions courtes, hooks Python stdlib en fail-closed, gates vérifiés par script** | respecte la mission, portable Windows, testable | 17 descriptions chargées à chaque session (coût modéré) |

## Mesures
Commit initial, machine du projet (Zenbook 14, Python 3.12.6, Git 2.49.0), 2026-11-10 :
- `python -m unittest discover -s tests/hooks -v` donne 8 tests OK. Ils couvrent :
  - 27 commandes refusées et 14 acceptées par `guard_shell` ;
  - 8 chemins refusés et 4 acceptés par `guard_paths` ;
  - `log_runs`, `stop_tests` et les motifs de `precommit`.
- Déclenchement prouvé en session réelle :
  - un `Write` vers `checkpoints/hook_probe.txt` a été refusé par `guard_paths.py` (exit 2) ;
  - `git config --get core.hooksPath` a d'abord été refusé à tort par `guard_shell.py`. C'était un faux positif, corrigé
    puis couvert par un test.
- La latence des hooks n'a pas été mesurée isolément. La suite de tests lance 40 et quelques sous-processus Python en ≈ 4,5 s,
  soit environ 0,1 s par appel (estimation dérivée).

## Décision
Option C.
- **Agents** dans `.claude/agents/`, sur 3 niveaux, avec parent, interdits, outils restreints, et `code-reviewer` en lecture
  seule.
- **Skills** dans `.claude/skills/`. Les skills à effet de bord (`train`, `benchmark`, `export-bot`) portent
  `disable-model-invocation: true`.
- **Hooks** Python stdlib dans `.claude/hooks/`, déclarés dans `.claude/settings.json` avec le matcher `Bash|PowerShell`.
  Les gardes sont en `onFailure: block` et une exception interne donne un exit 2.
- **Commit** : `scripts/precommit.py` tourne en hook git (`core.hooksPath=.githooks`). `guard_shell` refuse un commit si
  `hooksPath` est modifié, et refuse `--no-verify`.
- **Gates** : `scripts/gate_check.py`. Il ne remplace jamais le GO humain.
- **Protocole** : `docs/STUDIO.md`.

## Conséquences
- Une garde peut produire des faux positifs (cas vu ci-dessus). On ajoute alors un test avant de corriger la regex.
- Les regex shell ne sont pas une sandbox : un contournement délibéré reste possible. Le but est d'éviter les accidents, pas
  de résister à un adversaire.
- On révisera cette structure si les descriptions d'agents coûtent trop de contexte (mesurable avec `/usage`), ou si une
  garde bloque trop souvent du travail légitime.
