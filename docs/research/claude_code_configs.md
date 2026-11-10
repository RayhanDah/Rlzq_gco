# Recherche : configurations Claude Code multi-agents (studio RL Rocket League / PPO)

Date de recherche : 2026-11-10. Sources lues via WebFetch (résumé par petit modèle : les extraits "verbatim" ci-dessous sont ceux renvoyés par l'outil, pas relus à la main sur le brut). Tout ce qui n'est pas cité = "non vérifié".

---------------------------------------------------------------------
## (a) Claude-Code-Game-Studios (CCGS)
---------------------------------------------------------------------

### Identité (source : https://api.github.com/repos/Donchitos/Claude-Code-Game-Studios, lu 2026-11-10)
- Propriétaire : **Donchitos** ; repo `Donchitos/Claude-Code-Game-Studios` ; licence MIT ; branche `main`
- Stars : **37 532** ; forks : 4 925 ; `archived: false` ; créé 2026-02-12 ; dernier push **2026-11-09T23:33Z** (donc très actif)
- Note : des agrégateurs tiers affichent "archived" ou 9k/11k/19k stars = snapshots périmés ; l'API GitHub fait foi.
- Page repo (https://github.com/Donchitos/Claude-Code-Game-Studios) : "Turn Claude Code into a full game dev studio" ; README annonce 49 agents, ~72-75 skills (incohérence interne README : 72 dans "About", 75 dans l'arbre), 18 scripts hooks, 14 rules, 62 templates docs.

### Arborescence `.claude/` (README tree)
```
.claude/
  settings.json          # hooks, permissions allow/deny, statusLine, worktree.baseRef=head
  agents/                # 49 fichiers .md (frontmatter YAML)
  skills/<nom>/SKILL.md  # ~75 skills
  hooks/                 # 18 scripts .sh (validate-commit.sh, validate-push.sh, validate-write.sh,
                         #  validate-read.sh, validate-registry.sh, validate-assets.sh,
                         #  validate-skill-change.sh, session-start.sh, detect-gaps.sh, pre-compact.sh,
                         #  session-stop.sh, log-agent.sh, log-agent-stop.sh,
                         #  validate-agent-artifact.sh, notify.sh, yaml-helper.sh ...)
  scripts/               # helpers (project-state, gdd-structure-check, adr-status, review-receipts ...)
  rules/                 # 14 règles path-scopées
  docs/                  # workflow-catalog.yaml (pipeline 7 phases), templates, setup-requirements.md
  genres/ engines/ statusline.sh
CLAUDE.md ; project.yaml ; project.local.yaml (gitignored) ; design/ ; production/ ; docs/
```

### Hiérarchie d'agents (3 tiers) — README
- **Tier 1 Directors** : creative-director, technical-director, producer (modèle **opus**)
- **Tier 2 Leads** : game-designer, lead-programmer, art-director, audio-director, narrative-director, qa-lead, release-manager, localization-lead
- **Tier 3 Specialists** (~30) : gameplay/engine/ai/network/ui/qa programmers, level, economy, sound, writing...
- Répartition modèles (README) : 3 directors = opus ; lead-programmer + 15 specialists = sonnet ; ~30 autres = inherit.

### Définition d'un agent (frontmatter réel, fichiers lus)
`.claude/agents/technical-director.md` (https://raw.githubusercontent.com/Donchitos/Claude-Code-Game-Studios/main/.claude/agents/technical-director.md) :
```yaml
---
name: technical-director
description: "High-level technical decisions — engine architecture, technology choices, ..."
tools: Read, Glob, Grep, Write, Edit, Bash, WebSearch
model: opus
maxTurns: 30
memory: project
---
```
`.claude/agents/gameplay-programmer.md` : `tools: Read, Glob, Grep, Write, Edit, Bash`, `model: inherit`, `maxTurns: 20`.
Corps type (les deux fichiers) : Role -> **Collaboration Protocol** (poser questions, 2-3 options, recommandation, "ask before writing") -> Key Responsibilities -> Decision Framework -> "What this agent must NOT do" -> **Delegation map** (délègue à / escalade vers). Le directeur ne code pas ; le spécialiste n'édite pas la conception.
Mémoire : `memory: project` => `.claude/agent-memory/<agent>/` (vérifié côté doc officielle, voir (c)). Règle dédiée `.claude/rules/agent-memory.md` (paths: `.claude/agent-memory/**`) : mémoire gitignorée, "record methods, not states", divulguer toute écriture mémoire, vérifier les souvenirs avant d'agir.

### Protocole de coordination (README)
1. Délégation verticale directors -> leads -> specialists
2. Consultation horizontale (pas de décision contraignante inter-domaines)
3. Conflits remontent au parent commun (creative-director ou technical-director)
4. Changements inter-départements coordonnés par le producer
5. Un agent reste dans son domaine sauf délégation
+ "Ask before proposing, 2-4 options, draft before finalizing, approval before writing". Seul technical-director peut passer un ADR à Accepted, sur confirmation explicite de l'utilisateur.

### Skills / slash commands & gates
- Exemple lu : `.claude/skills/gate-check/SKILL.md` — frontmatter : `name`, `description`, `argument-hint`, `user-invocable: true`, `allowed-tools: <lecture/écriture + Agent + AskUserQuestion + scripts bash scopés>`, `model: opus`.
- `/gate-check [target-phase]` : verdict **PASS / CONCERNS / NOT ASSESSED / FAIL**, liste blockers + artefacts requis. FAIL/NOT ASSESSED => la phase ne bouge pas ; CONCERNS => avance si l'utilisateur accepte explicitement les risques. Ne corrige jamais automatiquement : nomme le skill à lancer. Sur PASS écrit la phase dans `project.yaml` et `production/stage.txt`.
- 7 phases : Concept -> Systems Design -> Technical Setup -> Pre-Production -> Production -> Polish -> Release (une définition de gate par transition).
- Réglages de rigueur : `modes.rigor` (minimal/standard/full) et `review_mode` (full/lean/solo = nb de revues directors).
- Autres commandes citées : `/start`, `/milestone-review`, `/project-stage-detect`.

### Hooks (source : https://raw.githubusercontent.com/Donchitos/Claude-Code-Game-Studios/main/.claude/settings.json)
| Event | Matcher | Script (`$CLAUDE_PROJECT_DIR/.claude/hooks/...`) | Timeout |
|---|---|---|---|
| SessionStart | tous | session-start.sh ; detect-gaps.sh | 20s ; 10s |
| PreToolUse | `Bash\|PowerShell` | validate-commit.sh ; validate-push.sh | 15s ; 10s |
| PreToolUse | `Write\|Edit` | validate-registry.sh ; validate-write.sh | 15s |
| PreToolUse | `Read` | validate-read.sh | 10s |
| PostToolUse | `Write\|Edit` | validate-assets.sh ; validate-skill-change.sh | 10s ; 5s |
| Notification | `permission_prompt\|idle_prompt` | notify.sh | 10s |
| PreCompact | tous | pre-compact.sh | 10s |
| Stop | tous | session-stop.sh (async) — archive `active.md` en session log | 10s |
| SubagentStart/Stop | tous | log-agent.sh / log-agent-stop.sh ; validate-agent-artifact.sh | 5-10s |
- Chaque commande est enveloppée pour finir en exit 0 (wrapper compatible bash ET PowerShell) => échec de hook non bloquant.
- `validate-commit.sh` (lu) : ne s'active que sur `git commit` ; BLOQUE (exit 2, message sur stderr) si un `.json` stagé sous assets/data est invalide ; WARN seulement pour : sections GDD manquantes, valeurs codées en dur, TODO/FIXME sans owner. Lit le JSON stdin avec `jq` (fallback grep/sed). **Gestion Windows soignée** : sonde `python`/`python3`/`py` en les exécutant (évite le faux python du Store), reconnaît `git.exe`, chemins `C:\...`, `core.quotePath=false`, fallback tmp.
- Permissions : allow `python -m pytest *`, `py -m pytest *`, scripts `.claude/scripts/*` ; deny : `rm -rf` (toutes variantes) + `Remove-Item -Recurse`, `git push --force/-f/+refspec`, `git reset --hard`, `git clean -f`, `sudo *`, `chmod 777*`, lecture/édition `**/.env*` (et `Get-Content *.env*`).
- Le hook SessionStart affiche un avertissement si Git Bash est absent (hooks désactivés).

### Règles path-scopées (14, https://api.github.com/repos/Donchitos/Claude-Code-Game-Studios/contents/.claude/rules)
agent-memory, ai-code, data-files, design-docs, engine-code, gameplay-code, narrative, network-code, prototype-code, scene-files, shader-code, skill-authoring, test-standards, ui-code (.md).
Exemple lu `test-standards.md` : frontmatter `paths:` liste de globs (`tests/**` ...) ; contenu : AAA, isolation (pas de FS/réseau/DB en unit), seuils de perf qui font échouer, fixtures sans état partagé mockées, **chaque bug fix = test de régression vu ÉCHOUER avant le fix**.

### État / mémoire
`CLAUDE.md` (config maître), `project.yaml` (+ `project.local.yaml` gitignoré), `production/session-state/active.md` (checkpoint append-only), `production/stage.txt`, `production/qa/evidence/`, `.claude/agent-memory/`, journal d'agents via log-agent*.sh.

### Spécifique jeu vs réutilisable pour ML
| Réutilisable | Spécifique jeu (à jeter) |
|---|---|
| Tiering directors/leads/specialists + règle "pas de décision hors domaine" ; escalade au parent | creative/art/audio/narrative/localization/level/economy directors et specialists |
| Frontmatter `maxTurns`, `memory: project`, `model` par tier | GDD, ADR "engine", moteurs Godot/Unity/Unreal, shaders, scenes, UI |
| Skill `/gate-check` à verdicts PASS/CONCERNS/FAIL, ne corrige pas lui-même | 7 phases de production de jeu |
| Hooks : validate-commit (bloque sur données invalides), deny `rm -rf`/force-push/.env, log SubagentStart/Stop, PreCompact, session-stop qui archive l'état | validate-assets, validate-registry (assets de jeu) |
| Rule test-standards (régression vue échouer), agent-memory | rules gameplay/network/ui/narrative |
| project.yaml + review_mode (rigueur réglable) | `/start` brief de jeu |
Limite du repo : 49 agents = coût de contexte élevé (descriptions listées à chaque session) ; pour un studio RL petit c'est excessif.

---------------------------------------------------------------------
## (b) Configs "hackathon" / pratiques reconnues
---------------------------------------------------------------------

### everything-claude-code (ECC) — affaan-m
- Repo : l'API GitHub renvoie `full_name: affaan-m/ECC` (renommé ; l'ancienne URL `affaan-m/everything-claude-code` est utilisée dans le README/raw), **300 221 stars**, 44 336 forks, push 2026-11-09, non archivé (https://api.github.com/repos/affaan-m/everything-claude-code ; https://github.com/affaan-m/everything-claude-code). MIT.
- Lien hackathon : la page GitHub ne le mentionne pas. Sources secondaires seulement (non primaires) : Affaan Mustafa a gagné un hackathon Anthropic x Forum Ventures (sept. 2025, New York selon Luma, 8 h, produit zenith.chat, ~15 000 $ de crédits) puis open-sourcé sa config. Lieu discordant selon sources ; à traiter comme **non vérifié en source primaire** (recherche web : eu.36kr.com/en/p/3823966768943239 ; forumvc.com/forum-ventures-x-anthropic-ai-hackathon ; luma.com/9reainnk).
- Structure (README) : `agents/` (71 ; planner.md, tdd-guide.md, code-reviewer.md), `skills/` (314 ; tdd-workflow/, security-review/, verification-loop/, eval-harness/, continuous-learning-v2/), `commands/` (98, shims de compat ; skills = surface principale), `hooks/hooks.json` + `scripts/hooks/`, `rules/common/` + packs par langage (`rules/python/`...). Conseil du README : commencer par `common` + UN pack de stack.
- Cycle de travail annoncé : plan -> test -> implement -> review -> verify -> remember -> improve.
- Agent lu `agents/code-reviewer.md` : `tools: Read, Grep, Glob, Bash`, `model: sonnet` ; filtrage par confiance (> 80 %), liste de faux positifs, sévérités CRITICAL/HIGH/MEDIUM/LOW, verdict Approve/Warning/Block ; "zéro finding = valide".
- Hooks (`hooks/hooks.json`, lu) : 9 types d'events ; chaque commande = `node -e "<bootstrap>"` qui résout le plugin root puis `scripts/hooks/run-with-flags.js <hookId> <script> <profils>` avec **profils minimal/standard/strict** (permet de désactiver des hooks coûteux). Exemples : PreToolUse `config-protection` (Write|Edit|MultiEdit), `doc-file-warning`, `suggest-compact`, `gateguard-fact-force`, `observe-runner` (async, matcher `.*`) ; PostToolUse dispatcher sync(30s)+async(45s) ; Stop : `format-typecheck` (timeout 300), `check-console-log`, `evaluate-session`, `cost-tracker`, `desktop-notify`, `session-end` (async) ; PreCompact, SessionStart.
- Windows : README classe Windows natif "Supported with limitations" (observer daemon et memory-vault de continuous-learning-v2 ont des défauts ouverts #2489, #2626 ; features shell exigent Git Bash/WSL). Leçon : ce repo est lourd (300+ skills) et partiellement fragile sur Windows ; piocher des idées, ne pas installer en bloc.

### Autres références (primaires, vérifiées)
- obra/superpowers : 315 135 stars, push 2026-11-03, "An agentic skills framework & software development methodology" (https://api.github.com/repos/obra/superpowers). Contenu interne (brainstorm -> plan -> TDD) : **non vérifié** (fichiers non lus).
- Doc officielle best practices (https://code.claude.com/docs/en/best-practices) — patterns concrets :
  * "Give Claude a check to run" : test/exit code/linter/diff vs fixture ; la ligne la plus utile du CLAUDE.md = la commande de vérification ; exiger des **preuves** (commande + sortie), pas "tests OK".
  * Escalade du niveau d'enforcement : prompt -> `/goal <condition>` (modèle séparé vérifie après chaque tour) -> **hook Stop** (exit 2 renvoie Claude au travail) -> relecture par subagent qui n'a pas vu le code (`/code-review`).
  * Stop hook : plafond de 8 blocages consécutifs sans tool call, réglable via `CLAUDE_CODE_STOP_HOOK_BLOCK_CAP` ; script doit tester `stop_hook_active` (hooks-guide : https://code.claude.com/docs/en/hooks-guide).
  * Revue vs spec : "use a subagent to review the diff against SPEC.md ... report gaps affecting correctness and skip style" (sinon le reviewer invente des problèmes). => **spec-first** : SPEC.md écrit avant, reviewer en `tools: Read, Grep, Glob`.
  * Règle d'or : "If something has to happen every time, make it a hook, a deny rule, or a managed setting. CLAUDE.md is for guidelines".
  * Coût subagents : tokens (chaque subagent a ses requêtes), peut être plus lent qu'inline ; `model: haiku` pour tâches simples ; demander des résumés courts ; `/usage` pour voir le coût par subagent ; `/clear` entre tâches sans rapport ; `/compact <focus>`.
  * Parallèle : `claude -w feature-auth` (worktree sous `.claude/worktrees/<name>/`, branche `worktree-<name>`) ; `.worktreeinclude` (syntaxe .gitignore) copie `.env` etc. ; `/batch` découpe en 5-30 unités, 1 worktree chacune (https://code.claude.com/docs/en/worktrees).
- Worktrees (https://code.claude.com/docs/en/worktrees) : fresh checkout => dépendances/venv/datasets absents ; mettre la commande d'install dans CLAUDE.md ; `worktree.symlinkDirectories` pour gros dossiers ; `${CLAUDE_PROJECT_DIR}` reste sur la racine principale dans les hooks, le champ `cwd` du JSON stdin suit le worktree ; sous Windows suppression d'un worktree ne supprime que les junctions, pas leur cible. Baseline : branche depuis la branche par défaut du remote (`worktree.baseRef: "fresh"`) sauf `"head"`.
- Pour RL spécifiquement (checkpoints, runs longs) : aucune config publique d'RL trouvée/vérifiée dans cette recherche => **non vérifié**.

---------------------------------------------------------------------
## (c) Formats officiels vérifiés (code.claude.com, lu 2026-11-10)
---------------------------------------------------------------------
Note : docs.claude.com/docs.anthropic.com redirigent vers code.claude.com/docs. Les pages renvoient `llms.txt` index : https://code.claude.com/docs/llms.txt.

### Subagents — https://code.claude.com/docs/en/sub-agents
Fichier Markdown + frontmatter YAML, corps = system prompt. **Requis : `name`, `description`.** Champs camelCase exacts ; champs inconnus ignorés SANS erreur (donc typos silencieuses).
Champs supportés : `name` (≤256 car., sans `:`), `description`, `tools` (liste CSV ou YAML ; hérite de tout si omis), `disallowedTools` (appliqué avant `tools`), `model` (`sonnet|opus|haiku|fable|<id complet>|inherit`), `permissionMode` (`default|acceptEdits|auto|dontAsk|bypassPermissions|plan`), `maxTurns`, `autoCompactWindow` (100k-1M, v2.1.296+), `skills` (préchargés, ≤32), `mcpServers`, `hooks`, `memory` (`user|project|local`), `background`, `omitClaudeMd` (v2.1.271+), `effort` (`low|medium|high|xhigh|max`), `isolation: worktree`, `color`, `initialPrompt`, `experimental.cacheTtl`.
Priorité : managed > `--agents` > `.claude/agents/` (projet, scan récursif) > `~/.claude/agents/` > plugin. Fichier ignoré si `---` pas en ligne 1, `name` sans description, YAML invalide ; debug : `claude --debug`, `claude plugin validate .claude/agents`.
Mémoire : `project` => `.claude/agent-memory/<name>/` ; injecte les 200 premières lignes / 25 KB de `MEMORY.md`, active Read/Write/Edit auto ; sans effet si auto memory désactivée.
Hooks en frontmatter : seulement pendant que l'agent est actif ; hooks d'agents de projet ne tournent qu'après acceptation du workspace trust. `SubagentStart`/`SubagentStop` se configurent dans settings.json (matcher = nom d'agent).
Windows : "Hook scripts on Windows should be PowerShell. Add `shell: powershell`" (indiqué dans la page sub-agents).

Exemple minimal correct (agent RL) :
```markdown
---
name: ppo-reviewer
description: Revoit un diff d'entraînement PPO contre SPEC.md (reward, GAE, normalisation, seeds). À utiliser avant tout run long.
tools: Read, Grep, Glob
model: sonnet
maxTurns: 15
---
Tu relis du code que tu n'as pas écrit. Compare le diff à SPEC.md ; ne rapporte que
les écarts qui affectent la correction (fuite de reward, mauvais masque done, lr/clip incohérents).
Ignore le style. Réponds en <= 20 lignes avec fichier:ligne.
```

### Skills — https://code.claude.com/docs/en/skills
Emplacement : `.claude/skills/<nom>/SKILL.md` (projet), `~/.claude/skills/` (perso), nested `<sousdossier>/.claude/skills/`, plugin. `.claude/commands/x.md` et `skills/x/SKILL.md` créent tous deux `/x` (commands fusionnés dans skills ; commands n'ont ni `name` ni `paths`).
Tous champs optionnels, `description` fortement recommandée : `name`, `description`, `when_to_use`, `argument-hint`, `arguments`, `disable-model-invocation` (true = utilisateur seul ; à utiliser pour effets de bord), `user-invocable` (false = connaissance de fond), `allowed-tools` (pré-autorise, ne restreint pas), `disallowed-tools`, `model`, `effort`, `context: fork` (+ `agent:` Explore/Plan/general-purpose/custom, `background`), `hooks`, `paths` (globs de chargement auto), `shell` (`bash` défaut | `powershell` pour l'injection `` !`cmd` ``), `metadata/license/compatibility`.
Substitutions : `$ARGUMENTS`, `$0`/`$1`, `$nom`, `${CLAUDE_SKILL_DIR}`, `${CLAUDE_PROJECT_DIR}`, `${CLAUDE_SESSION_ID}`. Injection dynamique `` !`commande` `` (échec non-zéro = abort du skill ; ajouter `|| true`). SKILL.md < 500 lignes, détails dans fichiers annexes. Description+when_to_use tronqués à 1 536 car.
Exemple minimal :
```markdown
---
name: eval-checkpoint
description: Évalue un checkpoint PPO contre le baseline et sort un verdict PASS/FAIL. Use when asked to evaluate or compare a model checkpoint.
argument-hint: "[chemin_checkpoint]"
disable-model-invocation: true
allowed-tools: Bash(python -m rlbot.eval *)
shell: powershell
---
Évalue $ARGUMENTS :
1. Lance `python -m rlbot.eval --ckpt $ARGUMENTS --episodes 200 --seed 0`
2. Compare reward moyen / win-rate au baseline dans runs/baseline.json
3. Verdict PASS si win-rate >= baseline - 1 écart-type ; sinon FAIL avec la commande et la sortie comme preuve.
```
(`shell: powershell` ne concerne que l'injection `!`cmd``, pas `allowed-tools`; ici facultatif.)

### Hooks — https://code.claude.com/docs/en/hooks et https://code.claude.com/docs/en/hooks-guide
- 33 events listés : SessionStart, Setup, UserPromptSubmit, UserPromptExpansion, PreToolUse, PermissionRequest, PermissionDenied, PostToolUse, PostToolUseFailure, PostToolBatch, Notification, MessageDisplay, SubagentStart, SubagentStop, TaskCreated, TaskCompleted, Stop, StopFailure, TeammateIdle, InstructionsLoaded, ConfigChange, CwdChanged, DirectoryAdded, FileChanged, WorktreeCreate, WorktreeRemove, PreCompact, PostCompact, PreModelSwitch, PostModelSwitch, Elicitation, ElicitationResult, SessionEnd.
- Où : `~/.claude/settings.json`, `.claude/settings.json`, `.claude/settings.local.json`, managed, plugin `hooks/hooks.json`, frontmatter skill/agent.
- Structure : event -> groupe `{matcher, hooks:[handlers]}` -> handler `{type: command|http|mcp_tool|prompt|agent, command, args?, if?, timeout?, shell?, once?, async?, statusMessage?, onFailure?}`. Défauts timeout : 600 s command/http/mcp_tool, 30 s prompt, 60 s agent.
- Matcher : `"*"`, `""` ou absent = tout ; lettres/chiffres/`_ - , |` et espaces = nom exact ou liste (`Edit|Write`) ; tout autre caractère => regex JS non ancrée. Pour events d'outils, le matcher = nom d'outil (`mcp__srv__.*`). `if` (syntaxe permission, ex. `Bash(git *)`) seulement sur events d'outils (sur un autre event le hook ne s'exécute pas).
- Exit codes : **0** succès (stdout parsé en JSON si objet unique ; texte ajouté au contexte seulement pour UserPromptSubmit, SessionStart, etc.) ; **2** erreur bloquante sur events bloquants (stderr renvoyé à Claude) ; **autre (dont 1)** non bloquant. `"onFailure": "block"` pour bloquer si le hook plante/timeout.
- JSON de sortie : universel `continue:false`, `stopReason`, `systemMessage` ; PreToolUse : `hookSpecificOutput.{hookEventName:"PreToolUse", permissionDecision: allow|deny|ask|defer, permissionDecisionReason, updatedInput}` ; PostToolUse/Stop/SubagentStop/UserPromptSubmit : `decision:"block"` + `reason` (PostToolUse : `continueOnBlock:true` pour réinjecter) ; `additionalContext` injecte du contexte.
- Entrée stdin JSON : `session_id, transcript_path, cwd, permission_mode, hook_event_name`, + `tool_name, tool_input, tool_use_id` pour les outils.
- Variables : `CLAUDE_PROJECT_DIR` (racine du démarrage, ne suit pas un worktree), `CLAUDE_ENV_FILE` (SessionStart/Setup/CwdChanged/FileChanged uniquement ; non disponible pour hooks PowerShell).
- Stop : bloqué 8 fois d'affilée sans tool call => Claude Code force l'arrêt ; lire `stop_hook_active` dans l'input pour éviter les boucles.
- **Windows** (hooks reference) : les hooks command tournent dans **Git Bash s'il est installé, sinon PowerShell** ; choix explicite via `"shell": "bash"|"powershell"`. Sans Git Bash, l'outil Bash n'est pas enregistré : un matcher `Bash` seul ne se déclenche jamais => utiliser **`Bash|PowerShell`**. Forme exec (`args` présent) : `command` doit être un vrai `.exe` ; `.cmd/.bat` nécessitent forme shell ou `node script`. Un hook qui échoue sans Git Bash peut bloquer l'outil ("blocking because Git Bash wasn't found"). Exemple officiel sans quoting :
```json
{ "type": "command", "command": "powershell.exe",
  "args": ["-NoProfile","-ExecutionPolicy","Bypass","-File","${CLAUDE_PROJECT_DIR}/.claude/hooks/x.ps1"] }
```
Exemple hook minimal officiel adapté (settings.json, Stop = gate de tests, exit 2 pour bloquer) :
```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash|PowerShell",
        "hooks": [ { "type": "command", "if": "Bash(rm *)",
                     "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/block-rm.sh", "timeout": 30 } ] }
    ],
    "Stop": [
      { "hooks": [ { "type": "command", "timeout": 120,
                     "command": "python -m pytest -x -q tests/unit 1>&2 || exit 2" } ] }
    ]
  }
}
```
(Le Stop gate s'inspire de l'exemple officiel `pnpm tsc --noEmit 1>&2 || exit 2`; la commande pytest est mon adaptation, non testée.)

### Rules path-scopées + CLAUDE.md + mémoire — https://code.claude.com/docs/en/memory
- `.claude/rules/*.md` (sous-dossiers OK ; symlinks OK mais une cible hors projet exige approbation et ne charge pas les règles à `paths`). Règle SANS `paths:` = chargée au lancement (même priorité que `.claude/CLAUDE.md`).
- **Mécanisme confirmé** : frontmatter `paths:` (liste YAML ou chaîne CSV) de globs relatifs au dossier contenant `.claude` ; `paths` est le SEUL champ lu, autres ignorés. La règle se charge quand Claude utilise Read/Write/Edit (ou Bash `cat/head` sur un fichier) correspondant. Brace expansion OK (`src/**/*.{ts,tsx}`). Pour worktrees frères : démarrer Claude dans le worktree ou `/cd`.
```markdown
---
paths:
  - "src/rlbot/**/*.py"
  - "configs/**/*.yaml"
---
# Règles training
- Toujours fixer et logger les seeds (python, numpy, torch).
- Ne jamais modifier la reward sans mettre à jour docs/reward.md.
```
- CLAUDE.md : emplacements projet `./CLAUDE.md` ou `./.claude/CLAUDE.md`, perso `~/.claude/CLAUDE.md`, local `./CLAUDE.local.md` (à gitignorer ; n'existe que dans le worktree où il est créé ; importer un fichier du home pour partager entre worktrees). Chargé depuis le cwd et tous les dossiers parents ; sous-dossiers à la demande. **Cible < 200 lignes**. Re-lu depuis disque après `/compact`.
- Imports : `@chemin/fichier` (relatif au fichier contenant l'import, absolu OK), récursif **max 4 sauts** ; ignoré dans code spans/blocs ; **les imports ne réduisent pas le coût de contexte** (chargés avec le CLAUDE.md). Import externe hors projet demande approbation. `AGENTS.md` lu seulement s'il n'y a pas de CLAUDE.md (sauf réglage) ; sous Windows ne pas symlinker CLAUDE.md -> AGENTS.md, utiliser `@AGENTS.md`.
- Auto memory : notes écrites par Claude ; 200 premières lignes/25 KB de `MEMORY.md` chargées à chaque session, détails dans fichiers thématiques ; `autoMemoryDirectory` pour relocaliser ; `/memory` pour gérer ; partagée entre worktrees d'un repo. `claudeMdExcludes` pour ignorer des CLAUDE.md.
- Un CLAUDE.md est du contexte, pas de l'enforcement : pour bloquer, utiliser PreToolUse (dit explicitement la doc).

### Divers vérifiés
- Subagents lancés en arrière-plan par défaut ; coûts et conseils : best-practices ci-dessus.
- Skill `context: fork` + `agent:` = alternative à un subagent dédié (un skill qui s'exécute isolé).
- Cache de prompt ~1 h abonnement / 5 min clé API ; subagents 5 min par défaut (best-practices).
- Non lu / non vérifié : agent teams (https://code.claude.com/docs/en/agent-teams — cité par la page worktrees, contenu non lu), `permissions` détaillées de settings.json, schéma exact `statusLine`.

---------------------------------------------------------------------
## (d) Recommandations pour un studio RL (Rocket League, PPO, Windows 11)
---------------------------------------------------------------------

### À adopter
1. **Peu d'agents, bien typés** (5-7 max au début), en 2 niveaux plutôt que 3 :
   - `rl-lead` (opus, orchestration/décisions d'architecture et de reward) — ou laisser la session principale jouer ce rôle ;
   - spécialistes : `env-engineer` (RLGym/bridge RocketSim/obs/action/reward), `trainer` (PPO, hyperparams, GPU, checkpoints), `eval-analyst` (métriques, courbes, comparaisons ; lecture seule + Bash), `reviewer` (Read/Grep/Glob seulement, sonnet), `explorer` haiku pour recherches de fichiers. Reprendre le patron CCGS : section "ne doit PAS faire" + delegation map + `maxTurns` + `memory: project` seulement pour lead/analyst.
2. **Spec-first** : `SPEC.md`/`docs/specs/<feature>.md` (obs space, action space, reward, critère d'arrêt) écrit avant le code ; reviewer compare le diff à la spec (modèle officiel best-practices). Une "definition of done" par type de tâche dans CLAUDE.md : tests unitaires verts + smoke run de N steps sans NaN + preuve (commande + sortie) collée.
3. **Gate skill à la CCGS** : `/gate-check [train|eval|promote]` avec verdicts PASS/CONCERNS/FAIL ; gates RL candidats : (a) smoke test env (reset/step déterministe avec seed, shapes obs/action), (b) reward sanity (pas de reward hacking connu), (c) no-NaN/entropy dans bornes sur 10k steps, (d) éval vs baseline (win-rate/ELO) avant de promouvoir un checkpoint. `disable-model-invocation: true` pour tout skill qui lance un entraînement long ou écrit dans `checkpoints/` (effet de bord, coûteux).
4. **Hooks minimaux mais déterministes** (source CCGS + doc) :
   - PreToolUse `Bash|PowerShell` : deny `rm -rf`/`Remove-Item -Recurse`, force-push, `git reset --hard`, écriture/suppression dans `checkpoints/` et `data/raw/` (copier la liste deny de CCGS settings.json, adaptée) ;
   - PreToolUse `Write|Edit` : protéger fichiers figés (baseline eval, `configs/reward_frozen.yaml`) ;
   - Stop : tests rapides (`pytest -x -q tests/unit`, < 30-60 s) avec exit 2 ; protéger avec `stop_hook_active` ;
   - PreCompact + SessionStart/Stop : sauver/recharger un petit `state/active.md` (run en cours, dernier checkpoint, hypothèse testée) — idem `production/session-state/active.md` de CCGS ;
   - SubagentStart/Stop : log léger (CCGS) pour audit.
5. **Règles path-scopées RL** (`.claude/rules/*.md` avec `paths:`) : `training.md` (src/**/train*.py, configs/** : seeds, logging, pas de reward changée sans doc), `env-reward.md` (src/**/env/**, reward/**), `tests.md` (tests/** : régression vue échouer, comme CCGS test-standards), `experiments.md` (runs/**, notebooks : nommage, un run = un config hashé). Garder chaque règle courte ; CLAUDE.md < 200 lignes.
6. **Mémoire projet** : CLAUDE.md (commandes train/eval/test, conventions Windows, layout) + `@docs/rl-glossary.md` si utile (rappel : l'import ne réduit pas le coût). Auto memory pour apprentissages ; un journal d'expériences versionné (`docs/experiments.md` : hypothèse, config, résultat, décision) plus fiable que la mémoire auto. Règle "record methods, not states" de CCGS agent-memory.
7. **Parallélisme** : worktrees pour tâches indépendantes (ex. refonte reward vs optimisation dataloader) ; `isolation: worktree` sur un agent qui modifie beaucoup de fichiers. Pour les RUNS d'entraînement : ne pas les lancer dans des subagents/worktrees concurrents sur le même GPU sans contrôle ; worktree = pas de venv/datasets/checkpoints => `.worktreeinclude` pour petits fichiers (`.env`), `worktree.symlinkDirectories` ou chemin absolu partagé pour data/checkpoints, install venv documentée dans CLAUDE.md.
8. **Review indépendante** avant promotion d'un checkpoint ou changement de reward : subagent read-only (modèle best-practices) + consigne "skip style".

### À jeter (spécifique jeu)
creative/art/audio/narrative/localization directors, level/economy/sound/writing specialists, GDD/ADR moteur, genres/engines, shaders/scenes/ui/narrative rules, validate-assets/registry, pipeline 7 phases de production, templates de design de jeu. Ne pas installer ECC en bloc (314 skills, 71 agents, fragile sur Windows natif selon son README).

### Pièges
- **Coût de contexte** : chaque agent/skill ajoute sa description à chaque session (doc : "descriptions consume context") ; CCGS 49 agents + 75 skills et ECC 71+314 sont des extrêmes. Viser < 10 agents, descriptions courtes, détails dans fichiers annexes de skill (SKILL.md < 500 lignes). Subagents = plus de tokens, parfois plus lents que travailler inline.
- **Imports `@` n'économisent rien** ; CLAUDE.md > 200 lignes dégrade l'adhérence (doc). Utiliser rules `paths:` pour le conditionnel.
- **Hooks lents** : un hook sur `PreToolUse` `.*` ou `PostToolUse` sans matcher s'exécute à chaque appel (Python sur Windows = 100-300 ms de démarrage). Toujours matcher étroit + `if`, timeouts courts (CCGS : 5-20 s), `async: true` pour logs, jamais pytest complet dans PostToolUse ; Stop gate rapide ou profils (idée ECC minimal/standard/strict via variable d'env) ; plafond 8 blocages Stop.
- **Windows** : (1) shell de hook = Git Bash si présent sinon PowerShell -> trancher et expliciter avec `"shell"` ; (2) matcher `Bash|PowerShell`, pas `Bash` seul ; (3) quoting : préférer la forme `command` + `args` (pas de shell-quoting, doc officielle) ou un script `.ps1`/`.py` appelé par chemin plutôt que des one-liners JSON échappés ; `.cmd/.bat` interdits en exec form ; (4) `python` peut être le faux alias du Windows Store -> sonder comme CCGS validate-commit.sh, ou appeler `py -3` / chemin du venv ; (5) chemins `\` vs `/` dans le JSON stdin (CCGS normalise les backslashes) ; (6) `CLAUDE_ENV_FILE` indisponible aux hooks PowerShell ; (7) symlinks Windows exigent Developer Mode/admin -> utiliser `@import` pas symlink ; Git checkout d'un symlink = fichier texte si `core.symlinks` off ; (8) CRLF : écrire les scripts `.sh` en LF (`.gitattributes` `*.sh text eol=lf`) sinon Git Bash échoue — point **non vérifié dans la doc**, connaissance générale, à tester ; (9) PowerShell worktree : seule la vérification du répertoire de travail s'applique (pas de contrôle git redirect).
- **Hooks fail-open** : CCGS enveloppe en exit 0 (rien ne bloque sur crash). Pour une garde de sécurité (checkpoints), préférer `"onFailure": "block"` ou exit 2 explicite, et tester que le hook se déclenche (`claude --debug`, `/hooks`... la commande `/hooks` n'a pas été vérifiée dans cette recherche).
- **Champs frontmatter inconnus ignorés silencieusement** : valider avec `claude --debug` / `claude plugin validate .claude/agents`. `hooks`, `mcpServers`, `permissionMode` sont ignorés pour agents de plugin ; les hooks d'agents projet exigent le trust dialog.
- **permissionMode hérité** : si la session principale est en `bypassPermissions`/`acceptEdits`/auto, le champ du subagent est ignoré -> ne pas s'appuyer dessus pour la sécurité ; utiliser `tools`/`disallowedTools` et deny rules.
- **Gates auto-déclarés** : le LLM peut dire "PASS" ; faire écrire les verdicts par un script qui exécute réellement le test (preuve = commande + sortie).
- **Entraînement long dans la session** : pas de subagent pour des runs de plusieurs heures ; utiliser processus détaché + log + skill `/eval-checkpoint` ; le cache de prompt expire (5 min API / ~1 h abonnement) donc sessions longues idle coûtent plus cher.
- **Données de sources secondaires** : tout ce qui concerne le hackathon (date, lieu, prix) provient de blogs/agrégateurs, pas de sources officielles Anthropic ; ne rien écrire de cela dans la config.

### Squelette proposé (à créer plus tard, rien créé ici)
```
CLAUDE.md                      # <200 lignes : commandes, DoD, layout, pièges Windows
.claude/settings.json          # hooks (deny, Stop gate, PreCompact), permissions
.claude/agents/                # rl-lead, env-engineer, trainer, eval-analyst, reviewer, explorer
.claude/skills/                # gate-check, eval-checkpoint, run-experiment, spec-new
.claude/rules/                 # training.md, env-reward.md, tests.md, experiments.md (paths:)
.claude/hooks/                 # guard-bash.py|.ps1, stop-tests.py, session-state.py
docs/specs/ ; docs/experiments.md ; state/active.md
```
