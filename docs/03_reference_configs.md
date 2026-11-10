# 03 — Configurations de référence et leçons d'ingénierie (Phase 0d)

Date : 2026-11-10. Notes détaillées : `docs/research/claude_code_configs.md` (formats officiels vérifiés sur code.claude.com),
`docs/02_ecosystem.md` §8 (Necto/Nexto), annexe privée non versionnée (pratiques Pulsar, Terminal, Disco, Titan).

## 1. Studios Claude Code étudiés

| Référence | Ce qu'on reprend | Ce qu'on jette |
|---|---|---|
| **Donchitos/Claude-Code-Game-Studios** (MIT, ≈ 37,5k★, push 2026-11-09) : 49 agents en 3 niveaux, ~75 skills, 18 hooks, 14 rules path-scopées | hiérarchie directors/leads/specialists et règle « pas de décision hors de son domaine », escalade vers le parent ; skill `/gate-check` à verdicts PASS/CONCERNS/FAIL qui **ne corrige pas lui-même** ; hooks validate-commit, deny `rm -rf`/force-push, PreCompact/SessionStart qui sauve et recharge l'état ; rule test-standards (une régression doit être vue échouer avant le fix) ; fichier d'état de session | tous les rôles créatifs (art, audio, narration, level design), pipeline de production en 7 phases, hooks d'assets ; hooks enveloppés en exit 0 (fail-open : une garde qui plante ne bloque rien) |
| **affaan-m/everything-claude-code** (≈ 300k★ ; lien avec le hackathon Anthropic × Forum Ventures **non vérifié en source primaire**) | cycle plan → test → implement → review → verify ; reviewer à filtrage par confiance et verdict Approve/Warning/Block ; **profils de hooks** minimal/standard/strict ; Stop hook de vérification | volume (71 agents, 314 skills), fragile sur Windows natif selon son propre README |
| Doc officielle best-practices | « give Claude a check to run » ; preuve = commande + sortie ; tout ce qui doit arriver à chaque fois devient un **hook** ; revue du diff **contre la spec** par un subagent qui n'a pas écrit le code ; worktrees pour le parallélisme | — |

**Formats vérifiés** (détails et exemples minimaux dans les notes) :
- agents `.claude/agents/*.md` : seuls `name` et `description` sont requis ; `tools`, `model`, `maxTurns`, `memory`,
  `isolation: worktree`… sont optionnels ; un champ inconnu est **ignoré en silence** ;
- skills `.claude/skills/<n>/SKILL.md` ; `disable-model-invocation: true` pour les skills à effet de bord (train, export) ;
- hooks dans `settings.json` : exit 2 = blocage ;
- rules `.claude/rules/*.md` avec `paths:`.

**Spécificités Windows** :
- les hooks tournent sous Git Bash s'il est présent, sinon sous PowerShell ;
- le matcher doit être `Bash|PowerShell` ;
- préférer des scripts `.py` appelés par chemin plutôt que des one-liners échappés ;
- `python` peut être l'alias du Microsoft Store.

## 2. Décisions qui en découlent pour la Phase 1 (proposées, pas encore appliquées)
1. **La hiérarchie demandée (17 agents) est conservée**, avec des descriptions **très courtes**. Chaque description est chargée
   à chaque session (coût de contexte). Les détails vont dans les corps d'agents et les skills.
2. **Gates exécutés par scripts** : un verdict PASS doit venir d'une commande réellement lancée (pytest, smoke test, SPRT), jamais
   d'un « PASS » déclaré par un LLM.
3. **Hooks étroits et rapides** :
   - deny sur les écritures/suppressions dans `checkpoints/` et `runs/` ;
   - Stop gate = tests unitaires rapides (< 60 s), qui testent `stop_hook_active` ;
   - SessionStart charge `production/state.md` ;
   - aucun hook sur `.*` (Python sur Windows ajoute 100–300 ms à chaque appel).
4. **Gardes de sécurité fail-closed** : exit 2 explicite, contrairement à CCGS.
5. **Pas de run d'entraînement dans un subagent** : processus détaché, log, skill `/train` qui lance et `/status` qui lit.

## 3. Leçons d'ingénierie des bots communautaires
| Leçon | Source | Application |
|---|---|---|
| Mesurer avant d'ajouter de la machinerie : chaque intervention exige une mesure qui condamne le problème, un critère pré-enregistré, une garde automatique et un chemin de revert ; un levier à la fois | `repos/pulsar2/CLAUDE.md` (run 9uz761ua : un mécanisme activé sans preuve a ralenti le meilleur run de 10×) | règle du studio + ADR |
| Checkpoints atomiques (tmp + rename), fallback newest→oldest, **golden archive** des top-3, sonde de kickoff au boot, **jamais** de checkpoint hybride | `repos/pulsar2/CLAUDE.md` | spec mlops Phase 3 |
| La parité d'architecture entre entraînement et déploiement est une source de bugs réels (incident 512 vs 517) | `repos/pulsar2/CLAUDE.md` | test de parité automatique |
| Le choix du simulateur change les résultats : RocketSim v2 battait v3 10-3 en jeu réel (contacts voiture-balle) ; Titan a réécrit le simulateur (PerfectSim) pour le sim-to-real | l'annexe privée, ANALYSE §4 (vérifié) | mesurer l'écart sim/jeu dès S2 |
| GigaLearn brut est jugé lent : les gains de débit viennent d'optimisations manuelles du pipeline | Disco [annexe] | profiler avant d'optimiser |
| Le coût SPS des critiques d'exploration varie de 1 % à 50 % selon les sources | [annexe], [annexe], [annexe] | benchmark dédié en Phase 2 |
| Rester petit (MLP) jusqu'au niveau Nexto ; l'itération compte plus que la taille | Terminal [annexe] | v0 = MLP |
| Necto/Nexto : PPO distribué, attention, table de 90 actions, state setter à base de replays | `02_ecosystem.md` §8, Lucy-SKG App. A | 90 actions « default » ; replays = GCO-ext |
| Rocket League PC sous Easy Anti-Cheat depuis le 2026-04-28 ; RLBot lance le jeu sans EAC ; **ne pas** utiliser de dumper mémoire de collision meshes, prendre les meshes livrés avec RLGym | `02_ecosystem.md` §5, risque 4 | règle de sécurité |
