# Protocole de coordination du studio

## Hiérarchie
```
utilisateur (gates G1/G2/G3, GO des runs longs)
├── research-director : spec GCO, priorités R&D, kill criteria
│   ├── rl-algo-lead → reward-engineer, obs-action-engineer
│   └── eval-lead → experiment-analyst
│   (+ paper-researcher)
└── tech-director : architecture, ADR, gates, intégration
    ├── infra-perf-lead → sim-engineer, profiler
    ├── mlops-lead
    └── integration-lead → rlbot-engineer
    (+ test-engineer, code-reviewer, doc-writer)
```
La session principale joue le rôle du directeur concerné et délègue aux leads et aux spécialistes.

## Règles
1. **Délégation verticale.** Un agent reste dans son domaine. Une décision qui sort de ce domaine remonte au parent, et un
   conflit entre deux leads remonte au directeur commun (ou à l'utilisateur s'ils dépendent de directeurs différents).
2. **Consultation horizontale.** Un lead peut consulter un autre lead, mais pas décider à sa place.
3. **Un writer par ensemble de fichiers.**
   - Deux tâches parallèles qui touchent des fichiers différents tournent chacune dans son worktree
     (`isolation: worktree`, branche `wt/<tâche>`).
   - Deux tâches qui touchent les mêmes fichiers sont séquentielles.
4. **Cycle d'une tâche de code** :
   1. spec ou critère de fin écrit ;
   2. test qui échoue ;
   3. implémentation ;
   4. preuve (commande + sortie) ;
   5. `/review` par `code-reviewer`, verdict APPROVE requis ;
   6. merge dans `main`.
5. **Aucun run d'entraînement dans un subagent.**
   - Les runs sont lancés par `/train` : processus détaché ou kernel Kaggle.
   - Ils sont journalisés automatiquement (`production/run_log.jsonl`) et suivis par `/status`.
6. **Gates.** `/gate-check` produit un verdict mécanique. Seul l'utilisateur franchit un gate, et son GO est consigné
   dans `production/state.md`.
7. **Mémoire projet.** `production/state.md` est la source de vérité de l'état. Les décisions vont dans `docs/adr/`, les
   résultats dans `docs/reports/` et `docs/postmortems/`.
8. **Mesure avant machinerie.** Toute intervention sur l'entraînement exige quatre choses avant d'être lancée :
   - une mesure qui montre le problème ;
   - un critère pré-enregistré ;
   - une garde automatique ;
   - un chemin de retour arrière.

   Un seul levier change à la fois.

## Branches
- `main` : toujours vert (tests des hooks et tests unitaires).
- `wt/<tâche>` : travail isolé, mergé après APPROVE.
- Jamais de force-push ni de réécriture de `main` (les hooks le refusent).
