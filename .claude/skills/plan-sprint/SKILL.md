---
name: plan-sprint
description: "Planifie le prochain sprint : tâches petites et vérifiables, assignées aux agents, avec critères de fin."
argument-hint: "[objectif du sprint]"
---
# /plan-sprint

1. Lire `production/state.md`, `docs/PLAN.md` et les ADR récents.
2. Découper l'objectif ($ARGUMENTS ou la prochaine action de state.md) en 3–8 tâches de ≤ 1 jour, chacune avec :
   agent responsable, fichiers touchés (un writer par ensemble de fichiers), test ou mesure qui prouve la fin, dépendances.
3. Marquer les tâches parallélisables (worktree) et celles qui exigent un gate humain.
4. Écrire le sprint dans `production/sprints/<date>.md` et le résumer en ≤ 15 lignes. Ne rien lancer.
