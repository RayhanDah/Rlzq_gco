---
name: status
description: "Affiche l'état du projet : phase, gate en attente, git, derniers runs, checkpoints, kernels Kaggle."
argument-hint: "[--kaggle user/slug ...]"
---
# /status

1. Lancer `python scripts/status.py $ARGUMENTS`.
2. Résumer en ≤ 10 lignes : phase et gate, runs en cours ou récents (commit), dernier checkpoint, risques ouverts dans `production/state.md`.
3. Signaler toute incohérence entre `state.md` et la réalité (run déclaré en cours mais absent du journal, etc.).
