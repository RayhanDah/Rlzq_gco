---
paths:
  - "src/env/**"
  - "src/obs/**"
  - "src/actions/**"
---
# Règles : env, observations, actions
- Vérifier l'API RLGym/RocketSim installée (version + source) avant tout appel. Pas d'API devinée.
- Les builders d'obs et d'actions sont partagés entre l'entraînement et `src/rlbot_bot/`. Toute modification met à jour le test de parité.
- Symétrie miroir bleu↔orange testée.
- États initiaux : setters génériques seulement (spec §2.3). Aucun setter ciblé sur une mécanique ni issu de replays dans la
  ligne principale.
- Les collision meshes ne sont jamais versionnées (licence des assets) : `data/` local, dataset Kaggle privé.
