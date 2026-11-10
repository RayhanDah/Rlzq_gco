---
paths:
  - "src/rlbot_bot/**"
  - "exports/**"
  - "scripts/export_bot.py"
---
# Règles : déploiement RLBot v5
- Bot = sous-classe de `Bot` du paquet `rlbot` installé ; vérifier la signature de `get_output` dans le source avant usage.
- Inférence CPU, latence p99 < 8,33 ms/tick mesurée sur le laptop.
- Le jeu se lance uniquement via RLBot (EAC actif depuis 2026-04-28). Aucun dumper mémoire, aucune injection.
- Parité numérique exportée/originale < 1e-5, testée.
