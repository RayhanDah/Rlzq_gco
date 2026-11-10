---
name: export-bot
description: "Exporte un checkpoint en ONNX/TorchScript, vérifie parité numérique et latence, empaquette le bot RLBot v5."
argument-hint: "<checkpoint>"
disable-model-invocation: true
---
# /export-bot

1. Vérifier que `scripts/export_bot.py` existe. Si le script cité n'existe pas encore (phase pas atteinte), le dire et s'arrêter : ne pas improviser un substitut.
2. `python scripts/export_bot.py --ckpt $ARGUMENTS --out exports/<nom>`.
3. Exiger : parité des sorties (écart max < 1e-5 sur 10 000 obs de test), parité obs/actions via rlgym-compat,
   latence p99 < 8,33 ms/tick sur le laptop (mesurée, CPU).
4. Générer `bot.toml` pour RLBot v5 ; tester un match local contre un bot du botpack.
5. Rapport `docs/reports/export-<nom>.md`.
