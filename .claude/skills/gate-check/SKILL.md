---
name: gate-check
description: "Vérifie mécaniquement un gate (G1, G2, G3) et prépare la présentation à l'utilisateur. Ne franchit jamais le gate seul."
argument-hint: "<G1|G2|G3>"
---
# /gate-check

1. Lancer `python scripts/gate_check.py $ARGUMENTS` et recopier la sortie telle quelle.
2. Verdict = celui du script. FAIL ⇒ lister les artefacts manquants et la skill/agent qui les produit ; ne rien corriger ici.
3. « PRÊT POUR REVUE HUMAINE » ⇒ présenter à l'utilisateur : résumé ≤ 15 lignes, mesures clés avec provenance,
   risques, questions groupées. Attendre son GO explicite ; le consigner dans `production/state.md`.
