---
name: benchmark
description: "Lance un benchmark de débit reproductible (collecte, apprentissage, bout-en-bout) et consigne le résultat."
argument-hint: "<config de benchmark> [--duration-min N]"
disable-model-invocation: true
---
# /benchmark

1. Vérifier que l'arbre git est propre (commit connu) et que `scripts/benchmark.py` existe. Si le script cité n'existe pas encore (phase pas atteinte), le dire et s'arrêter : ne pas improviser un substitut.
2. Relever le matériel : `python -c "import os,platform;print(platform.processor(), os.cpu_count())"` et l'état secteur/batterie.
3. Lancer `python scripts/benchmark.py $ARGUMENTS` en arrière-plan (durée ≥ 30 min pour un chiffre « soutenu »).
4. Consigner dans `docs/benchmarks/<date>-<nom>.md` : commande, commit, seed, matériel, durée, SPS médian et p10/p90, RAM max.
5. Aucun chiffre ne sort de ce fichier sans sa ligne de provenance.
