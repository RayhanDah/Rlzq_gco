---
name: smoke-test
description: "Smoke test d'entraînement court (local ou Kaggle) : pas de NaN, shapes, reprise de checkpoint, débit grossier."
argument-hint: "[local|kaggle-tpu|kaggle-t4]"
---
# /smoke-test

1. Vérifier que `scripts/smoke_test.py` existe. Si le script cité n'existe pas encore (phase pas atteinte), le dire et s'arrêter : ne pas improviser un substitut.
2. Local par défaut : `python scripts/smoke_test.py --steps 1000000 --seed 0`.
   Kaggle : pousser le kernel `kaggle/smoke` avec `kaggle kernels push -p kaggle/smoke` (accélérateur selon l'argument), timeout ≤ 30 min.
3. Critères PASS (tous) : aucune NaN/Inf ; reward non nulle uniquement sur but ; reprise depuis checkpoint donne des poids identiques ;
   journal TensorBoard écrit ; SPS consigné.
4. Écrire `docs/reports/smoke_test.md` (commande, commit, seed, matériel, résultat de chaque critère). Verdict PASS/FAIL.
