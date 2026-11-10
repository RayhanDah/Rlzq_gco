---
paths:
  - "src/training/**"
  - "src/models/**"
  - "configs/**"
  - "scripts/train*.py"
  - "scripts/kaggle_*.py"
---
# Règles : entraînement
- Seeds fixées et journalisées (python, numpy, torch/jax, env). Config versionnée, son hash est dans le checkpoint.
- Checkpoint atomique : écrire dans un fichier temporaire, puis rename. Il contient poids, optimiseur, normaliseurs, RNG,
  pool self-play, compteurs de steps, commit et hash de config. La reprise doit être exacte (test obligatoire).
- Garde automatique : NaN/Inf, ou KL approx > 0,05 pendant 5 itérations ⇒ arrêt et restauration du dernier checkpoint sain.
- Kaggle : sessions ≤ 9 h (TPU) ou ≤ 12 h (GPU). Sauvegarder dans `/kaggle/working` au moins toutes les 20 min et avant
  le timeout. Ne jamais écrire le jeton dans un kernel ou un log.
- Chaque hyperparamètre a sa source (`docs/01_papers.md` §2) ou une mesure. Un seul levier change à la fois entre deux runs comparés.
- `terminated` (but) sans bootstrap ; `truncated` (timeouts) avec bootstrap sur V(s′).
