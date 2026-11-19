---
paths:
  - "src/rewards/**"
  - "tests/unit/test_rewards*.py"
---
# Règles : récompenses
- Le signal est exactement `docs/GCO_SPEC.md` §2.1 : ±1 par équipe au but, csc selon le but touché, 0 sinon. Aucun shaping.
  Exception unique, étiquetée HORS SPEC : la variante de diagnostic `dense-diag` (`src/rewards/dense.py`, `docs/adr/011-temoin-dense-hors-spec.md`),
  jamais promue ni exportée ; les tests §7 restent verts pour les variantes in-spec.
- Toute variante (GCO-S, GCO-std, GCO-ext, scorer-only) passe par l'interface de variantes et est sélectionnée par config.
- Le hash de la config de récompense est écrit dans chaque checkpoint.
- Les 8 tests de la spec §7 doivent rester verts. Modifier `src/rewards/` exige de mettre à jour la spec et d'écrire un ADR.
- Un signal d'exploration appris n'est pas une récompense : il vit dans `src/training/`, respecte C1–C4 et est journalisé
  (alerte si > 50 % de la variance des avantages).
