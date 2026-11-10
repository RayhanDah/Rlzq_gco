---
name: review
description: "Revue indépendante du diff courant (ou d'une branche) par code-reviewer contre la spec et les ADR."
argument-hint: "[branche ou plage de commits]"
---
# /review

1. Obtenir le diff : `git diff` (ou `git diff main...$ARGUMENTS`).
2. Déléguer à l'agent `code-reviewer` (qui n'a pas écrit ce code) avec : le diff, `docs/GCO_SPEC.md`, les ADR concernés.
   Consigne : ne rapporter que ce qui affecte la correction ; ignorer le style.
3. Restituer le verdict APPROVE / WARN / BLOCK avec `fichier:ligne`. BLOCK ⇒ pas de merge.
