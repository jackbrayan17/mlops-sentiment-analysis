# Home

## Vue d'ensemble

Ce wiki documente le pipeline MLOps du projet d'analyse de sentiments sur des
produits technologiques.

## Contenu

- [Git Flow](Git-Flow.md)
- [Deployment](Deployment.md)

## Resume du pipeline

1. developpement par branches `feature/*`
2. integration via Pull Request vers `develop`
3. controle qualite local avec le hook `pre-commit`
4. validation GitHub Actions sur chaque PR
5. deploiement automatique vers Hugging Face lors d'un push sur `main`
