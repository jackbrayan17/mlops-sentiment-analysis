# Git Flow

## Branches utilisees

- `main` : branche stable et deployable
- `develop` : branche d'integration
- `feature/model-training` : script d'entrainement et tests
- `feature/gradio-app` : interface utilisateur
- `feature/ci-cd` : automatisation, hooks et workflow GitHub Actions

## Regles

1. aucune modification directe sur `main`
2. chaque fonctionnalite vit dans une branche `feature/*`
3. les merges se font via Pull Request
4. la CI doit passer avant integration

## Recommandation release

Avant rendu final, creer une branche `release/v1.0` depuis `develop`,
verifier la documentation, puis fusionner vers `main`.
