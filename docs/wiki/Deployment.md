# Deployment

## Secrets requis

- `HF_TOKEN`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `TEAM_EMAILS`

## Workflow

Le job `deploy` du workflow GitHub Actions s'execute uniquement lors d'un push
sur `main`.

## Hugging Face Hub

Le modele exporte dans `model/` est pousse vers :

- `jackbrayan17/mlops-sentiment-analysis`

## Hugging Face Spaces

L'application Gradio et le dossier `model/` sont pousses vers :

- `jackbrayan17/mlops-sentiment-analysis-demo`

## Verification post-deploiement

1. verifier que le Space demarre sans erreur
2. tester un avis positif et un avis negatif
3. verifier la presence des fichiers du modele sur le Hub
