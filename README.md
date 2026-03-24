# 🔍 MLOps Sentiment Analysis - Produits Tech

> **TP Final Git & GitHub - Master 2 IABD**
> Pipeline MLOps complet pour l'analyse de sentiments sur les avis de produits technologiques.

## 📋 Table des matières

- [Objectif](#objectif)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Git Flow](#git-flow)
- [Git LFS](#git-lfs)
- [Git Hooks](#git-hooks)
- [Sécurité](#sécurité)
- [CI/CD Pipeline](#cicd-pipeline)
- [Déploiement Hugging Face](#déploiement-hugging-face)
- [Équipe](#équipe)

## 🎯 Objectif

Développer et industrialiser un modèle d'IA pour la **classification de sentiments** sur les avis de produits tech, en utilisant les meilleures pratiques MLOps :
- Versionnement du code (Git Flow)
- Gestion des fichiers volumineux (Git LFS)
- Automatisation (Git Hooks + GitHub Actions)
- Déploiement continu (Hugging Face Hub & Spaces)

## 🏗️ Architecture

```
mlops-sentiment-analysis/
├── .github/
│   └── workflows/
│       └── main.yml              # Pipeline CI/CD
├── hooks/
│   └── pre-commit                # Hook de pré-commit
├── src/
│   ├── __init__.py
│   └── train.py                  # Script d'entraînement
├── tests/
│   ├── __init__.py
│   └── test_model.py             # Tests unitaires
├── model/                        # Modèle entraîné (Git LFS)
├── app.py                        # Interface Gradio
├── requirements.txt              # Dépendances Python
├── .gitattributes                # Configuration Git LFS
├── .gitignore                    # Fichiers ignorés
├── .pre-commit-config.yaml       # Configuration pre-commit
├── setup_hooks.sh                # Script d'installation des hooks
└── README.md                     # Ce fichier
```

## ⚙️ Installation

### Prérequis

- Python 3.11+
- Git avec Git LFS
- Compte GitHub
- Compte Hugging Face

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/jackbrayan17/mlops-sentiment-analysis.git
cd mlops-sentiment-analysis

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Installer Git LFS
git lfs install

# 5. Installer les Git Hooks
bash setup_hooks.sh
```

## 🚀 Utilisation

### Entraînement du modèle

```bash
python -m src.train
```

Le script :
1. Charge le modèle pré-entraîné DistilBERT (SST-2)
2. Fine-tune sur un sous-ensemble du dataset
3. Évalue les performances (accuracy > 80%)
4. Sauvegarde le modèle dans `model/`

### Tests

```bash
pytest tests/ -v
```

Les tests vérifient :
- Accuracy > 80% (seuil de qualité)
- F1-score > 75%
- Inférence correcte (sentiments positifs/négatifs)
- Forme des sorties du modèle
- Présence des fichiers du modèle

### Interface Gradio

```bash
python app.py
```

Ouvre une interface web pour tester le modèle en temps réel.

## 🌿 Git Flow

Nous suivons rigoureusement le **Git Flow** :

| Branche | Rôle |
|---------|------|
| `main` | Code stable et déployé |
| `develop` | Branche d'intégration |
| `feature/*` | Nouvelles fonctionnalités |
| `release/*` | Préparation de la mise en production |

**Règles :**
- ❌ Aucun commit direct sur `main`
- ✅ Toujours passer par des Pull Requests
- ✅ Review obligatoire avant merge

## 📦 Git LFS

Les fichiers volumineux sont suivis par **Git LFS** :

```bash
# Fichiers trackés
*.pkl, *.h5, *.pt, *.onnx, *.bin, *.safetensors

# Vérifier les fichiers LFS
git lfs ls-files

# Ajouter un nouveau type
git lfs track "*.onnx"
```

## 🪝 Git Hooks

### Hook de pré-commit

Le hook `.git/hooks/pre-commit` vérifie automatiquement :

1. **Syntaxe** : `flake8` (PEP 8 compliance)
2. **Formatage** : `black` (formatage automatique)
3. **Secrets** : `detect-secrets` (détection de clés API)
4. **Taille fichiers** : Bloque les fichiers > 5 Mo non trackés par LFS

### Installation

```bash
bash setup_hooks.sh
```

## 🔒 Sécurité

### detect-secrets (local)

```bash
# Scanner le projet
detect-secrets scan > .secrets.baseline

# Vérifier avant un commit
detect-secrets scan --baseline .secrets.baseline
```

### GitHub Secrets

Les tokens sensibles sont stockés dans **Settings > Secrets** du dépôt :

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Token Hugging Face pour le déploiement |
| `SMTP_USERNAME` | Adresse email pour les notifications |
| `SMTP_PASSWORD` | Mot de passe d'application Gmail |
| `TEAM_EMAILS` | Emails de l'équipe (séparés par des virgules) |

## 🔄 CI/CD Pipeline

Le workflow GitHub Actions (`.github/workflows/main.yml`) s'exécute à chaque PR vers `develop` ou `main` :

### Étape 1 : Tests et Validation
- Vérification syntaxe (flake8)
- Vérification formatage (black)
- Entraînement du modèle
- Tests unitaires (pytest)
- Validation accuracy > 80%

### Étape 2 : Notification par Mail
- Email automatique en cas de succès ou d'échec
- Contient le rapport complet des analyses
- Envoyé via `dawidd6/action-send-mail`

### Étape 3 : Déploiement (main uniquement)
- Upload du modèle sur **Hugging Face Hub**
- Mise à jour de l'application sur **Hugging Face Spaces**

## 🤗 Déploiement Hugging Face

- **Modèle** : [jackbrayan17/mlops-sentiment-analysis](https://huggingface.co/jackbrayan17/mlops-sentiment-analysis)
- **Application** : [jackbrayan17/mlops-sentiment-analysis-demo](https://huggingface.co/spaces/jackbrayan17/mlops-sentiment-analysis-demo)

## 👥 Équipe

Projet réalisé dans le cadre du Master 2 IABD.

---

*Pipeline MLOps industrialisé avec Git Flow, Git LFS, GitHub Actions et Hugging Face.*
