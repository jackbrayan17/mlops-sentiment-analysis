# MLOps Sentiment Analysis - Produits Tech

> TP final Git/GitHub - Master 2 IABD
> Pipeline MLOps complet pour classer des avis de produits technologiques.

## Objectif

Le projet industrialise un modele de classification de sentiments avec les
exigences du TP :

- Git Flow avec branches `main`, `develop`, `feature/*` et `release/*`
- Git LFS pour les artefacts lourds
- hook `pre-commit` local pour la qualite et la securite
- workflow GitHub Actions pour test, validation, notification et deploiement
- publication sur Hugging Face Hub et Hugging Face Spaces

## Architecture

```text
mlops-sentiment-analysis/
|-- .github/workflows/main.yml
|-- data/tech_reviews.csv
|-- docs/wiki/
|-- hooks/pre-commit
|-- model/
|-- src/
|   |-- __init__.py
|   |-- inference.py
|   `-- train.py
|-- tests/
|   |-- __init__.py
|   `-- test_model.py
|-- app.py
|-- requirements.txt
|-- .gitattributes
|-- .gitignore
|-- .pre-commit-config.yaml
|-- setup_hooks.sh
`-- README.md
```

## Choix techniques

- Modele : `distilbert-base-uncased-finetuned-sst-2-english`
- Donnees : dataset local versionne `data/tech_reviews.csv`
- Inference partagee : `src/inference.py`
- Interface : Gradio
- CI/CD : GitHub Actions + notifications SMTP + deploiement Hugging Face

Le dataset local rend le projet plus stable en TP et en CI : l'entrainement ne
depend pas d'un dataset externe pour reproduire les resultats.

## Installation

### Prerequis

- Python 3.11 recommande pour la CI
- Git et Git LFS
- un compte GitHub
- un compte Hugging Face

### Setup local

```bash
git clone https://github.com/jackbrayan17/mlops-sentiment-analysis.git
cd mlops-sentiment-analysis

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install -r requirements.txt
git lfs install
bash setup_hooks.sh
```

## Utilisation

### 1. Entrainement du modele

```bash
python -m src.train
```

Ce script :

1. charge le modele de base DistilBERT deja entraine sur SST-2
2. lit les avis du fichier `data/tech_reviews.csv`
3. fine-tune le modele sur les reviews tech
4. exporte le modele dans `model/`
5. ecrit les metriques dans `model/metrics.json`

Variables optionnelles :

```bash
TRAIN_EPOCHS=2
TRAIN_BATCH_SIZE=8
TRAIN_LEARNING_RATE=2e-5
ACCURACY_THRESHOLD=0.80
```

### 2. Tests unitaires

```bash
pytest tests/ -v
```

Les tests valident :

- accuracy >= 0.80
- F1-score >= 0.75
- inference positive et negative coherente
- somme des probabilites = 1
- presence des artefacts du modele

### 3. Application Gradio

```bash
python app.py
```

L'application charge le modele depuis `model/` et permet de tester des avis
tech en temps reel.

## Git Flow

Workflow attendu :

1. `main` contient uniquement le code stable et deploye
2. `develop` sert a integrer les fonctionnalites
3. chaque tache vit dans une branche `feature/*`
4. `release/*` sert a preparer une mise en production

Regles du depot :

- aucun commit direct sur `main`
- passage obligatoire par Pull Request
- validation CI avant merge

Branches deja utilisees pour ce TP :

- `feature/model-training`
- `feature/gradio-app`
- `feature/ci-cd`

## Simulation de travail en equipe

Comme le TP doit simuler un vrai travail collaboratif, on peut reproduire le
fonctionnement de plusieurs membres de l'equipe meme sur une seule machine.

### Repartition simple des roles

- `Jack Brayan` : pilotage du projet, integration sur `develop`, release
- `Membre 1` : entrainement, dataset, tests
- `Membre 2` : application Gradio, UX, demo
- `Membre 3` : CI/CD, hooks, documentation, deploiement

### Regle de travail a chaque update

Pour chaque evolution du TP :

1. partir de `develop`
2. creer une branche `feature/*` ciblee
3. faire une petite serie de commits propres
4. executer les verifications locales
5. ouvrir une Pull Request vers `develop`
6. merger dans `main` seulement quand `develop` est stable

### Exemple de simulation avec plusieurs auteurs

#### Update de Jack

```bash
git checkout develop
git checkout -b feature/training-improvement

git add src/train.py tests/test_model.py
git -c user.name="Jack Brayan" -c user.email="jack@example.com" commit -m "feat: improve training and tests"
```

#### Update d'un membre equipe app

```bash
git checkout develop
git checkout -b feature/gradio-enhancement

git add app.py README.md
git -c user.name="Membre 2" -c user.email="membre2@example.com" commit -m "feat: improve gradio interface"
```

#### Update d'un membre equipe CI/CD

```bash
git checkout develop
git checkout -b feature/ci-update

git add .github/workflows/main.yml hooks/pre-commit README.md
git -c user.name="Membre 3" -c user.email="membre3@example.com" commit -m "chore: improve ci and project docs"
```

### Routine de verification avant chaque merge

Avant de fusionner une branche dans `develop`, executer :

```bash
python -m src.train
python -m pytest tests/ -v --tb=short
python -m flake8 src tests app.py --max-line-length=120 --statistics
python -m black --check src tests app.py
```

### Routine de merge pour simuler le projet en equipe

```bash
git checkout develop
git merge --no-ff feature/training-improvement
git merge --no-ff feature/gradio-enhancement
git merge --no-ff feature/ci-update
```

Quand tout est valide sur `develop` :

```bash
git checkout main
git merge --no-ff develop
git push origin main
```

Ce schema permet de montrer clairement dans l'historique Git :

- qui a travaille sur quoi
- quel type de fonctionnalite a ete ajoute
- comment les branches ont ete integrees
- que la validation a ete faite avant passage sur `main`

## Git LFS

Le fichier `.gitattributes` configure Git LFS pour :

- `*.pkl`
- `*.h5`
- `*.pt`
- `*.onnx`
- `*.bin`
- `*.safetensors`

Commandes utiles :

```bash
git lfs install
git lfs track "*.onnx"
git lfs ls-files
```

## Git Hooks et securite

Le hook `hooks/pre-commit` copie dans `.git/hooks/pre-commit` verifie :

- flake8 sur les fichiers Python stages
- `python -m py_compile` pour attraper les erreurs de syntaxe
- black en mode `--check`
- detect-secrets ou detect-secrets-hook
- blocage des fichiers > 5 Mo non suivis par Git LFS

Le fichier `.pre-commit-config.yaml` complete ce dispositif avec les hooks
`check-added-large-files`, `black`, `flake8` et `detect-secrets`.

Secrets GitHub a configurer :

- `HF_TOKEN`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `TEAM_EMAILS`

## Pipeline CI/CD

Le workflow `.github/workflows/main.yml` s'execute :

- a chaque Pull Request vers `develop` ou `main`
- a chaque push sur `main`

### Etape 1 - Test et validation

- installation des dependances
- lint `flake8`
- verification de formatage `black --check`
- entrainement du modele
- lecture des metriques exportees
- execution des tests `pytest`
- publication des artefacts `test-results` et `trained-model`

### Etape 2 - Notification par email

Le job `notify` envoie un mail automatique via
`dawidd6/action-send-mail` avec :

- le statut du pipeline
- le repository, la branche, le commit et l'auteur
- les metriques du modele en cas de succes
- le lien direct vers l'execution GitHub Actions

### Etape 3 - Deploiement Hugging Face

Sur un push vers `main`, le job `deploy` :

1. telecharge l'artefact `trained-model`
2. pousse les artefacts du modele vers `jackbrayan17/mlops-sentiment-analysis`
3. pousse `app.py`, `requirements.txt` et `model/` vers le Space
   `jackbrayan17/mlops-sentiment-analysis-demo`

## Documentation wiki

Le contenu de wiki a ete prepare dans `docs/wiki/` pour faciliter la mise en
ligne du wiki GitHub :

- `Home.md`
- `Git-Flow.md`
- `Deployment.md`

## Livrables TP

Le depot contient deja :

- un projet Python structure
- un hook de pre-commit versionne
- une configuration Git LFS
- un workflow GitHub Actions complet
- un script d'entrainement
- des tests unitaires
- une application Gradio
- une base de documentation README + wiki source

## Hugging Face

- Modele : [jackbrayan17/mlops-sentiment-analysis](https://huggingface.co/jackbrayan17/mlops-sentiment-analysis)
- Application : [jackbrayan17/mlops-sentiment-analysis-demo](https://huggingface.co/spaces/jackbrayan17/mlops-sentiment-analysis-demo)

## Equipe

Projet realise dans le cadre du Master 2 IABD.
