"""Script d'entrainement du modele de sentiment sur des avis tech locaux."""

from __future__ import annotations

import csv
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "tech_reviews.csv"
MODEL_DIR = PROJECT_ROOT / "model"
METRICS_FILE = MODEL_DIR / "metrics.json"
TRAINING_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "training"
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "distilbert-base-uncased-finetuned-sst-2-english",
)
EPOCHS = int(os.getenv("TRAIN_EPOCHS", "2"))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "8"))
LEARNING_RATE = float(os.getenv("TRAIN_LEARNING_RATE", "2e-5"))
MAX_LENGTH = int(os.getenv("TRAIN_MAX_LENGTH", "128"))
ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.80"))
SEED = int(os.getenv("TRAIN_SEED", "42"))


class SentimentDataset(Dataset):
    """Dataset PyTorch minimal pour les reviews tech."""

    def __init__(self, tokenizer, rows: list[dict[str, str]]):
        texts = [row["text"] for row in rows]
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )
        self.labels = [int(row["label"]) for row in rows]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value[index]) for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index])
        return item


def load_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Charge les jeux train/test depuis le CSV versionne."""
    with DATA_FILE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "test"]
    if not train_rows or not eval_rows:
        raise ValueError("Le dataset doit contenir des lignes `train` et `test`.")
    return train_rows, eval_rows


def compute_metrics(eval_pred) -> dict[str, float]:
    """Calcule les metriques suivies dans le pipeline."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }


def ensure_directories() -> None:
    """Cree les dossiers d'artefacts si besoin."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_metrics(metrics: dict[str, float]) -> None:
    """Serialise les metriques pour les tests et la CI."""
    with METRICS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> dict[str, float]:
    """Entraine, evalue et sauvegarde le modele."""
    set_seed(SEED)
    ensure_directories()

    print("=" * 72)
    print("ENTRAINEMENT - Analyse de sentiments sur avis de produits tech")
    print("=" * 72)
    print(f"[1/5] Chargement du modele de base : {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    print(f"[2/5] Lecture du dataset local : {DATA_FILE}")
    train_rows, eval_rows = load_rows()
    print(
        f"       {len(train_rows)} exemples train | " f"{len(eval_rows)} exemples test"
    )

    print("[3/5] Preparation des jeux tokenises")
    train_dataset = SentimentDataset(tokenizer, train_rows)
    eval_dataset = SentimentDataset(tokenizer, eval_rows)

    print("[4/5] Fine-tuning du modele")
    training_kwargs = {
        "output_dir": str(TRAINING_OUTPUT_DIR),
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.01,
        "logging_strategy": "epoch",
        "report_to": "none",
        "seed": SEED,
    }
    training_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_signature.parameters:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    print("[5/5] Evaluation finale et export des artefacts")
    evaluation = trainer.evaluate()
    accuracy = float(evaluation["eval_accuracy"])
    f1_score_value = float(evaluation["eval_f1"])

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1_score_value, 4),
        "threshold": ACCURACY_THRESHOLD,
        "model_name": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "dataset_path": DATA_FILE.relative_to(PROJECT_ROOT).as_posix(),
        "seed": SEED,
    }

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    save_metrics(metrics)

    print("=" * 72)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-score : {f1_score_value:.4f}")
    print(f"Seuil    : {ACCURACY_THRESHOLD:.2f}")
    print("Statut   : " f"{'PASS' if accuracy >= ACCURACY_THRESHOLD else 'FAIL'}")
    print(f"Modele   : {MODEL_DIR}")
    print("=" * 72)

    if accuracy < ACCURACY_THRESHOLD:
        raise ValueError(f"Accuracy {accuracy:.4f} < seuil {ACCURACY_THRESHOLD:.2f}.")

    return metrics


if __name__ == "__main__":
    main()
