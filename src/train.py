"""
train.py - Script d'entrainement du modele de classification de sentiments.
Sujet : Analyse d'opinions sur les produits tech.
Modele : DistilBERT fine-tune sur SST-2 + fine-tuning supplementaire.
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.json")
NUM_TRAIN_SAMPLES = 2000
NUM_EVAL_SAMPLES = 500
EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
ACCURACY_THRESHOLD = 0.80


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


def main():
    print("=" * 60)
    print("ENTRAINEMENT - Analyse de sentiments (produits tech)")
    print("=" * 60)

    # 1. Charger le tokenizer et le modele pre-entraine
    print(f"\n[1/5] Chargement du modele : {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # 2. Charger le dataset SST-2
    print("[2/5] Chargement du dataset SST-2...")
    dataset = load_dataset("glue", "sst2")

    train_dataset = dataset["train"].select(range(NUM_TRAIN_SAMPLES))
    eval_dataset = dataset["validation"].select(range(NUM_EVAL_SAMPLES))

    # 3. Tokenization
    print("[3/5] Tokenization des donnees...")

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    eval_dataset = eval_dataset.rename_column("label", "labels")

    train_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    eval_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # 4. Configuration de l'entrainement
    print("[4/5] Lancement de l'entrainement...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 5. Evaluation finale et sauvegarde
    print("[5/5] Evaluation finale et sauvegarde...")
    eval_results = trainer.evaluate()

    accuracy = eval_results["eval_accuracy"]
    f1 = eval_results["eval_f1"]

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1, 4),
        "model_name": MODEL_NAME,
        "num_train_samples": NUM_TRAIN_SAMPLES,
        "num_eval_samples": NUM_EVAL_SAMPLES,
        "epochs": EPOCHS,
        "threshold": ACCURACY_THRESHOLD,
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    # Sauvegarder le modele et le tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print(f"RESULTATS :")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  Seuil    : {ACCURACY_THRESHOLD}")
    print(f"  Status   : {'PASS' if accuracy >= ACCURACY_THRESHOLD else 'FAIL'}")
    print(f"{'=' * 60}")

    if accuracy < ACCURACY_THRESHOLD:
        raise ValueError(
            f"Accuracy {accuracy:.4f} < seuil {ACCURACY_THRESHOLD}. "
            "Le modele ne respecte pas les criteres de qualite."
        )

    print(f"\nModele sauvegarde dans : {OUTPUT_DIR}")
    return metrics


if __name__ == "__main__":
    main()
