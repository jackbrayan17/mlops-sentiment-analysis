"""Fonctions partagees pour charger le modele et lancer l'inference."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "model"))
LABELS = {0: "Negative", 1: "Positive"}


@lru_cache(maxsize=2)
def load_model_bundle(model_dir: str | Path | None = None):
    """Charge le modele et le tokenizer depuis le dossier d'artefacts."""
    resolved_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if not resolved_dir.exists():
        raise FileNotFoundError(
            f"Le dossier modele est introuvable: {resolved_dir}. "
            "Lancez d'abord `python -m src.train`."
        )

    tokenizer = AutoTokenizer.from_pretrained(resolved_dir)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_dir)
    model.eval()
    return model, tokenizer


def predict_probabilities(
    text: str,
    model_dir: str | Path | None = None,
) -> dict[str, float]:
    """Retourne les probabilites positive/negative pour un texte."""
    cleaned_text = text.strip()
    if not cleaned_text:
        return {label: 0.0 for label in LABELS.values()}

    model, tokenizer = load_model_bundle(model_dir)
    encoded = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**encoded)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    return {LABELS[index]: float(probabilities[index]) for index in range(len(LABELS))}


def predict_label(
    text: str,
    model_dir: str | Path | None = None,
) -> tuple[str, float]:
    """Retourne la classe dominante et son score de confiance."""
    probabilities = predict_probabilities(text, model_dir=model_dir)
    label, score = max(probabilities.items(), key=lambda item: item[1])
    return label, score
