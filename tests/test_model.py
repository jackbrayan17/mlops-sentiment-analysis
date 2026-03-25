"""Tests unitaires du modele de sentiment et du pipeline d'inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.inference import predict_label, predict_probabilities


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
METRICS_FILE = MODEL_DIR / "metrics.json"
ACCURACY_THRESHOLD = 0.80


@pytest.fixture(scope="module")
def metrics() -> dict[str, float]:
    """Charge les metriques serialisees apres entrainement."""
    if not METRICS_FILE.exists():
        pytest.skip("Modele non entraine. Lancez `python -m src.train`.")
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Charge le bundle du modele pour les assertions bas niveau."""
    if not MODEL_DIR.exists():
        pytest.skip("Dossier modele absent.")

    from src.inference import load_model_bundle

    return load_model_bundle(MODEL_DIR)


class TestModelPerformance:
    """Controle des metriques minimales imposees par le TP."""

    def test_accuracy_above_threshold(self, metrics):
        accuracy = metrics["accuracy"]
        assert (
            accuracy >= ACCURACY_THRESHOLD
        ), f"Accuracy {accuracy:.4f} < seuil {ACCURACY_THRESHOLD:.2f}"

    def test_f1_score_above_threshold(self, metrics):
        f1_score = metrics["f1_score"]
        assert f1_score >= 0.75, f"F1-score {f1_score:.4f} < seuil 0.75"

    def test_metrics_contains_dataset_metadata(self, metrics):
        assert metrics["train_examples"] > 0
        assert metrics["eval_examples"] > 0
        assert metrics["dataset_path"] == "data/tech_reviews.csv"


class TestModelInference:
    """Verifie le comportement d'inference du modele exporte."""

    def test_positive_prediction(self):
        label, score = predict_label(
            "This headset sounds fantastic and the battery life is excellent."
        )
        assert label == "Positive"
        assert score > 0.50

    def test_negative_prediction(self):
        label, score = predict_label(
            "This smartphone freezes all the time and the camera is awful."
        )
        assert label == "Negative"
        assert score > 0.50

    def test_probabilities_sum_to_one(self):
        probabilities = predict_probabilities(
            "A reliable monitor with sharp colors and a sturdy stand."
        )
        total = sum(probabilities.values())
        assert abs(total - 1.0) < 1e-5

    def test_batch_shape(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        texts = [
            "Amazing keyboard, very comfortable to type on.",
            "The tablet is slow and the speakers crackle.",
            "Good laptop for work and meetings.",
        ]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**encoded)
        assert outputs.logits.shape == (3, 2)

    def test_model_files_exist(self):
        required_files = [
            "config.json",
            "metrics.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ]
        for filename in required_files:
            assert (MODEL_DIR / filename).exists(), f"Artefact manquant : {filename}"
