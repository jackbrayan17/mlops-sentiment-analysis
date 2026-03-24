"""
test_model.py - Tests unitaires pour le modele de classification de sentiments.
Verifie les performances et le bon fonctionnement du pipeline d'inference.
"""

import os
import json
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
ACCURACY_THRESHOLD = 0.80


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Charge le modele et le tokenizer sauvegardes."""
    if not os.path.exists(MODEL_DIR):
        pytest.skip("Modele non trouve. Lancez train.py d'abord.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def metrics():
    """Charge les metriques sauvegardees."""
    if not os.path.exists(METRICS_FILE):
        pytest.skip("Fichier metrics.json non trouve.")
    with open(METRICS_FILE, "r") as f:
        return json.load(f)


class TestModelPerformance:
    """Tests de performance du modele."""

    def test_accuracy_above_threshold(self, metrics):
        """Verifie que l'accuracy depasse le seuil de 80%."""
        accuracy = metrics["accuracy"]
        assert accuracy >= ACCURACY_THRESHOLD, (
            f"Accuracy {accuracy:.4f} < seuil {ACCURACY_THRESHOLD}"
        )

    def test_f1_score_above_threshold(self, metrics):
        """Verifie que le F1-score est acceptable."""
        f1 = metrics["f1_score"]
        assert f1 >= 0.75, f"F1-score {f1:.4f} trop bas (< 0.75)"


class TestModelInference:
    """Tests d'inference du modele."""

    def test_positive_sentiment(self, model_and_tokenizer):
        """Verifie que le modele detecte un sentiment positif."""
        model, tokenizer = model_and_tokenizer
        text = "This product is amazing, I love it!"
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        assert prediction == 1, f"Attendu positif (1), obtenu {prediction}"

    def test_negative_sentiment(self, model_and_tokenizer):
        """Verifie que le modele detecte un sentiment negatif."""
        model, tokenizer = model_and_tokenizer
        text = "This is terrible, worst purchase ever."
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        assert prediction == 0, f"Attendu negatif (0), obtenu {prediction}"

    def test_output_shape(self, model_and_tokenizer):
        """Verifie la forme de la sortie du modele."""
        model, tokenizer = model_and_tokenizer
        text = "This is a test sentence."
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.logits.shape == (1, 2), (
            f"Shape attendue (1, 2), obtenue {outputs.logits.shape}"
        )

    def test_probabilities_sum_to_one(self, model_and_tokenizer):
        """Verifie que les probabilites somment a 1."""
        model, tokenizer = model_and_tokenizer
        text = "Just a regular tech product."
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        total = probs.sum().item()
        assert abs(total - 1.0) < 1e-5, f"Somme des probs = {total}, attendu 1.0"

    def test_batch_inference(self, model_and_tokenizer):
        """Verifie l'inference par batch."""
        model, tokenizer = model_and_tokenizer
        texts = [
            "Great product, highly recommend!",
            "Awful quality, do not buy.",
            "It's okay, nothing special.",
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.logits.shape[0] == 3, "Batch size incorrect"

    def test_model_files_exist(self):
        """Verifie que tous les fichiers du modele sont presents."""
        required_files = ["config.json", "metrics.json"]
        for fname in required_files:
            fpath = os.path.join(MODEL_DIR, fname)
            assert os.path.exists(fpath), f"Fichier manquant : {fname}"
