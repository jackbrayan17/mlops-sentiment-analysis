"""
app.py - Interface Gradio pour l'analyse de sentiments sur les produits tech.
Deploye sur Hugging Face Spaces.
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_PATH = "model"
LABELS = {0: "Negatif 👎", 1: "Positif 👍"}


def load_model():
    """Charge le modele et le tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()


def predict_sentiment(text: str) -> dict:
    """Predit le sentiment d'un texte donne."""
    if not text.strip():
        return {v: 0.0 for v in LABELS.values()}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    return {
        LABELS[i]: float(probabilities[i]) for i in range(len(LABELS))
    }


EXAMPLES = [
    ["This laptop is incredible, the battery lasts all day!"],
    ["The phone screen broke after just one week, terrible quality."],
    ["Average product, does what it's supposed to do."],
    ["Best headphones I've ever owned, crystal clear sound!"],
    ["The software is buggy and crashes constantly, very disappointed."],
    ["Ce produit est fantastique, je le recommande vivement !"],
]

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Saisissez un avis sur un produit tech...",
        label="Avis produit",
    ),
    outputs=gr.Label(
        num_top_classes=2,
        label="Analyse de sentiment",
    ),
    title="🔍 Analyse de Sentiments - Produits Tech",
    description=(
        "Entrez un avis sur un produit technologique et le modele "
        "predira s'il est **positif** ou **negatif**.\n\n"
        "Modele : DistilBERT fine-tune sur SST-2 | "
        "Projet MLOps M2 IABD"
    ),
    examples=EXAMPLES,
    theme=gr.themes.Soft(),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
