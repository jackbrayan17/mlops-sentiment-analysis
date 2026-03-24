"""Interface Gradio pour tester le modele de sentiment en direct."""

from __future__ import annotations

import gradio as gr

from src.inference import predict_label, predict_probabilities


DISPLAY_LABELS = {
    "Negative": "Negatif",
    "Positive": "Positif",
}
EXAMPLES = [
    ["This laptop is fast, quiet, and the battery easily lasts a full day."],
    ["The phone gets hot after five minutes and the camera quality is poor."],
    ["Excellent headset, crisp audio and a microphone that sounds clear."],
    ["The software crashes every time I try to export a file."],
]


def analyze_review(text: str) -> tuple[dict[str, float], str]:
    """Retourne la prediction formatee pour Gradio."""
    if not text.strip():
        return (
            {label: 0.0 for label in DISPLAY_LABELS.values()},
            "Saisissez un avis pour lancer l'analyse.",
        )

    try:
        probabilities = predict_probabilities(text)
        predicted_label, confidence = predict_label(text)
    except FileNotFoundError as error:
        raise gr.Error(str(error)) from error

    translated_scores = {
        DISPLAY_LABELS[label]: score for label, score in probabilities.items()
    }
    translated_label = DISPLAY_LABELS[predicted_label]
    summary = (
        f"Prediction principale : {translated_label}\n" f"Confiance : {confidence:.1%}"
    )
    return translated_scores, summary


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as demo:
    gr.Markdown(
        """
        # Analyse de sentiments sur des produits tech
        Pipeline MLOps du TP final M2 IABD.

        L'application charge le modele entraine dans `model/` et retourne un
        score de probabilite pour les classes positive et negative.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            review_input = gr.Textbox(
                lines=5,
                label="Avis produit",
                placeholder=(
                    "Exemple : The smartwatch looks premium, but the battery "
                    "drains before the end of the day."
                ),
            )
            submit_button = gr.Button("Analyser", variant="primary")
        with gr.Column(scale=2):
            sentiment_output = gr.Label(
                num_top_classes=2,
                label="Score du modele",
            )
            summary_output = gr.Textbox(
                label="Resume",
                interactive=False,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=review_input,
    )

    submit_button.click(
        fn=analyze_review,
        inputs=review_input,
        outputs=[sentiment_output, summary_output],
    )
    review_input.submit(
        fn=analyze_review,
        inputs=review_input,
        outputs=[sentiment_output, summary_output],
    )


if __name__ == "__main__":
    demo.launch()
