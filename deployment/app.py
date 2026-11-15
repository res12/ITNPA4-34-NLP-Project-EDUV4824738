"""Streamlit web application for Researchly Research Abstract Classifier (Q2.3)."""

import sys
from pathlib import Path

import joblib
import streamlit as st

DEPLOY_DIR = Path(__file__).resolve().parent
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

import text_preprocessor  # noqa: E402, F401  # Required for joblib pipeline deserialisation

MODEL_PATH = DEPLOY_DIR / "researchly_best_model.joblib"

DEFAULT_SAMPLE = (
    "Artificial intelligence (AI) has become a transformative technology in healthcare by enabling more "
    "accurate diagnosis, predictive analytics, and personalised treatment planning. Machine learning algorithms "
    "can analyse large volumes of medical data to identify patterns that assist clinicians in detecting "
    "diseases at earlier stages. In addition, AI-powered decision support systems improve clinical workflows "
    "by reducing manual effort and enhancing diagnostic consistency. Despite these benefits, challenges such "
    "as data privacy, model interpretability, and ethical considerations remain significant barriers to "
    "widespread adoption. Continued research is required to develop robust, transparent, and trustworthy AI "
    "systems that can be safely integrated into modern healthcare environments."
)


@st.cache_resource
def load_classifier_bundle():
    """Load the serialised Researchly NLP pipeline, threshold, and label map."""
    return joblib.load(MODEL_PATH)


def predict_abstract(text: str, bundle: dict) -> tuple[str, float]:
    """Run denoising and classification, returning predicted class and P(AI)."""
    pipeline = bundle["pipeline"]
    threshold = float(bundle.get("threshold", 0.5))
    labels = bundle.get("labels", {0: "classical research", 1: "artificial intelligence"})
    prob_ai = float(pipeline.predict_proba([text])[0, 1])
    pred_idx = 1 if prob_ai >= threshold else 0
    return labels[pred_idx], prob_ai


def main():
    st.set_page_config(
        page_title="Researchly Research Abstract Classifier",
        layout="centered",
    )

    st.title("Researchly Research Abstract Classifier")
    st.write("This app classifies research abstracts as either AI-related or classical research.")

    abstract_input = st.text_area(
        "Enter research abstract:",
        value=DEFAULT_SAMPLE,
        height=195,
    )

    if st.button("Classify Abstract"):
        if not abstract_input.strip():
            st.warning("Please enter a research abstract before running classification.")
            return

        bundle = load_classifier_bundle()
        predicted_label, prob_ai = predict_abstract(abstract_input, bundle)

        st.subheader("Prediction")
        st.success(f"The abstract is classified as: {predicted_label}")
        st.caption(
            f"Estimated AI probability: {prob_ai:.2%} "
            f"(calibrated decision threshold: {bundle.get('threshold', 0.5):.2f})"
        )


if __name__ == "__main__":
    main()
