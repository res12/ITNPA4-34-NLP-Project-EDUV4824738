# Researchly Research Abstract Classifier — Deployment Guide (Questions 1 & 2)

This folder packages the production NLP inference interface for **Researchly** (`ITNPA4-34 Project 1`, Questions 1 and 2). It deploys the top-performing Hybrid Word + Character N-Gram TF-IDF Soft-Voting Ensemble (`LogisticRegression` + calibrated `LinearSVC` + `ComplementNB`, tuned via Stratified 5-Fold `GridSearchCV` and out-of-fold probability thresholding) inside a Streamlit web application.

## Folder Contents

| File | Purpose |
| :--- | :--- |
| `app.py` | Streamlit web application replicating the Researchly classifier UI (`Classify Abstract` workflow). |
| `text_preprocessor.py` | Shared NLTK + regex text denoising module wrapped in a scikit-learn `TextDenoiserTransformer` to guarantee train/inference parity. |
| `researchly_best_model.joblib` | Serialised end-to-end scikit-learn `Pipeline`, calibrated decision threshold, label mapping, and held-out validation metrics. |
| `requirements.txt` | Minimal runtime dependencies required to serve the classifier cleanly. |

## Quick Local Setup & Launch

1. **Navigate into the `deployment/` directory** (or run from the repository root):
   ```bash
   cd deployment
   ```

2. **Create and activate a Python 3.10–3.12 virtual environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # macOS / Linux:
   # source .venv/bin/activate
   ```

3. **Install runtime dependencies and NLTK corpora**:
   ```bash
   pip install -r requirements.txt
   python -c "import nltk; [nltk.download(x) for x in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4')]"
   ```

4. **Launch the Streamlit web interface**:
   ```bash
   streamlit run app.py
   ```

5. **Test in your browser**:
   Open `http://localhost:8501`, paste any academic research abstract (or use the preloaded healthcare AI sample), and click **Classify Abstract**. The **Prediction** panel displays `The abstract is classified as: artificial intelligence` or `The abstract is classified as: classical research` alongside the calibrated probability estimate.
