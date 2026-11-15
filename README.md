# ITNPA4-34 Project 1 — Natural Language Processing with Python

**Student:** Resego Ngwenya (`EDUV4824738`) | **Campus:** Online (Pretoria) | **Lecturer:** Mr. T. Mavetera | **NQF Level:** 8

This repository contains the end-to-end NLP implementation, datasets, reproducible Jupyter notebooks, trained model artefact, live Streamlit web application, interactive `pyLDAvis` visualisations, and full formal project documentation for **Researchly's Research Abstract Classifier** (Questions 1–2) and **Academic Methodology Topic Discovery & Summarisation** (Questions 3–4).

---

## Repository Structure

```text
ITNPA4-34-NLP-Project-EDUV4824738/
├── datasets/
│   ├── research_abstracts.json        # 1,138 labelled AI vs classical abstracts
│   ├── research_corpus.zip            # Archive of 5 methodology research articles
│   └── research_corpus/               # Extracted research_1.txt .. research_5.txt
├── deployment/
│   ├── app.py                         # Streamlit Researchly web application
│   ├── text_preprocessor.py           # Shared stateless & sklearn-compatible denoiser
│   ├── researchly_best_model.joblib   # Serialised Hybrid Soft-Voting Ensemble (t = 0.45)
│   ├── requirements.txt               # Deployment runtime dependencies
│   └── README.md                      # Standalone deployment guide
├── docs/
│   └── ITNPA4-34 - Project - Online - EDUV4824738.pdf
├── figures/                           # 12 publication-grade 300 DPI PNG visualisations
├── notebooks/
│   ├── ITNPA4_34_Q1_Q2_Researchly_Classifier.ipynb
│   └── ITNPA4_34_Q3_Q4_Topic_Analysis_and_LDA.ipynb
├── outputs/
│   ├── q1_q2_results.json             # Full Q1–Q2 audit, CV, test & ablation metrics
│   ├── q1_q2_model_comparison.csv     # Baseline vs Q2.1 models vs Q2.2 ablations
│   ├── q1_2_top_features.csv          # Top Chi-square & Mutual Info n-grams
│   ├── q3_corpus_dataframe.csv        # Structured corpus audit & bigram collocations
│   ├── q3_q4_results.json             # TextRank, LSA+MMR, LDA & pyLDAvis exports
│   ├── q4_doc_topic_distribution.csv  # Article-level Dirichlet topic matrix (Theta)
│   └── pyldavis_intertopic_map.html   # Interactive pyLDAvis MMDS distance map
├── scripts/
│   ├── run_q1_q2_pipeline.py          # Full Q1–Q2 experimental pipeline
│   ├── run_q3_q4_pipeline.py          # Full Q3–Q4 summarisation & LDA pipeline
│   ├── build_notebooks.py             # Generates and executes both .ipynb notebooks
│   └── build_itnpa4_report.py         # Builds signed Eduvos Word & PDF report
├── requirements.txt
└── README.md
```

---

## Key Verified Experimental Results

### Questions 1 & 2 — Researchly Semantic Classification (`research_abstracts.json`)

- **Data Integrity & Denoising (`1,138` raw -> `1,125` unique abstracts)**: Removed 13 duplicate abstracts pre-split to prevent train-test leakage (`890` `classical` [79.11%], `235` `ai` [20.89%]). NFKC/LaTeX/URL cleaning + WordNet lemmatisation compressed corpus tokens by **38.40%** (`193,044 -> 118,912`) and vocabulary by **51.88%** (`17,734 -> 8,533`).
- **Baseline (`MultinomialNB` on unigram TF-IDF)**: `Test Acc = 0.8089`, `ROC-AUC = 0.8578`, yet severe minority recall collapse (`Test Rec = 0.0851`, `Test F1(AI) = 0.1569`, `5-Fold CV F1 = 0.0803`).
- **Q2.1 Model Benchmark**: `Linear SVM (Calibrated)` achieved strongest held-out F1 (`0.7356`, `Acc = 0.8978`); `Random Forest` achieved highest probability separation (`ROC-AUC = 0.9234`, `PR-AUC = 0.8564`, `F1 = 0.6571`); followed by `Complement Naive Bayes` (`0.5507`), `MLP Neural Network` (`0.5152`), and `Logistic Regression` (`0.3729`).
- **Q2.2 Final Enhanced Pipeline (`Enhancement C2`)**: Hybrid `FeatureUnion` (`word (1,2)` + `char_wb (3,5)` sublinear TF-IDF) + `class_weight='balanced'` + Stratified 5-Fold `GridSearchCV` + Calibrated Soft-Voting Ensemble (`LogisticRegression` + `LinearSVC` + `ComplementNB`) + OOF Decision Threshold (`t = 0.45`):
  - **5-Fold CV Accuracy:** `0.9311` | **5-Fold CV F1 (AI):** `0.8070`
  - **Held-Out Test Accuracy:** `0.9022` | **Test Precision:** `0.8378` | **Test Recall:** `0.6596` | **Test F1 (AI):** `0.7381` (**4.7x** baseline) | **Macro-F1:** `0.8390` | **ROC-AUC:** `0.8885`

### Questions 3 & 4 — Corpus Summarisation, LDA & `pyLDAvis` (`research_corpus.zip`)

- **Q3.2 Dual-Stage Denoising**: Repaired PDF line-wraps, eliminated duplicate body copy in `research_1.txt` (`80` clean unique sentences), stripped control bytes (`\x01`), running headers, footnotes, and APA citations, and discovered domain bigrams (`cronbach_alpha`, `mixed_method`, `science_education`, `data_triangulation`).
- **Q3.3 Single-Document Summarisation (`research_1.txt`)**: Graph-based **TextRank** (NetworkX PageRank over TF-IDF cosine similarity) extracted 6 key chronological hub sentences (`96.27%` compression).
- **Q3.4 Multi-Document Aggregate Summarisation (`research_2.txt`–`research_5.txt`)**: **LSA (`TruncatedSVD`, 4 components) + Maximal Marginal Relevance (`λ = 0.68`, `max_per_doc = 2`)** synthesised 8 balanced, non-redundant summary sentences across all four methodologies.
- **Q4.1–Q4.2 LDA & Interactive `pyLDAvis`**: Passage-segmented LDA (`111` ~150-word passages, `1,161` filtered terms, symmetric priors) selected optimal **`n = 3` topics** by both highest **$C_v$ coherence (`0.5456`)** and lowest **perplexity (`813.16`)**:
  - **Topic 1 (`12.6%`)**: *Methodological & Data Triangulation* (peaks in `Article 5: Triangulation`, $\theta_1 = 0.570$).
  - **Topic 2 (`33.7%`)**: *Qualitative, Reflexive & Mixed-Methods Inquiry* (peaks in `Article 2: Reflexivity`, $\theta_2 = 0.953$, and `Article 1: Mixed Methods`, $\theta_2 = 0.867$).
  - **Topic 3 (`53.7%`)**: *Instructional Development & Psychometric Scale Reliability* (peaks in `Article 4: Cronbach's Alpha`, $\theta_3 = 0.995$, and `Article 3: Developmental Research`, $\theta_3 = 0.950$).

---

## Reproducing Locally

```powershell
# 1. Create and activate Python 3.12 virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run Q1–Q2 and Q3–Q4 analytical pipelines
python scripts\run_q1_q2_pipeline.py
python scripts\run_q3_q4_pipeline.py

# 3. Launch the Researchly Streamlit Web App
streamlit run deployment\app.py
```
