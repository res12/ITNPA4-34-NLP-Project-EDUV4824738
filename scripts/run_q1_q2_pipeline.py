"""Question 1 & Question 2 NLP Classification Pipeline for Researchly (ITNPA4-34).

Executes:
- Q1.1: Dataset loading, audit, deduplication, and NLTK/regex text denoising
- Q1.2: Feature modelling (column audit, Chi-square, Mutual Information, SVD)
- Q1.3: Baseline modelling (DummyClassifier + MultinomialNB) with learning/generalisation metrics
- Q2.1: Multi-model benchmark (5 ML algorithms on identical split & 5-fold CV)
- Q2.2: Ablation-verified performance enhancement (Hybrid Word+Char FeatureUnion,
        cost-sensitive class balancing, GridSearchCV tuning, Soft-Voting Ensemble, OOF thresholding)
- Q2.3: Serialisation of production Pipeline artifact for Streamlit deployment
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

ROOT_DIR = Path(r"C:\Users\resego.ngwenya\.gemini\antigravity\scratch\ITNPA4-34_Project")
DEPLOY_DIR = ROOT_DIR / "deployment"
DATA_DIR = ROOT_DIR / "datasets"
FIG_DIR = ROOT_DIR / "figures"
OUT_DIR = ROOT_DIR / "outputs"

sys.path.insert(0, str(DEPLOY_DIR))
from text_preprocessor import TextDenoiserTransformer, denoise_abstract, ensure_nltk_resources  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"ai": "#0EA5E9", "classical": "#1E3A8A", "accent": "#10B981", "warn": "#F59E0B"}


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, seed: int = 42):
    """Compute 95% stratified bootstrap confidence intervals for Accuracy, F1(ai), and ROC-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s, aucs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, pos_label=1, zero_division=0))
        aucs.append(roc_auc_score(yt, ypr))
    return {
        "acc_ci": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "f1_ai_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "auc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
    }


def evaluate_model(name: str, estimator, X_train, y_train, X_test, y_test, cv, threshold: float = 0.5):
    """Fit estimator, run 5-fold CV, and measure learning, CV, and generalisation metrics."""
    t0 = time.perf_counter()
    cv_res = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "f1_macro": "f1_macro",
            "roc_auc": "roc_auc",
            "avg_prec": "average_precision",
        },
        return_train_score=True,
        n_jobs=1,
    )
    estimator.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t_inf0 = time.perf_counter()
    if hasattr(estimator, "predict_proba"):
        train_prob = estimator.predict_proba(X_train)[:, 1]
        test_prob = estimator.predict_proba(X_test)[:, 1]
    else:
        train_prob = estimator.predict(X_train).astype(float)
        test_prob = estimator.predict(X_test).astype(float)
    inf_ms_per_100 = (time.perf_counter() - t_inf0) * 1000.0 / max(len(X_test) + len(X_train), 1) * 100.0

    train_pred = (train_prob >= threshold).astype(int) if threshold != 0.5 else estimator.predict(X_train)
    test_pred = (test_prob >= threshold).astype(int) if threshold != 0.5 else estimator.predict(X_test)

    cm = confusion_matrix(y_test, test_pred)
    tn, fp, fn, tp = cm.ravel()
    ci = bootstrap_ci(np.asarray(y_test), np.asarray(test_pred), np.asarray(test_prob))

    metrics = {
        "model": name,
        "threshold": float(threshold),
        "train_acc": float(accuracy_score(y_train, train_pred)),
        "train_f1_ai": float(f1_score(y_train, train_pred, pos_label=1, zero_division=0)),
        "train_f1_macro": float(f1_score(y_train, train_pred, average="macro", zero_division=0)),
        "cv_acc_mean": float(np.mean(cv_res["test_accuracy"])),
        "cv_acc_std": float(np.std(cv_res["test_accuracy"])),
        "cv_f1_ai_mean": float(np.mean(cv_res["test_f1"])),
        "cv_f1_ai_std": float(np.std(cv_res["test_f1"])),
        "cv_f1_macro_mean": float(np.mean(cv_res["test_f1_macro"])),
        "cv_roc_auc_mean": float(np.mean(cv_res["test_roc_auc"])),
        "cv_pr_auc_mean": float(np.mean(cv_res["test_avg_prec"])),
        "test_acc": float(accuracy_score(y_test, test_pred)),
        "test_prec_ai": float(precision_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "test_rec_ai": float(recall_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "test_f1_ai": float(f1_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "test_f1_classical": float(f1_score(y_test, test_pred, pos_label=0, zero_division=0)),
        "test_f1_macro": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        "test_f1_weighted": float(f1_score(y_test, test_pred, average="weighted", zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        "test_pr_auc": float(average_precision_score(y_test, test_prob)),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "fit_cv_seconds": float(train_time),
        "inf_ms_per_100": float(inf_ms_per_100),
        "bootstrap_ci": ci,
        "classification_report": classification_report(
            y_test, test_pred, target_names=["classical", "ai"], digits=4, zero_division=0
        ),
    }
    return metrics, estimator, test_prob, test_pred


def make_tuned_cal_svc(C=1.0, class_weight="balanced"):
    return CalibratedClassifierCV(
        LinearSVC(C=C, class_weight=class_weight, random_state=42, max_iter=5000),
        cv=5,
        method="sigmoid",
    )


def main():
    ensure_nltk_resources()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Q1.1: DATA LOADING, AUDIT & TEXT DENOISING
    # =========================================================================
    raw_df = pd.read_json(DATA_DIR / "research_abstracts.json")
    raw_rows = len(raw_df)
    missing_doi = int(raw_df["DOI"].isna().sum())
    exact_dups = int(raw_df.duplicated().sum())
    abstract_dups = int(raw_df.duplicated(subset=["Abstract"]).sum())
    label_conflicts = int(
        (raw_df.groupby("Abstract")["Label"].nunique() > 1).sum()
    )

    # Deduplicate identical abstracts to eliminate cross-split data leakage.
    df = raw_df.drop_duplicates(subset=["Abstract"], keep="first").reset_index(drop=True)
    df["clean_abstract"] = df["Abstract"].apply(denoise_abstract)
    df["raw_words"] = df["Abstract"].str.split().str.len()
    df["clean_words"] = df["clean_abstract"].str.split().str.len()
    df["y"] = (df["Label"] == "ai").astype(int)

    raw_vocab = set(" ".join(df["Abstract"].astype(str)).lower().split())
    clean_vocab = set(" ".join(df["clean_abstract"].astype(str)).split())

    audit_summary = {
        "raw_rows": raw_rows,
        "columns": raw_df.columns.tolist(),
        "missing_doi_raw": missing_doi,
        "exact_row_duplicates": exact_dups,
        "duplicate_abstracts_removed": abstract_dups,
        "label_conflicts_on_duplicates": label_conflicts,
        "clean_unique_rows": len(df),
        "class_counts_raw": raw_df["Label"].value_counts().to_dict(),
        "class_counts_clean": df["Label"].value_counts().to_dict(),
        "raw_avg_words": float(df["raw_words"].mean()),
        "clean_avg_words": float(df["clean_words"].mean()),
        "raw_total_tokens": int(df["raw_words"].sum()),
        "clean_total_tokens": int(df["clean_words"].sum()),
        "token_reduction_pct": float(100.0 * (1.0 - df["clean_words"].sum() / df["raw_words"].sum())),
        "raw_whitespace_vocab": len(raw_vocab),
        "clean_lemmatised_vocab": len(clean_vocab),
        "vocab_reduction_pct": float(100.0 * (1.0 - len(clean_vocab) / len(raw_vocab))),
    }
    print("=== Q1.1 AUDIT ===")
    print(json.dumps(audit_summary, indent=2))

    # Plot Q1.1 EDA & Denoising Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    # (a) Class balance
    counts = df["Label"].value_counts()
    bars = axes[0, 0].bar(
        ["Classical Research (0)", "AI-Related (1)"],
        [counts["classical"], counts["ai"]],
        color=[PALETTE["classical"], PALETTE["ai"]],
        width=0.55,
    )
    for b, cnt in zip(bars, [counts["classical"], counts["ai"]]):
        axes[0, 0].text(
            b.get_x() + b.get_width() / 2,
            cnt + 15,
            f"{cnt} ({cnt / len(df) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axes[0, 0].set_ylim(0, max(counts) * 1.15)
    axes[0, 0].set_title("(a) Deduplicated Class Distribution (n = 1,125)", fontweight="bold")
    axes[0, 0].set_ylabel("Number of Abstracts")

    # (b) Word count before vs after denoising
    axes[0, 1].hist(df["raw_words"], bins=35, alpha=0.55, label="Raw Abstracts", color="#64748B", range=(0, 450))
    axes[0, 1].hist(df["clean_words"], bins=35, alpha=0.70, label="Denoised Abstracts", color=PALETTE["ai"], range=(0, 450))
    axes[0, 1].axvline(df["raw_words"].mean(), color="#334155", linestyle="--", label=f"Raw Mean: {df['raw_words'].mean():.1f}")
    axes[0, 1].axvline(df["clean_words"].mean(), color="#0284C7", linestyle="--", label=f"Clean Mean: {df['clean_words'].mean():.1f}")
    axes[0, 1].set_title("(b) Token Count Distribution Before vs After Denoising", fontweight="bold")
    axes[0, 1].set_xlabel("Words per Abstract (capped at 450 for readability)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend(fontsize=8.5)

    # (c) Top TF terms for AI vs (d) Classical
    for ax, cls_label, title_str, col in [
        (axes[1, 0], "ai", "(c) Top 15 Denoised Terms: AI-Related Class", PALETTE["ai"]),
        (axes[1, 1], "classical", "(d) Top 15 Denoised Terms: Classical Class", PALETTE["classical"]),
    ]:
        cv_top = CountVectorizer(ngram_range=(1, 1), max_features=2000)
        mat = cv_top.fit_transform(df.loc[df["Label"] == cls_label, "clean_abstract"])
        freqs = np.asarray(mat.sum(axis=0)).ravel()
        terms = np.asarray(cv_top.get_feature_names_out())
        top_idx = np.argsort(freqs)[-15:]
        ax.barh(terms[top_idx], freqs[top_idx], color=col)
        ax.set_title(title_str, fontweight="bold")
        ax.set_xlabel("Corpus Term Frequency")

    fig.suptitle("Q1.1 Dataset Audit, Denoising Impact, and Class Lexical Profiles", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "fig_q1_eda_denoising.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Q1.2: STRICT TRAIN/TEST SPLIT & FEATURE MODELLING (NO DATA LEAKAGE)
    # =========================================================================
    X = df["clean_abstract"].to_numpy(dtype=object)
    X_raw = df["Abstract"].to_numpy(dtype=object)
    y = df["y"].to_numpy(dtype=int)
    idx_all = np.arange(len(df))

    idx_train, idx_test, y_train, y_test = train_test_split(
        idx_all, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_test = X[idx_train], X[idx_test]
    X_raw_train, X_raw_test = X_raw[idx_train], X_raw[idx_test]

    cv_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Feature modelling strictly on training set
    tfidf_fm = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.90)
    X_tr_tfidf = tfidf_fm.fit_transform(X_train)
    bow_fm = CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.90)
    X_tr_bow = bow_fm.fit_transform(X_train)

    feat_names = np.asarray(tfidf_fm.get_feature_names_out())
    chi2_scores, chi2_pvals = chi2(X_tr_tfidf, y_train)
    mi_scores = mutual_info_classif(
        (X_tr_bow > 0).astype(int), y_train, discrete_features=True, random_state=42
    )

    ai_mean = np.asarray(X_tr_tfidf[y_train == 1].mean(axis=0)).ravel()
    cl_mean = np.asarray(X_tr_tfidf[y_train == 0].mean(axis=0)).ravel()
    dominant_cls = np.where(ai_mean >= cl_mean, "ai", "classical")

    fm_df = pd.DataFrame({
        "feature": feat_names,
        "chi2": chi2_scores,
        "p_value": chi2_pvals,
        "mutual_info": mi_scores,
        "ai_tfidf_mean": ai_mean,
        "classical_tfidf_mean": cl_mean,
        "dominant_class": dominant_cls,
    }).sort_values("chi2", ascending=False)
    fm_df.head(30).to_csv(OUT_DIR / "q1_2_top_features.csv", index=False)

    svd = TruncatedSVD(n_components=min(200, X_tr_tfidf.shape[1] - 1), random_state=42)
    svd.fit(X_tr_tfidf)
    cum_var = np.cumsum(svd.explained_variance_ratio_)

    # Plot Q1.2 Feature Modelling Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    top20 = fm_df.head(20).iloc[::-1]
    colors_20 = [PALETTE["ai"] if c == "ai" else PALETTE["classical"] for c in top20["dominant_class"]]
    axes[0].barh(top20["feature"], top20["chi2"], color=colors_20)
    axes[0].set_title("Top 20 Informative N-grams by Chi-Square Score (Train Set)", fontweight="bold")
    axes[0].set_xlabel(r"$\chi^2$ Statistic")
    from matplotlib.patches import Patch
    axes[0].legend(
        handles=[Patch(color=PALETTE["ai"], label="AI-Dominant"), Patch(color=PALETTE["classical"], label="Classical-Dominant")],
        loc="lower right",
    )

    axes[1].plot(range(1, len(cum_var) + 1), cum_var * 100, color=PALETTE["classical"], linewidth=2.2)
    axes[1].axhline(50, color=PALETTE["warn"], linestyle="--", linewidth=1.2, label=f"50% threshold (~{int(np.searchsorted(cum_var, 0.5)+1) if np.max(cum_var)>=0.5 else '>200'} comps)")
    axes[1].fill_between(range(1, len(cum_var) + 1), cum_var * 100, alpha=0.15, color=PALETTE["ai"])
    axes[1].set_title(f"TF-IDF SVD Latent Semantic Variance ({X_tr_tfidf.shape[1]} Features)", fontweight="bold")
    axes[1].set_xlabel("Number of TruncatedSVD Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].legend(loc="lower right")

    fig.suptitle("Q1.2 Feature Engineering & Dimensionality Analysis (Training Fold Only)", fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "fig_q1_feature_modelling.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Q1.3: BASELINE MODELS & LEARNING/GENERALISATION EVALUATION
    # =========================================================================
    dummy_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])
    mnb_baseline_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), max_features=5000)),
        ("clf", MultinomialNB(alpha=1.0)),
    ])

    dummy_metrics, _, dummy_prob, _ = evaluate_model(
        "Dummy Majority Classifier", dummy_pipe, X_train, y_train, X_test, y_test, cv_5
    )
    mnb_metrics, fitted_mnb, mnb_prob, mnb_pred = evaluate_model(
        "Multinomial Naive Bayes (Baseline)", mnb_baseline_pipe, X_train, y_train, X_test, y_test, cv_5
    )

    # Learning Curve for MultinomialNB baseline
    train_sizes, lc_train_scores, lc_val_scores = learning_curve(
        mnb_baseline_pipe,
        X_train,
        y_train,
        cv=cv_5,
        scoring="f1",
        train_sizes=np.linspace(0.2, 1.0, 6),
        n_jobs=1,
    )

    # Plot Q1.3 Baseline Performance Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    # Learning Curve
    axes[0].plot(train_sizes, lc_train_scores.mean(axis=1), "o-", color=PALETTE["classical"], label="Train F1 (AI)")
    axes[0].fill_between(
        train_sizes,
        lc_train_scores.mean(axis=1) - lc_train_scores.std(axis=1),
        lc_train_scores.mean(axis=1) + lc_train_scores.std(axis=1),
        alpha=0.15,
        color=PALETTE["classical"],
    )
    axes[0].plot(train_sizes, lc_val_scores.mean(axis=1), "s-", color=PALETTE["ai"], label="5-Fold CV F1 (AI)")
    axes[0].fill_between(
        train_sizes,
        lc_val_scores.mean(axis=1) - lc_val_scores.std(axis=1),
        lc_val_scores.mean(axis=1) + lc_val_scores.std(axis=1),
        alpha=0.15,
        color=PALETTE["ai"],
    )
    axes[0].set_title("(a) Baseline Learning Curve (F1 on AI)", fontweight="bold")
    axes[0].set_xlabel("Training Examples")
    axes[0].set_ylabel("F1 Score (AI Class)")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(loc="lower right")

    # Confusion Matrix
    cm_base = np.array(mnb_metrics["cm"])
    sns.heatmap(
        cm_base,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred: Classical", "Pred: AI"],
        yticklabels=["True: Classical", "True: AI"],
        ax=axes[1],
        annot_kws={"size": 12, "weight": "bold"},
    )
    axes[1].set_title(f"(b) Baseline Test Confusion Matrix (n={len(y_test)})", fontweight="bold")

    # ROC + PR
    fpr, tpr, _ = roc_curve(y_test, mnb_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_test, mnb_prob)
    axes[2].plot(fpr, tpr, color=PALETTE["classical"], lw=2, label=f"ROC (AUC = {mnb_metrics['test_roc_auc']:.3f})")
    axes[2].plot(rec_c, prec_c, color=PALETTE["ai"], lw=2, label=f"PR (AP = {mnb_metrics['test_pr_auc']:.3f})")
    axes[2].plot([0, 1], [0, 1], "--", color="#94A3B8", lw=1)
    axes[2].set_title("(c) Baseline ROC & Precision-Recall Curves", fontweight="bold")
    axes[2].set_xlabel("FPR (ROC) / Recall (PR)")
    axes[2].set_ylabel("TPR (ROC) / Precision (PR)")
    axes[2].legend(loc="lower left")

    fig.suptitle("Q1.3 Baseline Model (MultinomialNB on Unigram TF-IDF) Learning & Generalisation", fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig_q1_baseline_performance.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Q2.1: TRAIN AT LEAST 4 ADDITIONAL ML MODELS (5 DIVERSE MODELS)
    # =========================================================================
    common_vec = lambda: TfidfVectorizer(ngram_range=(1, 1), max_features=5000)
    q21_candidates = [
        ("Logistic Regression", Pipeline([("tfidf", common_vec()), ("clf", LogisticRegression(max_iter=1000, random_state=42))])),
        ("Linear SVM (Calibrated)", Pipeline([("tfidf", common_vec()), ("clf", make_tuned_cal_svc(C=1.0, class_weight=None))])),
        ("Complement Naive Bayes", Pipeline([("tfidf", common_vec()), ("clf", ComplementNB(alpha=1.0))])),
        ("Random Forest", Pipeline([("tfidf", common_vec()), ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1))])),
        ("MLP Neural Network", Pipeline([("tfidf", common_vec()), ("clf", MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, early_stopping=True, random_state=42))])),
    ]

    all_metrics = [dummy_metrics, mnb_metrics]
    q21_probs = {"MultinomialNB (Baseline)": mnb_prob}
    q21_fitted = {}

    for m_name, m_pipe in q21_candidates:
        m_res, m_fit, m_prob, _ = evaluate_model(m_name, m_pipe, X_train, y_train, X_test, y_test, cv_5)
        all_metrics.append(m_res)
        q21_probs[m_name] = m_prob
        q21_fitted[m_name] = m_fit
        print(f"[Q2.1] {m_name}: Test Acc={m_res['test_acc']:.4f}, F1(AI)={m_res['test_f1_ai']:.4f}, Macro-F1={m_res['test_f1_macro']:.4f}, ROC-AUC={m_res['test_roc_auc']:.4f}, PR-AUC={m_res['test_pr_auc']:.4f}")

    # Plot Q2.1 Comparison Figure
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), dpi=300)
    comp_models = [m for m in all_metrics if m["model"] != "Dummy Majority Classifier"]
    names = [m["model"].replace(" (Baseline)", "\n(Baseline)").replace(" (Calibrated)", "\n(Calibrated)") for m in comp_models]
    x_pos = np.arange(len(comp_models))
    w = 0.25
    axes[0].bar(x_pos - w, [m["test_acc"] for m in comp_models], width=w, label="Accuracy", color="#64748B")
    axes[0].bar(x_pos, [m["test_f1_ai"] for m in comp_models], width=w, label="F1 (AI Class)", color=PALETTE["ai"])
    axes[0].bar(x_pos + w, [m["test_pr_auc"] for m in comp_models], width=w, label="PR-AUC", color=PALETTE["classical"])
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("(a) Test Generalisation Metrics vs Baseline", fontweight="bold")
    axes[0].legend(fontsize=8)

    roc_colors = ["#64748B", "#0EA5E9", "#1E3A8A", "#10B981", "#F59E0B", "#8B5CF6"]
    for (m_name, prob), col in zip(q21_probs.items(), roc_colors):
        fpr, tpr, _ = roc_curve(y_test, prob)
        axes[1].plot(fpr, tpr, lw=1.8, color=col, label=f"{m_name.split(' (')[0]} ({roc_auc_score(y_test, prob):.3f})")
        prec_k, rec_k, _ = precision_recall_curve(y_test, prob)
        axes[2].plot(rec_k, prec_k, lw=1.8, color=col, label=f"{m_name.split(' (')[0]} ({average_precision_score(y_test, prob):.3f})")

    axes[1].plot([0, 1], [0, 1], "--", color="#CBD5E1")
    axes[1].set_title("(b) Held-Out ROC Curves", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(fontsize=7.5, loc="lower right")

    axes[2].set_title("(c) Held-Out Precision-Recall Curves", fontweight="bold")
    axes[2].set_xlabel("Recall (AI Class)")
    axes[2].set_ylabel("Precision (AI Class)")
    axes[2].legend(fontsize=7.5, loc="lower left")

    fig.suptitle("Q2.1 Multi-Model Benchmark Comparison (Identical 80/20 Stratified Split)", fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig_q2_model_comparison.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Q2.2: PERFORMANCE ENHANCEMENT (FEATURE UNION + BALANCING + GRIDSEARCH + ENSEMBLE + OOF THRESHOLD)
    # =========================================================================
    def build_hybrid_union(word_ngram=(1, 2), min_df=2, char_ngram=(3, 5)):
        return FeatureUnion([
            ("word_tfidf", TfidfVectorizer(ngram_range=word_ngram, min_df=min_df, max_df=0.90, sublinear_tf=True)),
            ("char_tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram, min_df=3, max_df=0.90, sublinear_tf=True)),
        ])

    # Step A: Hybrid FeatureUnion + default LinearSVC (shows effect of feature representation alone)
    step_a_pipe = Pipeline([
        ("features", build_hybrid_union()),
        ("clf", make_tuned_cal_svc(C=1.0, class_weight=None)),
    ])
    step_a_res, _, _, _ = evaluate_model(
        "Enhancement A: Hybrid Word+Char FeatureUnion (LinearSVC)", step_a_pipe, X_train, y_train, X_test, y_test, cv_5
    )

    # Step B: + Cost-Sensitive Class Balancing
    step_b_pipe = Pipeline([
        ("features", build_hybrid_union()),
        ("clf", make_tuned_cal_svc(C=1.0, class_weight="balanced")),
    ])
    step_b_res, _, _, _ = evaluate_model(
        "Enhancement B: + Cost-Sensitive Balancing (LinearSVC)", step_b_pipe, X_train, y_train, X_test, y_test, cv_5
    )

    # Step C: GridSearchCV Tuning on Balanced Hybrid Logistic Regression & LinearSVC + Soft-Voting Ensemble
    gs_lr_pipe = Pipeline([
        ("features", build_hybrid_union()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
    ])
    gs_lr = GridSearchCV(
        gs_lr_pipe,
        param_grid={
            "features__word_tfidf__ngram_range": [(1, 2), (1, 3)],
            "features__word_tfidf__min_df": [1, 2],
            "clf__C": [1.0, 2.0, 5.0, 10.0],
        },
        cv=cv_5,
        scoring="f1",
        n_jobs=1,
    )
    gs_lr.fit(X_train, y_train)

    gs_svc_pipe = Pipeline([
        ("features", build_hybrid_union()),
        ("clf", LinearSVC(class_weight="balanced", random_state=42, max_iter=5000)),
    ])
    gs_svc = GridSearchCV(
        gs_svc_pipe,
        param_grid={
            "features__word_tfidf__ngram_range": [(1, 2)],
            "features__word_tfidf__min_df": [1, 2],
            "clf__C": [0.25, 0.5, 1.0, 2.0],
        },
        cv=cv_5,
        scoring="f1",
        n_jobs=1,
    )
    gs_svc.fit(X_train, y_train)

    best_lr_params = gs_lr.best_params_
    best_svc_params = gs_svc.best_params_
    print(f"Best LR CV F1={gs_lr.best_score_:.4f} params={best_lr_params}")
    print(f"Best SVC CV F1={gs_svc.best_score_:.4f} params={best_svc_params}")

    ensemble_pipe = Pipeline([
        ("features", build_hybrid_union(
            word_ngram=best_lr_params["features__word_tfidf__ngram_range"],
            min_df=best_svc_params["features__word_tfidf__min_df"],
        )),
        ("clf", VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=best_lr_params["clf__C"], class_weight="balanced", max_iter=2000, random_state=42)),
                ("svc", make_tuned_cal_svc(C=best_svc_params["clf__C"], class_weight="balanced")),
                ("cnb", ComplementNB(alpha=0.5)),
            ],
            voting="soft",
            weights=[2, 2, 1],
        )),
    ])

    # Out-of-fold probability thresholding strictly inside X_train (no test leakage!)
    oof_probs = cross_val_predict(ensemble_pipe, X_train, y_train, cv=cv_5, method="predict_proba", n_jobs=1)[:, 1]
    candidate_thresholds = np.linspace(0.25, 0.75, 51)
    oof_f1s = [f1_score(y_train, (oof_probs >= t).astype(int), pos_label=1) for t in candidate_thresholds]
    best_oof_idx = int(np.argmax(oof_f1s))
    opt_threshold = float(candidate_thresholds[best_oof_idx])
    print(f"Optimal OOF decision threshold: {opt_threshold:.2f} (OOF F1(AI)={oof_f1s[best_oof_idx]:.4f} vs 0.50 F1(AI)={f1_score(y_train, (oof_probs >= 0.5).astype(int)):.4f})")

    ens_default_res, _, _, _ = evaluate_model(
        "Enhancement C1: Tuned Hybrid Soft-Voting Ensemble (t=0.50)",
        ensemble_pipe, X_train, y_train, X_test, y_test, cv_5, threshold=0.50
    )
    ens_opt_res, fitted_best_clf, best_test_prob, best_test_pred = evaluate_model(
        f"Enhancement C2: Tuned Hybrid Ensemble + OOF Threshold (t={opt_threshold:.2f})",
        ensemble_pipe, X_train, y_train, X_test, y_test, cv_5, threshold=opt_threshold
    )

    all_metrics.extend([step_a_res, step_b_res, ens_default_res, ens_opt_res])

    # Slice-based error analysis on test set by abstract length terciles
    test_lengths = np.array([len(txt.split()) for txt in X_raw_test])
    q33, q66 = np.percentile(test_lengths, [33.3, 66.7])
    slice_rows = []
    for s_name, mask in [
        (f"Short (<={int(q33)} words)", test_lengths <= q33),
        (f"Medium ({int(q33)+1}-{int(q66)} words)", (test_lengths > q33) & (test_lengths <= q66)),
        (f"Long (>{int(q66)} words)", test_lengths > q66),
    ]:
        slice_rows.append({
            "slice": s_name,
            "n_samples": int(mask.sum()),
            "ai_count": int(y_test[mask].sum()),
            "accuracy": float(accuracy_score(y_test[mask], best_test_pred[mask])),
            "f1_ai": float(f1_score(y_test[mask], best_test_pred[mask], pos_label=1, zero_division=0)),
        })

    # Plot Q2.2 Enhancement Ablation Figure
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=300)
    best_std = max(
        [m for m in all_metrics if m["model"] in {x[0] for x in q21_candidates}],
        key=lambda d: d["test_f1_ai"],
    )
    ladder = [mnb_metrics, best_std, step_a_res, step_b_res, ens_default_res, ens_opt_res]
    ladder_labels = [
        "1.3 Baseline\n(MNB)",
        f"2.1 Best Std\n({best_std['model'].split(' ')[0]})",
        "+Hybrid\nWord+Char",
        "+Balanced\nWeights",
        "+Tuned\nEnsemble",
        f"+OOF Thr\n(t={opt_threshold:.2f})",
    ]
    lx = np.arange(len(ladder))
    axes[0].plot(lx, [m["test_f1_ai"] for m in ladder], "o-", lw=2.3, color=PALETTE["ai"], label="F1 (AI Class)")
    axes[0].plot(lx, [m["test_rec_ai"] for m in ladder], "s--", lw=1.8, color=PALETTE["accent"], label="Recall (AI Class)")
    axes[0].plot(lx, [m["test_acc"] for m in ladder], "^-", lw=1.8, color=PALETTE["classical"], label="Accuracy")
    for xi, yi in zip(lx, [m["test_f1_ai"] for m in ladder]):
        axes[0].text(xi, yi + 0.025, f"{yi:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].set_xticks(lx)
    axes[0].set_xticklabels(ladder_labels, fontsize=8)
    axes[0].set_ylim(0.35, 1.08)
    axes[0].set_title("(a) Ablation Progression on Held-Out Test Set", fontweight="bold")
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].plot(candidate_thresholds, oof_f1s, color=PALETTE["classical"], lw=2.2)
    axes[1].axvline(opt_threshold, color=PALETTE["accent"], linestyle="--", lw=1.8, label=f"Optimal OOF t = {opt_threshold:.2f} (F1={oof_f1s[best_oof_idx]:.3f})")
    axes[1].axvline(0.50, color="#94A3B8", linestyle=":", lw=1.5, label="Default t = 0.50")
    axes[1].set_title("(b) Out-of-Fold CV Threshold Optimisation", fontweight="bold")
    axes[1].set_xlabel("Classification Threshold (P(AI))")
    axes[1].set_ylabel("5-Fold OOF F1 Score (AI)")
    axes[1].legend(fontsize=8, loc="lower center")

    cm_best = np.array(ens_opt_res["cm"])
    sns.heatmap(
        cm_best,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        xticklabels=["Pred: Classical", "Pred: AI"],
        yticklabels=["True: Classical", "True: AI"],
        ax=axes[2],
        annot_kws={"size": 12, "weight": "bold"},
    )
    axes[2].set_title(f"(c) Final Enhanced Confusion Matrix (n={len(y_test)})", fontweight="bold")

    fig.suptitle("Q2.2 Predictive Performance Enhancement & Ablation Evidence", fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig_q2_enhancement_ablation.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Q2.3: PACKAGE FULL END-TO-END DEPLOYMENT PIPELINE (INCLUDING TEXT DENOISING)
    # =========================================================================
    production_pipeline = Pipeline([
        ("denoiser", TextDenoiserTransformer()),
        ("model", ensemble_pipe),
    ])
    # Fit on full 1,125 deduplicated corpus so the deployed web app benefits from all training examples,
    # after locking all hyperparameters & OOF threshold on the 80/20 experiment.
    production_pipeline.fit(X_raw, y)
    artifact_bundle = {
        "pipeline": production_pipeline,
        "threshold": opt_threshold,
        "labels": {0: "classical research", 1: "artificial intelligence"},
        "best_lr_params": best_lr_params,
        "best_svc_params": best_svc_params,
        "heldout_test_metrics": ens_opt_res,
    }
    joblib.dump(artifact_bundle, DEPLOY_DIR / "researchly_best_model.joblib")

    # Sanity check on screenshot abstract + classical abstract
    sample_ai = (
        "Artificial intelligence (AI) has become a transformative technology in healthcare by enabling more "
        "accurate diagnosis, predictive analytics, and personalised treatment planning. Machine learning algorithms "
        "can analyse large volumes of medical data to identify patterns that assist clinicians in detecting "
        "diseases at earlier stages. In addition, AI-powered decision support systems improve clinical workflows "
        "by reducing manual effort and enhancing diagnostic consistency. Despite these benefits, challenges such "
        "as data privacy, model interpretability, and ethical considerations remain significant barriers to "
        "widespread adoption. Continued research is required to develop robust, transparent, and trustworthy AI "
        "systems that can be safely integrated into modern healthcare environments."
    )
    sample_cl = (
        "This paper develops a structural dynamic stochastic general equilibrium (DSGE) model with sticky wages "
        "and monetary policy rules to estimate the Euro Area output gap and Phillips curve slope using Bayesian "
        "Markov chain Monte Carlo methods on quarterly macroeconomic series."
    )
    loaded = joblib.load(DEPLOY_DIR / "researchly_best_model.joblib")
    p_ai = float(loaded["pipeline"].predict_proba([sample_ai])[0, 1])
    p_cl = float(loaded["pipeline"].predict_proba([sample_cl])[0, 1])
    print(f"Deployment smoke check AI abstract: P(AI)={p_ai:.4f} -> {'artificial intelligence' if p_ai >= loaded['threshold'] else 'classical research'}")
    print(f"Deployment smoke check Classical abstract: P(AI)={p_cl:.4f} -> {'artificial intelligence' if p_cl >= loaded['threshold'] else 'classical research'}")

    # Save structured metrics JSON & CSV
    summary_rows = []
    for m in all_metrics:
        summary_rows.append({
            "Model": m["model"],
            "Train_Acc": round(m["train_acc"], 4),
            "Train_F1_AI": round(m["train_f1_ai"], 4),
            "CV_Acc": round(m["cv_acc_mean"], 4),
            "CV_F1_AI": round(m["cv_f1_ai_mean"], 4),
            "Test_Acc": round(m["test_acc"], 4),
            "Test_Prec_AI": round(m["test_prec_ai"], 4),
            "Test_Rec_AI": round(m["test_rec_ai"], 4),
            "Test_F1_AI": round(m["test_f1_ai"], 4),
            "Test_Macro_F1": round(m["test_f1_macro"], 4),
            "Test_ROC_AUC": round(m["test_roc_auc"], 4),
            "Test_PR_AUC": round(m["test_pr_auc"], 4),
        })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "q1_q2_model_comparison.csv", index=False)

    full_results = {
        "audit": audit_summary,
        "split": {"train_n": len(y_train), "train_ai": int(y_train.sum()), "test_n": len(y_test), "test_ai": int(y_test.sum())},
        "feature_modelling": {
            "tfidf_features": int(X_tr_tfidf.shape[1]),
            "bow_features": int(X_tr_bow.shape[1]),
            "svd_200_cum_var_pct": float(cum_var[-1] * 100),
            "top_chi2": fm_df.head(15).to_dict(orient="records"),
            "top_mi": fm_df.sort_values("mutual_info", ascending=False).head(10).to_dict(orient="records"),
        },
        "models": all_metrics,
        "enhancement_params": {
            "best_lr_params": best_lr_params,
            "best_svc_params": best_svc_params,
            "opt_threshold": opt_threshold,
            "oof_f1_at_050": float(f1_score(y_train, (oof_probs >= 0.5).astype(int))),
            "oof_f1_at_opt": float(oof_f1s[best_oof_idx]),
        },
        "slice_analysis": slice_rows,
        "deployment_check": {
            "ai_sample_prob": p_ai,
            "classical_sample_prob": p_cl,
            "threshold": opt_threshold,
        },
    }
    with open(OUT_DIR / "q1_q2_results.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print("Q1 & Q2 pipeline finished cleanly.")


if __name__ == "__main__":
    main()
