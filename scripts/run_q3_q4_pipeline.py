"""Question 3 & Question 4 Topic Analysis, Summarisation, LDA, and pyLDAvis Pipeline (ITNPA4-34).

Executes:
- Q3.1: Load research_corpus.zip directly into a structured pandas DataFrame
- Q3.2: Dual-stage state-of-the-art denoising (paragraph unwrap, citation/number stripping,
        lemmatisation, custom stopword removal, Gensim Phrase bigram collocations)
- Q3.3: TextRank (TF-IDF + cosine graph + NetworkX PageRank) extractive summarisation of Article 1
- Q3.4: Aggregate multi-document LSA (TruncatedSVD + MMR) summarisation of Articles 2-5
        with two visualisations (LSA concept/sentence saliency + WordCloud/keyphrase chart)
- Q4.1: Passage-segmented Gensim LDA with C_v coherence sweep, topic-word beta & doc-topic theta plots
- Q4.2: Interactive pyLDAvis Intertopic Distance Map (HTML + static 300 DPI salient-term figure)
"""

import io
import json
import os
import re
import sys
import warnings
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
import seaborn as sns
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel, Phrases
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

ROOT_DIR = Path(r"C:\Users\resego.ngwenya\.gemini\antigravity\scratch\ITNPA4-34_Project")
DATA_DIR = ROOT_DIR / "datasets"
DEPLOY_DIR = ROOT_DIR / "deployment"
FIG_DIR = ROOT_DIR / "figures"
OUT_DIR = ROOT_DIR / "outputs"

sys.path.insert(0, str(DEPLOY_DIR))
from text_preprocessor import ensure_nltk_resources, normalize_unicode_and_contractions  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")

APA_PAREN_CITE_RE = re.compile(
    r"\([A-Z][A-Za-z&.,\s\-']{1,90}?\b(?:19|20)\d{2}[a-z]?(?:,\s*pp?\.\s*\d+(?:[-–]\d+)?)?\)"
)
PAREN_PAGE_RE = re.compile(r"\(\s*pp?\.\s*\d+(?:[-–]\d+)?\s*\)")
SECTION_HEADER_NUM_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s+[A-Z][A-Z\s\-:]+\s*$", re.MULTILINE)
URL_EMAIL_RE = re.compile(r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE)

DOMAIN_ACADEMIC_STOPWORDS = {
    "article", "research", "researcher", "researchers", "study", "studies", "paper",
    "papers", "author", "authors", "section", "table", "figure", "example", "examples",
    "also", "may", "might", "must", "need", "used", "using", "use", "uses", "one", "two",
    "three", "four", "well", "many", "much", "often", "within", "across", "however",
    "therefore", "thus", "although", "whether", "toward", "towards", "among", "upon",
    "first", "second", "third", "new", "make", "made", "way", "ways", "work", "et", "al",
    "pp", "vol", "no", "page", "journal", "university", "abstract", "introduction", "cited",
}

ARTICLE_META = {
    "research_1.txt": {
        "short_label": "Article 1: Mixed Methods",
        "title": "Mixed Methods Research: A Research Paradigm Whose Time Has Come",
        "authors": "R. Burke Johnson & Anthony J. Onwuegbuzie",
    },
    "research_2.txt": {
        "short_label": "Article 2: Reflexivity",
        "title": "Now I See It, Now I Don't: Researcher's Position and Reflexivity in Qualitative Research",
        "authors": "Roni Berger",
    },
    "research_3.txt": {
        "short_label": "Article 3: Developmental Research",
        "title": "Developmental Research: Studies of Instructional Design and Development",
        "authors": "Rita C. Richey, James D. Klein & Wayne A. Nelson",
    },
    "research_4.txt": {
        "short_label": "Article 4: Cronbach's Alpha",
        "title": "The Use of Cronbach's Alpha When Developing and Reporting Research Instruments in Science Education",
        "authors": "Keith S. Taber",
    },
    "research_5.txt": {
        "short_label": "Article 5: Triangulation",
        "title": "Denzin's Paradigm Shift: Revisiting Triangulation in Qualitative Research",
        "authors": "Patricia Fusch, Gene E. Fusch & Lawrence R. Ness",
    },
}


RUNNING_HDR_RE = re.compile(
    r"Fusch,\s*Fusch,\s*&\s*Ness,\s*2018(?:\s+Journal\s+of\s+Social\s+Change\s*\d+)?",
    re.IGNORECASE,
)


def unwrap_and_clean_prose(raw_text: str) -> str:
    """Clean PDF line-wraps, running headers, footnote digits, APA citations, and URLs."""
    txt = raw_text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    txt = (
        txt.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
    )
    txt = re.sub(r"[\x00-\x08\x0b-\x1f\x7f\u2022]+", " ", txt)
    txt = SECTION_HEADER_NUM_RE.sub(" ", txt)
    txt = RUNNING_HDR_RE.sub(" ", txt)
    # Collapse hard line-wraps inside paragraphs while preserving double-newline breaks
    txt = re.sub(r"\n{2,}", "\n\n", txt)
    txt = re.sub(r"(?<!\n)\n(?!\n)", " ", txt)
    txt = re.sub(r"\n+", " ", txt)
    txt = RUNNING_HDR_RE.sub(" ", txt)
    txt = URL_EMAIL_RE.sub(" ", txt)
    txt = APA_PAREN_CITE_RE.sub("", txt)
    txt = PAREN_PAGE_RE.sub("", txt)
    # Strip footnote superscripts stuck after sentence-ending periods (e.g., research.5 -> research.)
    txt = re.sub(r"(?<=[a-zA-Z]\.)\d+\b", "", txt)
    txt = re.sub(r"\s+([,.;:!?])", r"\1", txt)
    return re.sub(r"\s+", " ", txt).strip()


def extract_clean_sentences(prose_text: str, min_words: int = 8) -> list[str]:
    """Tokenise denoised prose into unique, substantive standalone sentences."""
    sents = sent_tokenize(prose_text)
    cleaned, seen = [], set()
    for s in sents:
        s_clean = re.sub(r"\s+", " ", s).strip()
        words = s_clean.split()
        if len(words) < min_words or s_clean.isupper():
            continue
        key = s_clean.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s_clean)
    return cleaned


def tokenize_for_topics(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> list[str]:
    """Normalise, strip non-alphabetic noise, stopword-filter, and lemmatise into unigram tokens."""
    norm = normalize_unicode_and_contractions(text).lower()
    norm = re.sub(r"[-/]+", " ", norm)
    norm = re.sub(r"[^a-z\s]", " ", norm)
    toks = word_tokenize(norm)
    out = []
    for t in toks:
        if t in stop_words:
            continue
        lemma = lemmatizer.lemmatize(t)
        if lemma in stop_words or len(lemma) < 3:
            continue
        out.append(lemma)
    return out


def textrank_summarise(sentences: list[str], top_n: int = 6):
    """Run TextRank extractive summarisation using TF-IDF cosine graph + NetworkX PageRank."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform(sentences)
    sim = cosine_similarity(mat)
    np.fill_diagonal(sim, 0.0)
    # Prune very weak edges below threshold for cleaner centrality topology
    sim[sim < 0.05] = 0.0
    graph = nx.from_numpy_array(sim)
    scores = nx.pagerank(graph, max_iter=500, tol=1e-6)
    score_arr = np.array([scores[i] for i in range(len(sentences))])
    top_idx = sorted(np.argsort(score_arr)[-top_n:].tolist())
    return top_idx, score_arr, graph


def lsa_mmr_summarise(
    sentences: list[str],
    doc_labels: list[str],
    n_components: int = 4,
    top_n: int = 8,
    lam: float = 0.68,
    max_per_doc: int = 2,
):
    """Run LSA (TruncatedSVD) sentence scoring with MMR and per-document balance across multi-document corpus."""
    from collections import Counter

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
    X_tfidf = vec.fit_transform(sentences)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(X_tfidf)  # shape: (n_sentences, n_components)
    sv = svd.singular_values_
    saliency = np.sqrt(np.sum((U ** 2) * (sv ** 2), axis=1))
    saliency_norm = saliency / (saliency.max() + 1e-12)

    sim_matrix = cosine_similarity(X_tfidf)
    selected_idx = []
    candidates = list(range(len(sentences)))
    doc_counts = Counter()

    for _ in range(min(top_n, len(sentences))):
        eligible = [c for c in candidates if doc_counts[doc_labels[c]] < max_per_doc]
        if not eligible:
            eligible = candidates
        if not selected_idx:
            best = max(eligible, key=lambda c: saliency_norm[c])
        else:
            best_score = -1e9
            best = eligible[0]
            for c in eligible:
                redundancy = max(sim_matrix[c, s] for s in selected_idx)
                mmr_score = lam * saliency_norm[c] - (1.0 - lam) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = c
        selected_idx.append(best)
        doc_counts[doc_labels[best]] += 1
        candidates.remove(best)

    return selected_idx, saliency_norm, U, svd


def build_passages(sentences: list[str], doc_names: list[str], target_words: int = 160):
    """Segment each article's clean sentences into ~160-word coherent passages for LDA topic modelling."""
    passages = []
    for doc_name in sorted(set(doc_names)):
        doc_sents = [s for s, d in zip(sentences, doc_names) if d == doc_name]
        cur_chunk, cur_words = [], 0
        for s in doc_sents:
            w = len(s.split())
            cur_chunk.append(s)
            cur_words += w
            if cur_words >= target_words:
                passages.append((doc_name, " ".join(cur_chunk)))
                cur_chunk, cur_words = [], 0
        if cur_chunk and cur_words >= 45:
            passages.append((doc_name, " ".join(cur_chunk)))
    return passages


def main():
    ensure_nltk_resources()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stop_words = set(stopwords.words("english")) | DOMAIN_ACADEMIC_STOPWORDS
    lemmatizer = WordNetLemmatizer()

    # =========================================================================
    # Q3.1: GENERATE DATAFRAME OF ALL RESEARCH ARTICLES FROM research_corpus.zip
    # =========================================================================
    records = []
    zip_path = DATA_DIR / "research_corpus.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        txt_members = sorted([m for m in zf.namelist() if m.endswith(".txt")])
        for doc_num, member in enumerate(txt_members, start=1):
            fname = os.path.basename(member)
            raw_text = zf.read(member).decode("utf-8", errors="replace")
            meta = ARTICLE_META[fname]
            records.append({
                "doc_id": doc_num,
                "file_name": fname,
                "short_label": meta["short_label"],
                "title": meta["title"],
                "authors": meta["authors"],
                "content": raw_text,
                "raw_char_count": len(raw_text),
                "raw_word_count": len(raw_text.split()),
                "raw_line_count": len(raw_text.splitlines()),
            })
    df_corpus = pd.DataFrame(records)

    # =========================================================================
    # Q3.2: STATE-OF-THE-ART DUAL-STAGE DENOISING
    # =========================================================================
    df_corpus["prose_cleaned"] = df_corpus["content"].apply(unwrap_and_clean_prose)
    df_corpus["sentences"] = df_corpus["prose_cleaned"].apply(extract_clean_sentences)
    df_corpus["clean_sentence_count"] = df_corpus["sentences"].apply(len)
    df_corpus["prose_dedup"] = df_corpus["sentences"].apply(" ".join)
    unigram_docs = [tokenize_for_topics(txt, stop_words, lemmatizer) for txt in df_corpus["prose_dedup"]]
    # Fit Gensim bigram Phraser across corpus so multi-word methodology terms stay intact
    bigram_model = Phrases(unigram_docs, min_count=4, threshold=8.0, delimiter="_")
    bigram_phraser = Phraser(bigram_model)
    phrased_docs = [bigram_phraser[doc] for doc in unigram_docs]

    df_corpus["clean_tokens"] = phrased_docs
    df_corpus["clean_token_count"] = [len(t) for t in phrased_docs]
    df_corpus["unique_lemma_vocab"] = [len(set(t)) for t in phrased_docs]
    df_corpus["lexical_diversity_ttr"] = [
        round(len(set(t)) / max(len(t), 1), 4) for t in phrased_docs
    ]
    df_corpus["top_bigrams"] = [
        ", ".join(pd.Series([tok for tok in t if "_" in tok]).value_counts().head(5).index.tolist())
        for t in phrased_docs
    ]

    df_save = df_corpus.drop(columns=["content", "prose_cleaned", "prose_dedup", "sentences", "clean_tokens"])
    df_save.to_csv(OUT_DIR / "q3_corpus_dataframe.csv", index=False)
    print("=== Q3.1 & Q3.2 CORPUS DATAFRAME ===")
    print(df_save.to_string(index=False))

    # Plot Q3.2 Denoising Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    x = np.arange(len(df_corpus))
    w = 0.36
    axes[0].bar(x - w / 2, df_corpus["raw_word_count"], width=w, color="#94A3B8", label="Raw Word Count")
    axes[0].bar(x + w / 2, df_corpus["clean_token_count"], width=w, color="#0EA5E9", label="Denoised Phrased Tokens")
    for xi, (rw, cw) in enumerate(zip(df_corpus["raw_word_count"], df_corpus["clean_token_count"])):
        pct = 100.0 * (1.0 - cw / rw)
        axes[0].text(xi, rw + 120, f"-{pct:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#1E3A8A")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Art. {i}\n({fn.replace('.txt','')})" for i, fn in zip(df_corpus["doc_id"], df_corpus["file_name"])], fontsize=8.5)
    axes[0].set_ylim(0, df_corpus["raw_word_count"].max() * 1.35)
    axes[0].set_title("(a) Token Compression via State-of-the-Art Denoising", fontweight="bold")
    axes[0].set_ylabel("Word / Token Count")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=True)

    all_bigrams = pd.Series([tok for doc in phrased_docs for tok in doc if "_" in tok]).value_counts().head(15).iloc[::-1]
    axes[1].barh(all_bigrams.index, all_bigrams.values, color="#1E3A8A")
    axes[1].set_title("(b) Top 15 Discovered Domain Bigram Collocations", fontweight="bold")
    axes[1].set_xlabel("Corpus Frequency")

    fig.suptitle("Q3.2 Corpus Denoising Impact and Phrasal Collocation Extraction", fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig_q3_corpus_denoising.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Q3.3: SUMMARISE FIRST ARTICLE (TEXTRANK + NETWORK GRAPH)
    # =========================================================================
    art1_sents = df_corpus.loc[0, "sentences"]
    top_idx_1, tr_scores, tr_graph = textrank_summarise(art1_sents, top_n=6)
    art1_summary_sents = [art1_sents[i] for i in top_idx_1]
    art1_summary_words = sum(len(s.split()) for s in art1_summary_sents)
    art1_comp_pct = 100.0 * (1.0 - art1_summary_words / df_corpus.loc[0, "raw_word_count"])

    print(f"\n=== Q3.3 ARTICLE 1 TEXTRANK SUMMARY ({len(art1_sents)} sents -> 6 sents, {art1_comp_pct:.1f}% compression) ===")
    for idx_s in top_idx_1:
        print(f"  [S{idx_s+1}, score={tr_scores[idx_s]:.4f}] {art1_sents[idx_s]}")

    # Plot Q3.3 TextRank Network & Trajectory
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=300)
    # Subgraph of top 25 highest PageRank sentences for clear readability
    top25_nodes = sorted(np.argsort(tr_scores)[-25:].tolist())
    sub_g = tr_graph.subgraph(top25_nodes)
    pos = nx.spring_layout(sub_g, seed=11, k=0.95)
    axes[0].margins(0.14)
    node_colors = ["#10B981" if n in top_idx_1 else "#0EA5E9" for n in sub_g.nodes()]
    node_sizes = [1800 * (tr_scores[n] / tr_scores.max()) + 180 for n in sub_g.nodes()]
    edge_weights = [d["weight"] * 4.5 for _, _, d in sub_g.edges(data=True)]
    nx.draw_networkx_edges(sub_g, pos, ax=axes[0], alpha=0.35, width=edge_weights, edge_color="#64748B")
    nx.draw_networkx_nodes(sub_g, pos, ax=axes[0], node_color=node_colors, node_size=node_sizes, edgecolors="#0F172A", linewidths=1.0)
    nx.draw_networkx_labels(sub_g, pos, labels={n: f"S{n+1}" for n in sub_g.nodes()}, font_size=8, font_weight="bold", ax=axes[0])
    from matplotlib.patches import Patch
    axes[0].legend(
        handles=[Patch(color="#10B981", label="Selected Top-6 Summary Sentences"), Patch(color="#0EA5E9", label="High-Centrality Supporting Sentences")],
        loc="upper right",
        fontsize=8,
    )
    axes[0].set_title("(a) TextRank Sentence Similarity Graph (Top-25 Subgraph)", fontweight="bold")
    axes[0].axis("off")

    sent_positions = np.arange(1, len(art1_sents) + 1)
    axes[1].plot(sent_positions, tr_scores, color="#1E3A8A", lw=1.3, alpha=0.85, label="PageRank Centrality")
    axes[1].scatter(
        [i + 1 for i in top_idx_1],
        [tr_scores[i] for i in top_idx_1],
        color="#10B981",
        s=90,
        zorder=5,
        edgecolor="#0F172A",
        label="Selected Summary Sentences",
    )
    for i in top_idx_1:
        axes[1].annotate(f"S{i+1}", (i + 1, tr_scores[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")
    axes[1].set_title(f"(b) Sentence Saliency Profile Across Article 1 ({len(art1_sents)} Sentences)", fontweight="bold")
    axes[1].set_xlabel("Sentence Position Index (Start -> End)")
    axes[1].set_ylabel("TextRank PageRank Score")
    axes[1].legend(loc="upper right", fontsize=8.5)

    fig.suptitle("Q3.3 TextRank Extractive Summarisation — research_1.txt (Mixed Methods Research)", fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig_q3_article1_summary.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Q3.4: AGGREGATE MULTI-DOCUMENT SUMMARISATION OF ARTICLES 2-5 (LSA + MMR)
    # =========================================================================
    agg_sents, agg_doc_labels = [], []
    for i in range(1, len(df_corpus)):
        for s in df_corpus.loc[i, "sentences"]:
            agg_sents.append(s)
            agg_doc_labels.append(df_corpus.loc[i, "file_name"])

    sel_agg_idx, lsa_saliency, U_lsa, svd_agg = lsa_mmr_summarise(
        agg_sents, agg_doc_labels, n_components=4, top_n=8, lam=0.68, max_per_doc=2
    )
    agg_summary_records = [
        {
            "rank": r + 1,
            "source_file": agg_doc_labels[idx],
            "source_label": ARTICLE_META[agg_doc_labels[idx]]["short_label"],
            "lsa_saliency": float(lsa_saliency[idx]),
            "sentence": agg_sents[idx],
        }
        for r, idx in enumerate(sel_agg_idx)
    ]
    print(f"\n=== Q3.4 AGGREGATE LSA+MMR SUMMARY (Articles 2-5, {len(agg_sents)} sents -> {len(sel_agg_idx)} sents) ===")
    for rec in agg_summary_records:
        print(f"  [{rec['rank']}] ({rec['source_file']}, saliency={rec['lsa_saliency']:.3f}) {rec['sentence']}")

    # Plot Q3.4 Aggregate Two-Method Visualisation: LSA Heatmap + Saliency + WordCloud
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), dpi=300)
    # Vis 1: Article-by-Latent Concept Mean Absolute Loading Heatmap
    u_abs_df = pd.DataFrame(np.abs(U_lsa), columns=[f"Concept {c+1}" for c in range(4)])
    u_abs_df["Article"] = [ARTICLE_META[fn]["short_label"] for fn in agg_doc_labels]
    hm_data = u_abs_df.groupby("Article").mean()
    sns.heatmap(hm_data, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0], cbar_kws={"label": "Mean |Concept Loading|"})
    axes[0].set_title("(a) LSA Latent Concept Alignment (Articles 2–5)", fontweight="bold")
    axes[0].set_ylabel("")

    # Vis 2: Selected Summary Sentence LSA Saliency Ranked Bars
    bar_labels = [f"#{r['rank']} ({r['source_file'].replace('.txt','')})" for r in agg_summary_records][::-1]
    bar_vals = [r["lsa_saliency"] for r in agg_summary_records][::-1]
    art_palette = {"research_2.txt": "#0EA5E9", "research_3.txt": "#10B981", "research_4.txt": "#1E3A8A", "research_5.txt": "#F59E0B"}
    bar_cols = [art_palette[r["source_file"]] for r in agg_summary_records][::-1]
    axes[1].barh(bar_labels, bar_vals, color=bar_cols)
    axes[1].set_xlim(0, 1.12)
    axes[1].set_title(f"(b) Top-{len(sel_agg_idx)} Aggregate MMR Sentence Saliency", fontweight="bold")
    axes[1].set_xlabel("Normalised LSA Singular Saliency")

    # Vis 3: Aggregate Denoised Phrasal WordCloud (Articles 2-5)
    agg_tokens = [tok for i in range(1, len(phrased_docs)) for tok in phrased_docs[i]]
    wc = WordCloud(
        width=800,
        height=520,
        background_color="white",
        colormap="ocean",
        max_words=90,
        collocations=False,
        random_state=42,
    ).generate(" ".join(agg_tokens))
    axes[2].imshow(wc, interpolation="bilinear")
    axes[2].axis("off")
    axes[2].set_title("(c) Aggregate Corpus Word Cloud (Articles 2–5)", fontweight="bold")

    fig.suptitle("Q3.4 Multi-Document Aggregate LSA+MMR Summarisation & Visualisation", fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig_q3_aggregate_summary.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Q4.1: LDA TOPIC MODELLING ON SEGMENTED PASSAGES + COHERENCE SWEEP
    # =========================================================================
    all_clean_sents = [s for slist in df_corpus["sentences"] for s in slist]
    all_sent_docs = [fn for fn, slist in zip(df_corpus["file_name"], df_corpus["sentences"]) for _ in slist]
    passages = build_passages(all_clean_sents, all_sent_docs, target_words=150)
    passage_docs = [p[0] for p in passages]
    passage_texts = [p[1] for p in passages]
    passage_tokens = [bigram_phraser[tokenize_for_topics(t, stop_words, lemmatizer)] for t in passage_texts]

    id2word = corpora.Dictionary(passage_tokens)
    id2word.filter_extremes(no_below=2, no_above=0.65)
    bow_corpus = [id2word.doc2bow(toks) for toks in passage_tokens]

    k_candidates = [3, 4, 5, 6, 7, 8]
    sweep_rows = []
    lda_models = {}
    for k in k_candidates:
        lda_k = LdaModel(
            corpus=bow_corpus,
            id2word=id2word,
            num_topics=k,
            random_state=42,
            update_every=1,
            chunksize=25,
            passes=45,
            iterations=400,
            alpha="symmetric",
            eta="auto",
            per_word_topics=True,
        )
        cm_k = CoherenceModel(
            model=lda_k, texts=passage_tokens, dictionary=id2word, coherence="c_v", processes=1
        )
        cv_score = float(cm_k.get_coherence())
        log_perp = float(lda_k.log_perplexity(bow_corpus))
        perp = float(np.exp(-log_perp))
        sweep_rows.append({"k": k, "coherence_cv": cv_score, "log_perplexity": log_perp, "perplexity": perp})
        lda_models[k] = lda_k
        print(f"[Q4.1 LDA Sweep] k={k}: C_v={cv_score:.4f}, Perplexity={perp:.2f}")

    sweep_df = pd.DataFrame(sweep_rows)
    best_k = int(sweep_df.loc[sweep_df["coherence_cv"].idxmax(), "k"])
    best_lda = lda_models[best_k]
    print(f"Selected optimal LDA topics: k={best_k} (highest C_v coherence={sweep_df['coherence_cv'].max():.4f})")

    # Extract top 10 words per topic + posterior weights
    topic_words_records = []
    for tid in range(best_k):
        top_terms = best_lda.show_topic(tid, topn=10)
        topic_words_records.append({
            "topic_id": tid + 1,
            "top_words": [{"word": w, "prob": float(p)} for w, p in top_terms],
            "words_str": ", ".join([f"{w} ({p:.3f})" for w, p in top_terms]),
        })

    # Compute Document-Topic distributions aggregated over passages per research article AND at article level
    doc_topic_mat = np.zeros((len(df_corpus), best_k))
    for i, fname in enumerate(df_corpus["file_name"]):
        art_bow = id2word.doc2bow(phrased_docs[i])
        dist = best_lda.get_document_topics(art_bow, minimum_probability=0.0)
        for tid, prob in dist:
            doc_topic_mat[i, tid] = prob

    doc_topic_df = pd.DataFrame(
        doc_topic_mat,
        columns=[f"Topic {t+1}" for t in range(best_k)],
    )
    doc_topic_df.insert(0, "Article", [ARTICLE_META[fn]["short_label"] for fn in df_corpus["file_name"]])
    doc_topic_df["Dominant_Topic"] = [f"Topic {int(np.argmax(doc_topic_mat[i]) + 1)}" for i in range(len(df_corpus))]
    doc_topic_df.to_csv(OUT_DIR / "q4_doc_topic_distribution.csv", index=False)

    # Identify human-readable labels for each discovered topic from top words & dominant article
    for rec in topic_words_records:
        tid0 = rec["topic_id"] - 1
        dom_art = doc_topic_df.loc[doc_topic_mat[:, tid0].argmax(), "Article"]
        rec["peak_article"] = dom_art
        print(f"  Topic {rec['topic_id']} (peak={dom_art}): {rec['words_str']}")

    # Plot Q4.1 Topics & Model Selection Figure
    ncols_grid = 2 if best_k <= 3 else 3
    n_rows_grid = (1 + best_k + ncols_grid - 1) // ncols_grid
    fig = plt.figure(figsize=(16.5 if ncols_grid == 2 else 17.5, 5.0 * n_rows_grid), dpi=300)
    gs = fig.add_gridspec(n_rows_grid, ncols_grid, hspace=0.45, wspace=0.72 if ncols_grid == 2 else 0.58)
    ax_coh = fig.add_subplot(gs[0, 0])
    ax_coh2 = ax_coh.twinx()
    l1 = ax_coh.plot(sweep_df["k"], sweep_df["coherence_cv"], "o-", color="#0EA5E9", lw=2.2, label=r"$C_v$ Coherence")
    l2 = ax_coh2.plot(sweep_df["k"], sweep_df["perplexity"], "s--", color="#F59E0B", lw=1.8, label="Perplexity")
    ax_coh.axvline(best_k, color="#10B981", linestyle=":", lw=1.8, label=f"Optimal n={best_k}")
    ax_coh.set_title("(a) LDA Coherence & Perplexity Sweep", fontweight="bold")
    ax_coh.set_xlabel("Number of Topics (n)")
    ax_coh.set_ylabel(r"$C_v$ Coherence", color="#0EA5E9")
    ax_coh2.set_ylabel("Perplexity", color="#D97706")
    lines_leg = l1 + l2
    ax_coh.legend(lines_leg, [ll.get_label() for ll in lines_leg], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=True)

    topic_colors = ["#1E3A8A", "#0EA5E9", "#10B981", "#8B5CF6", "#F59E0B", "#EF4444", "#14B8A6", "#EC4899"]
    for idx_t, rec in enumerate(topic_words_records):
        row = (idx_t + 1) // ncols_grid
        col = (idx_t + 1) % ncols_grid
        ax_t = fig.add_subplot(gs[row, col])
        words = [w["word"] for w in rec["top_words"]][::-1]
        probs = [w["prob"] for w in rec["top_words"]][::-1]
        ax_t.barh(words, probs, color=topic_colors[idx_t % len(topic_colors)])
        ax_t.set_title(f"Topic {rec['topic_id']} Top Words\n(Peak: {rec['peak_article']})", fontweight="bold", fontsize=10)
        ax_t.set_xlabel(r"Term Posterior Probability $\beta$")

    fig.suptitle(f"Q4.1 Latent Dirichlet Allocation (LDA, n = {best_k} Topics) Selection & Word Profiles", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(FIG_DIR / "fig_q4_lda_topics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot Q4.1 Document-Topic Distribution Heatmap & Stacked Bars
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 5.2), dpi=300)
    sns.heatmap(
        doc_topic_df.set_index("Article")[[f"Topic {t+1}" for t in range(best_k)]],
        annot=True,
        fmt=".3f",
        cmap="Blues",
        ax=axes[0],
        cbar_kws={"label": r"Topic Proportion $\theta$"},
    )
    axes[0].set_title("(a) Document-Topic Probability Matrix (Theta)", fontweight="bold")
    axes[0].set_ylabel("")

    doc_topic_df.set_index("Article")[[f"Topic {t+1}" for t in range(best_k)]].plot(
        kind="barh",
        stacked=True,
        color=topic_colors[:best_k],
        ax=axes[1],
    )
    axes[1].set_title("(b) Stacked Topic Mixture per Research Article", fontweight="bold")
    axes[1].set_xlabel(r"Cumulative Topic Probability ($\Sigma\theta = 1.0$)")
    axes[1].set_ylabel("")
    axes[1].invert_yaxis()
    axes[1].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8.5)

    fig.suptitle("Q4.1 Article-Level LDA Topic Probability Distributions", fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93], w_pad=3.5)
    fig.savefig(FIG_DIR / "fig_q4_lda_distributions.png", dpi=300)
    plt.close(fig)

    # =========================================================================
    # Q4.2: INTERACTIVE pyLDAvis INTERTOPIC DISTANCE MAP + SALIENT WORDS VISUAL
    # =========================================================================
    vis_data = pyLDAvis.gensim_models.prepare(
        best_lda, bow_corpus, id2word, mds="mmds", sort_topics=False, n_jobs=1
    )
    pyldavis_html = OUT_DIR / "pyldavis_intertopic_map.html"
    pyLDAvis.save_html(vis_data, str(pyldavis_html))
    print(f"Saved interactive pyLDAvis HTML -> {pyldavis_html}")

    # Generate high-resolution 300 DPI static rendering of the pyLDAvis Intertopic Distance Map + Top-25 Salient Terms
    tcoords = vis_data.topic_coordinates.copy()
    tinfo = vis_data.topic_info.copy()
    default_salient = tinfo[tinfo["Category"] == "Default"].sort_values("logprob", ascending=False).head(25)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), dpi=300)
    axes[0].axhline(0, color="#CBD5E1", lw=1, zorder=1)
    axes[0].axvline(0, color="#CBD5E1", lw=1, zorder=1)
    bubble_sizes = (tcoords["Freq"].values / tcoords["Freq"].values.max()) * 2800 + 600
    for idx_b, row in tcoords.iterrows():
        tid = int(row["topics"])
        axes[0].scatter(row["x"], row["y"], s=bubble_sizes[list(tcoords.index).index(idx_b)], color="#93C5FD", alpha=0.65, edgecolors="#1E3A8A", lw=2, zorder=3)
        axes[0].text(row["x"], row["y"], f"Topic {tid}\n({row['Freq']:.1f}%)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0F172A", zorder=4)
    x_min, x_max = float(tcoords["x"].min()), float(tcoords["x"].max())
    y_min, y_max = float(tcoords["y"].min()), float(tcoords["y"].max())
    x_pad = max(0.14, 0.28 * max(x_max - x_min, 0.2))
    y_pad = max(0.14, 0.28 * max(y_max - y_min, 0.2))
    axes[0].set_xlim(x_min - x_pad, x_max + x_pad)
    axes[0].set_ylim(y_min - y_pad, y_max + y_pad)
    axes[0].set_title("(a) pyLDAvis Intertopic Distance Map (via MMDS)", fontweight="bold")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    ds = default_salient.iloc[::-1]
    y_idx = np.arange(len(ds))
    axes[1].barh(y_idx, ds["Total"], color="#EF4444", alpha=0.82, label="Overall Term Frequency (Ranked by pyLDAvis Saliency)")
    axes[1].set_yticks(y_idx)
    axes[1].set_yticklabels(ds["Term"], fontsize=8.5)
    axes[1].set_title("(b) Top-25 Most Salient Terms (pyLDAvis Saliency View)", fontweight="bold")
    axes[1].set_xlabel("Overall Corpus Frequency (Ordered Top-to-Bottom by Saliency)")
    axes[1].legend(loc="lower right", fontsize=8.5)

    fig.suptitle(f"Q4.2 Interactive pyLDAvis Intertopic Distance Map & Salient Terms (n = {best_k} Topics)", fontsize=13.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "fig_q4_pyldavis_map.png", dpi=300)
    plt.close(fig)

    # Save JSON summary of Q3 & Q4 results
    q3_q4_results = {
        "corpus_summary": df_save.to_dict(orient="records"),
        "q3_3_textrank": {
            "article": "research_1.txt",
            "total_clean_sentences": len(art1_sents),
            "selected_indices": [int(i + 1) for i in top_idx_1],
            "summary_sentences": [
                {"sent_num": int(i + 1), "score": float(tr_scores[i]), "text": art1_sents[i]}
                for i in top_idx_1
            ],
            "compression_pct": float(art1_comp_pct),
        },
        "q3_4_aggregate_lsa": {
            "total_clean_sentences_art2_5": len(agg_sents),
            "svd_explained_variance_4comp": [float(v) for v in svd_agg.explained_variance_ratio_],
            "summary_sentences": agg_summary_records,
        },
        "q4_1_lda": {
            "num_passages": len(passages),
            "vocab_size_filtered": len(id2word),
            "coherence_sweep": sweep_rows,
            "optimal_k": best_k,
            "topics": topic_words_records,
            "doc_topic_table": doc_topic_df.to_dict(orient="records"),
        },
        "q4_2_pyldavis": {
            "html_file": str(pyldavis_html.name),
            "topic_coordinates": tcoords[["topics", "x", "y", "Freq"]].to_dict(orient="records"),
            "top_salient_terms": default_salient[["Term", "Total", "Freq"]].to_dict(orient="records"),
        },
    }
    with open(OUT_DIR / "q3_q4_results.json", "w", encoding="utf-8") as f:
        json.dump(q3_q4_results, f, indent=2)
    print("Q3 & Q4 pipeline finished cleanly.")


if __name__ == "__main__":
    main()
