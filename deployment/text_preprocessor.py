"""Shared text denoising and preprocessing utilities for the Researchly system.

Provides deterministic text cleaning, contraction expansion, noise stripping
(URLs, DOIs, LaTeX commands, HTML tags, bracketed citations, standalone
digits), stopword filtering, and WordNet lemmatisation packaged in a
scikit-learn compatible transformer for leakage-free pipelines and deployment.
"""

import re
import unicodedata
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HTML_RE = re.compile(r"<[^>]+>")
LATEX_MATH_RE = re.compile(r"\$[^$]*\$")
LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+(?:\{[^}]*\})?")
BRACKET_CITE_RE = re.compile(r"\[\s*\d+(?:\s*[,;\-–]\s*\d+)*\s*\]")
PAREN_YEAR_RE = re.compile(r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)")
SEPARATOR_RE = re.compile(r"[-–—/]+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTISPACE_RE = re.compile(r"\s+")

CONTRACTIONS = (
    (re.compile(r"\bwon't\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bcan't\b", re.IGNORECASE), "cannot"),
    (re.compile(r"\bshan't\b", re.IGNORECASE), "shall not"),
    (re.compile(r"n't\b", re.IGNORECASE), " not"),
    (re.compile(r"'re\b", re.IGNORECASE), " are"),
    (re.compile(r"'ve\b", re.IGNORECASE), " have"),
    (re.compile(r"'ll\b", re.IGNORECASE), " will"),
    (re.compile(r"'d\b", re.IGNORECASE), " would"),
    (re.compile(r"'m\b", re.IGNORECASE), " am"),
    (re.compile(r"'s\b", re.IGNORECASE), ""),
)

# Short domain tokens such as 'ai' and 'ml' must never be removed by length rules.
PRESERVED_DOMAIN_TOKENS = {"ai", "ml", "dl", "ar", "var", "nlp", "svm"}


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora silently if missing."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def normalize_unicode_and_contractions(text: str) -> str:
    """Normalise typographic punctuation/accents and expand English contractions."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    for pattern, repl in CONTRACTIONS:
        text = pattern.sub(repl, text)
    return text


def strip_structured_noise(text: str) -> str:
    """Remove URLs, DOIs, emails, HTML/LaTeX markup, citations, and digits."""
    text = URL_RE.sub(" ", text)
    text = DOI_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = LATEX_MATH_RE.sub(" ", text)
    text = LATEX_CMD_RE.sub(" ", text)
    text = BRACKET_CITE_RE.sub(" ", text)
    text = PAREN_YEAR_RE.sub(" ", text)
    text = SEPARATOR_RE.sub(" ", text)
    text = text.lower()
    text = NON_ALPHA_RE.sub(" ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def denoise_abstract(
    text: str,
    stop_words: set = None,
    lemmatizer: WordNetLemmatizer = None,
    return_tokens: bool = False,
):
    """Apply full NLP denoising (cleaning, stopword removal, lemmatisation)."""
    ensure_nltk_resources()
    if stop_words is None:
        stop_words = set(stopwords.words("english")) - {"no", "nor", "not"}
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

    cleaned = strip_structured_noise(normalize_unicode_and_contractions(text))
    tokens = word_tokenize(cleaned)
    processed: List[str] = []
    for tok in tokens:
        if tok in stop_words:
            continue
        lemma = lemmatizer.lemmatize(tok)
        if lemma in stop_words:
            continue
        if len(lemma) >= 2 or lemma in PRESERVED_DOMAIN_TOKENS:
            processed.append(lemma)
    if return_tokens:
        return processed
    return " ".join(processed)


class TextDenoiserTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer wrapping denoise_abstract for pipeline portability."""

    def fit(self, X: Iterable[str], y=None):
        ensure_nltk_resources()
        return self

    def transform(self, X: Iterable[str]) -> List[str]:
        ensure_nltk_resources()
        stop_words = set(stopwords.words("english")) - {"no", "nor", "not"}
        lemmatizer = WordNetLemmatizer()
        return [denoise_abstract(doc, stop_words=stop_words, lemmatizer=lemmatizer) for doc in X]
