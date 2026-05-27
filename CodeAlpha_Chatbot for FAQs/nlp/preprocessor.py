"""Text preprocessing utilities for FAQ chatbot NLP pipeline."""

from __future__ import annotations

import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker


def _safe_nltk_download(resource: str) -> None:
    """Download NLTK resources when missing, without crashing startup."""
    try:
        nltk.data.find(resource)
    except LookupError:
        package = resource.split("/")[-1]
        nltk.download(package, quiet=True)


_safe_nltk_download("tokenizers/punkt")
_safe_nltk_download("corpora/stopwords")
_safe_nltk_download("corpora/wordnet")
_safe_nltk_download("corpora/omw-1.4")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
SPELL_CHECKER = SpellChecker(distance=1)


def normalize_text(text: str) -> str:
    """Normalize text using lowercase, tokenization, stopword removal, and lemmatization."""
    if not text:
        return ""

    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = lowered.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(lowered)
    cleaned_tokens: List[str] = []
    for token in tokens:
        if token.isalpha() and token not in STOP_WORDS:
            lemma = LEMMATIZER.lemmatize(token)
            cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


def correct_spelling(text: str) -> str:
    """Correct obvious misspellings while preserving overall sentence structure."""
    if not text:
        return ""

    words = text.split()
    corrected_words: List[str] = []
    for word in words:
        bare = re.sub(r"[^A-Za-z]", "", word)
        if not bare or len(bare) <= 2:
            corrected_words.append(word)
            continue

        suggestion = SPELL_CHECKER.correction(bare)
        if suggestion and suggestion != bare.lower():
            corrected_words.append(word.replace(bare, suggestion))
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)
