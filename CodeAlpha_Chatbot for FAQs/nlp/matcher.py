"""FAQ matching engine with TF-IDF and semantic fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessor import correct_spelling, normalize_text

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:  # pragma: no cover
    SentenceTransformer = None
    util = None


ROOT_DIR = Path(__file__).resolve().parent.parent
FAQ_PATH = ROOT_DIR / "data" / "faq.json"

TFIDF_THRESHOLD = 0.30
SEMANTIC_THRESHOLD = 0.45


class FAQMatcher:
    """Matches user questions to FAQ answers using layered NLP retrieval."""

    def __init__(self, faq_path: Path = FAQ_PATH):
        with faq_path.open("r", encoding="utf-8") as file:
            self.faq_data: List[Dict[str, Any]] = json.load(file)

        self.questions = [item["question"] for item in self.faq_data]
        self.normalized_questions = [normalize_text(question) for question in self.questions]

        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.normalized_questions)

        self.semantic_model = None
        self.semantic_embeddings = None
        if SentenceTransformer:
            try:
                self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.semantic_embeddings = self.semantic_model.encode(
                    self.questions, convert_to_tensor=True, normalize_embeddings=True
                )
            except Exception:
                self.semantic_model = None
                self.semantic_embeddings = None

    def _tfidf_match(self, processed_query: str) -> Tuple[int, float]:
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        best_idx = int(np.argmax(scores))
        return best_idx, float(scores[best_idx])

    def _semantic_match(self, query: str) -> Tuple[int, float]:
        if not self.semantic_model or self.semantic_embeddings is None or util is None:
            return -1, 0.0

        query_embedding = self.semantic_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        cosine_scores = util.cos_sim(query_embedding, self.semantic_embeddings)[0].cpu().numpy()
        best_idx = int(np.argmax(cosine_scores))
        return best_idx, float(cosine_scores[best_idx])

    def find_best_match(self, user_query: str) -> Dict[str, Any]:
        """Return best answer and metadata, with fallback for weak matches."""
        corrected_query = correct_spelling(user_query)
        processed_query = normalize_text(corrected_query)
        if not processed_query:
            return {
                "answer": "Please enter a meaningful question so I can help you.",
                "category": "General",
                "confidence": 0.0,
                "matched_question": "",
                "method": "fallback",
                "corrected_message": corrected_query,
            }

        tfidf_idx, tfidf_score = self._tfidf_match(processed_query)
        method = "tfidf"
        best_idx = tfidf_idx
        best_score = tfidf_score

        if tfidf_score < TFIDF_THRESHOLD:
            semantic_idx, semantic_score = self._semantic_match(corrected_query)
            if semantic_score > best_score:
                best_idx = semantic_idx
                best_score = semantic_score
                method = "semantic"

        threshold = TFIDF_THRESHOLD if method == "tfidf" else SEMANTIC_THRESHOLD
        if best_idx == -1 or best_score < threshold:
            return {
                "answer": (
                    "I could not find a confident answer for that question yet. "
                    "Please rephrase it or try one of the suggested questions."
                ),
                "category": "Fallback",
                "confidence": round(best_score, 4),
                "matched_question": "",
                "method": "fallback",
                "corrected_message": corrected_query,
            }

        matched_item = self.faq_data[best_idx]
        return {
            "answer": matched_item["answer"],
            "category": matched_item["category"],
            "confidence": round(best_score, 4),
            "matched_question": matched_item["question"],
            "method": method,
            "corrected_message": corrected_query,
        }
