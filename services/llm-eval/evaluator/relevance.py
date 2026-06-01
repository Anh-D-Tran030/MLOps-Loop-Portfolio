"""
RelevanceScorer: Sentence Transformers cosine similarity scorer.

Uses sentence-transformers/all-MiniLM-L6-v2 to compute semantic similarity
between a question and an answer. Deterministic, offline, no API calls.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """Score how relevant an answer is to a given question.

    Embeds both strings using all-MiniLM-L6-v2 and returns the cosine
    similarity clipped to [0, 1]. Higher values indicate the answer
    addresses the question more directly.
    """

    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        """Initialise the scorer and load the model."""
        logger.info("Loading RelevanceScorer model: %s", self.MODEL_ID)
        self._model = SentenceTransformer(self.MODEL_ID)
        logger.info("RelevanceScorer ready.")

    def score(self, question: str, answer: str) -> float:
        """Compute cosine similarity between question and answer embeddings.

        Args:
            question: The original question or query.
            answer:   The generated answer to evaluate.

        Returns:
            Float in [0, 1]. Values closer to 1 mean the answer is semantically
            similar / relevant to the question.
        """
        if not question or not isinstance(question, str):
            raise ValueError("question must be a non-empty string")
        if not answer or not isinstance(answer, str):
            raise ValueError("answer must be a non-empty string")

        embeddings = self._model.encode(
            [question, answer],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        q_emb: np.ndarray = embeddings[0]
        a_emb: np.ndarray = embeddings[1]

        # Cosine similarity on unit-norm vectors reduces to dot product.
        # Clip to [0, 1]: negative cosine similarity is possible but we treat
        # scores below 0 as 0 (completely unrelated).
        raw_sim: float = float(np.dot(q_emb, a_emb))
        return float(np.clip(raw_sim, 0.0, 1.0))
