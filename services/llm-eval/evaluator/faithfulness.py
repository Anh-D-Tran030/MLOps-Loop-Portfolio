"""
FaithfulnessScorer: DeBERTa NLI entailment scorer.

Uses cross-encoder/nli-deberta-v3-large via transformers pipeline.
Deterministic, CPU-based, no API calls.
"""

from __future__ import annotations

import logging

from transformers import pipeline

logger = logging.getLogger(__name__)

# Label constants (DeBERTa-v3 NLI output labels)
_ENTAILMENT = "ENTAILMENT"
_CONTRADICTION = "CONTRADICTION"
_NEUTRAL = "NEUTRAL"


class FaithfulnessScorer:
    """Score faithfulness of an answer given a supporting context.

    Uses cross-encoder/nli-deberta-v3-large to predict entailment between
    context and answer. Returns a float in [0, 1] where higher values indicate
    the answer is better supported by the context.

    Label mapping:
        ENTAILMENT   -> model confidence score (high = faithful)
        CONTRADICTION -> 1 - model confidence score (low = unfaithful)
        NEUTRAL      -> 0.5 (ambiguous)
    """

    MODEL_ID = "cross-encoder/nli-deberta-v3-large"

    def __init__(self, device: int = -1) -> None:
        """Initialise the scorer and load the model.

        Args:
            device: -1 for CPU (default), 0+ for GPU index.
        """
        logger.info("Loading FaithfulnessScorer model: %s", self.MODEL_ID)
        self._pipe = pipeline(
            task="text-classification",
            model=self.MODEL_ID,
            device=device,
        )
        logger.info("FaithfulnessScorer ready.")

    def score(self, context: str, answer: str) -> float:
        """Score how faithfully the answer is supported by the context.

        Args:
            context: The source context (e.g. retrieved document passage).
            answer:  The generated answer to evaluate.

        Returns:
            Float in [0, 1]. Values closer to 1 mean the context entails the
            answer; values closer to 0 mean contradiction.
        """
        if not context or not isinstance(context, str):
            raise ValueError("context must be a non-empty string")
        if not answer or not isinstance(answer, str):
            raise ValueError("answer must be a non-empty string")

        # Format: standard NLI premise/hypothesis separator
        model_input = f"{context} [SEP] {answer}"

        result = self._pipe(model_input, truncation=True, max_length=512)

        if not result:
            logger.warning("Pipeline returned empty result; defaulting to 0.5")
            return 0.5

        item = result[0] if isinstance(result, list) else result
        label: str = item["label"].upper()
        score: float = float(item["score"])

        if label == _ENTAILMENT:
            return float(score)
        elif label == _CONTRADICTION:
            return float(1.0 - score)
        else:
            # NEUTRAL or any unexpected label
            return 0.5
