"""
PrecisionScorer: Context precision scorer.

Originally intended to use ragas.metrics.context_precision, but ragas==0.1.21
fails to import on Python 3.14 because it depends on langchain_core.pydantic_v1
which was removed in langchain-core >=0.2. Substituting with a deterministic
token-overlap precision metric: |answer_tokens ∩ context_tokens| / |answer_tokens|.

This is consistent with the plan risk register (note 3) which explicitly allows
this substitution when ragas forces an LLM judge or cannot be imported.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenisation, stripping punctuation."""
    return set(re.findall(r"\b\w+\b", text.lower()))


class PrecisionScorer:
    """Score context precision using token-overlap between answer and context.

    Measures what fraction of the answer's tokens are also present in the
    context. A score of 1.0 means every answer token is grounded in the
    context; 0.0 means no overlap.

    Substitution note: ragas==0.1.21 requires langchain_core.pydantic_v1
    which was removed in langchain-core>=0.2 (installed as a transitive dep).
    This deterministic fallback preserves CI gate correctness without an LLM
    judge or OpenAI key. See plan.md risk register note 3.
    """

    def score(self, question: str, context: str, answer: str) -> float:
        """Compute token-overlap precision of the answer against the context.

        Args:
            question: The original question (accepted for API compatibility
                      with RAGAS interface; not used in token-overlap calc).
            context:  The source passage the answer should be grounded in.
            answer:   The generated answer to evaluate.

        Returns:
            Float in [0, 1]. Values closer to 1 mean the answer is well
            grounded in the context.
        """
        if not context or not isinstance(context, str):
            raise ValueError("context must be a non-empty string")
        if not answer or not isinstance(answer, str):
            raise ValueError("answer must be a non-empty string")

        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context)

        if not answer_tokens:
            logger.warning("Answer tokenised to empty set; returning 0.0")
            return 0.0

        overlap = answer_tokens & context_tokens
        precision = len(overlap) / len(answer_tokens)
        return float(precision)
