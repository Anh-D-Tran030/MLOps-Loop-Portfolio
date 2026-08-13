"""
Unit tests for all three scorers and the CLI.

No mocking — models are loaded for real.
Model downloads are cached to ~/.cache/huggingface after the first run.
"""

from __future__ import annotations

import json
import tempfile

import pytest
from click.testing import CliRunner
from evaluator.cli import cli
from evaluator.faithfulness import FaithfulnessScorer
from evaluator.precision import PrecisionScorer
from evaluator.relevance import RelevanceScorer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def faithful_scorer() -> FaithfulnessScorer:
    return FaithfulnessScorer()


@pytest.fixture(scope="module")
def relevant_scorer() -> RelevanceScorer:
    return RelevanceScorer()


@pytest.fixture(scope="module")
def precision_scorer() -> PrecisionScorer:
    return PrecisionScorer()


# ---------------------------------------------------------------------------
# 1. Faithfulness — entailment pair should score high
# ---------------------------------------------------------------------------


def test_faithfulness_entailment_returns_high_score(faithful_scorer):
    """Context and answer are nearly identical: model should return ENTAILMENT."""
    context = (
        "The Eiffel Tower is located in Paris, France. "
        "It was constructed between 1887 and 1889."
    )
    answer = "The Eiffel Tower is in Paris, France."
    result = faithful_scorer.score(context=context, answer=answer)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    assert result > 0.7, f"Expected > 0.7 for entailment pair, got {result:.4f}"


# ---------------------------------------------------------------------------
# 2. Faithfulness — contradiction pair should score low
# ---------------------------------------------------------------------------


def test_faithfulness_contradiction_returns_low_score(faithful_scorer):
    """Answer directly contradicts the context: should score low."""
    context = "The Eiffel Tower is located in Paris, France."
    answer = "The Eiffel Tower is located in London, England."
    result = faithful_scorer.score(context=context, answer=answer)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    assert result < 0.3, f"Expected < 0.3 for contradiction pair, got {result:.4f}"


# ---------------------------------------------------------------------------
# 3. Relevance — identical strings should score near 1.0
# ---------------------------------------------------------------------------


def test_relevance_identical_returns_high_score(relevant_scorer):
    """Same text as question and answer: cosine similarity should be ~1.0."""
    text = "What is the capital city of Australia?"
    result = relevant_scorer.score(question=text, answer=text)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    assert result > 0.9, f"Expected > 0.9 for identical strings, got {result:.4f}"


# ---------------------------------------------------------------------------
# 4. Relevance — unrelated strings should score low
# ---------------------------------------------------------------------------


def test_relevance_unrelated_returns_low_score(relevant_scorer):
    """Semantically unrelated question and answer should score below 0.5."""
    question = "What is photosynthesis in plants?"
    answer = "The stock market closed higher on Tuesday due to tech gains."
    result = relevant_scorer.score(question=question, answer=answer)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
    assert result < 0.5, f"Expected < 0.5 for unrelated pair, got {result:.4f}"


# ---------------------------------------------------------------------------
# 5. Precision — returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_precision_scorer_returns_float_in_range(precision_scorer):
    """PrecisionScorer must return a float strictly within [0, 1]."""
    context = (
        "Python is a high-level programming language known for its clear syntax. "
        "It supports object-oriented, functional, and procedural styles."
    )
    question = "What kind of language is Python?"
    answer = "Python is a high-level programming language with clear syntax."
    result = precision_scorer.score(question=question, context=context, answer=answer)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 6. CLI — high-quality JSONL exits with code 0
# ---------------------------------------------------------------------------


def test_cli_pass_exits_zero():
    """CLI should exit 0 when all mean scores are above the threshold."""
    samples = [
        {
            "question": "Where is the Eiffel Tower located?",
            "context": "The Eiffel Tower is located in Paris, France.",
            "answer": "The Eiffel Tower is located in Paris, France.",
        },
        {
            "question": "What is the boiling point of water?",
            "context": "Water boils at 100 degrees Celsius at standard pressure.",
            "answer": "Water boils at 100 degrees Celsius.",
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "context": "Romeo and Juliet is a tragedy written by William Shakespeare.",
            "answer": "William Shakespeare wrote Romeo and Juliet.",
        },
    ]

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for s in samples:
            tmp.write(json.dumps(s) + "\n")
        tmp_path = tmp.name

    result = runner.invoke(cli, ["score", tmp_path, "--threshold", "0.7"])
    assert (
        result.exit_code == 0
    ), f"Expected exit code 0, got {result.exit_code}.\nOutput:\n{result.output}"
    assert "OVERALL: PASS" in result.output


# ---------------------------------------------------------------------------
# 7. CLI — low-quality JSONL exits with code 1
# ---------------------------------------------------------------------------


def test_cli_fail_exits_one():
    """CLI should exit 1 when at least one mean score is below threshold."""
    # Use nonsense answers that contradict/are irrelevant to questions
    samples = [
        {
            "question": "What is the speed of light?",
            "context": "The speed of light in a vacuum is approximately 299,792 km/s.",
            "answer": "Elephants migrate south during winter months every year.",
        },
        {
            "question": "Where is Mount Everest?",
            "context": "Mount Everest is in the Himalayas on the Nepal-Tibet border.",
            "answer": "Jazz music originated in New Orleans in the early 20th century.",
        },
    ]

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for s in samples:
            tmp.write(json.dumps(s) + "\n")
        tmp_path = tmp.name

    result = runner.invoke(cli, ["score", tmp_path, "--threshold", "0.7"])
    assert (
        result.exit_code == 1
    ), f"Expected exit code 1, got {result.exit_code}.\nOutput:\n{result.output}"
    assert "FAIL" in result.output


# ---------------------------------------------------------------------------
# 8. CLI — missing required field raises error
# ---------------------------------------------------------------------------


def test_cli_missing_field_raises_error():
    """CLI should exit non-zero and report an error for JSONL missing a field."""
    # 'context' field is missing
    bad_samples = [
        {"question": "What is Python?", "answer": "A programming language."},
    ]

    runner = CliRunner()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for s in bad_samples:
            tmp.write(json.dumps(s) + "\n")
        tmp_path = tmp.name

    result = runner.invoke(cli, ["score", tmp_path, "--threshold", "0.7"])
    assert result.exit_code != 0, (
        f"Expected non-zero exit for missing field, got {result.exit_code}.\n"
        f"Output:\n{result.output}"
    )
    assert "context" in result.output.lower() or "missing" in result.output.lower()
