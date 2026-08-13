"""
mlops-eval CLI: score <input_file> --threshold 0.7

Reads a JSONL file, scores each sample on faithfulness / relevance / precision,
prints per-sample results and mean summary, then exits with code 0 (all PASS)
or 1 (any metric below threshold).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from evaluator.faithfulness import FaithfulnessScorer
from evaluator.precision import PrecisionScorer
from evaluator.relevance import RelevanceScorer

REQUIRED_FIELDS = {"question", "context", "answer"}


def _load_jsonl(path: Path) -> list[dict]:
    """Load and validate a JSONL file.

    Raises:
        click.ClickException: on file-not-found, invalid JSON, or missing fields.
    """
    if not path.exists():
        raise click.ClickException(f"File not found: {path}")

    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise click.ClickException(
                    f"Invalid JSON on line {lineno}: {exc}"
                ) from exc

            missing = REQUIRED_FIELDS - set(record.keys())
            if missing:
                raise click.ClickException(
                    f"Line {lineno} is missing required fields: {sorted(missing)}"
                )

            # Validate field values are non-empty strings
            for field in REQUIRED_FIELDS:
                if not isinstance(record[field], str) or not record[field].strip():
                    raise click.ClickException(
                        f"Line {lineno}: field '{field}' must be a non-empty string"
                    )

            samples.append(record)

    if not samples:
        raise click.ClickException("Input file contains no valid samples.")

    return samples


@click.group()
def cli() -> None:
    """mlops-eval: LLM evaluation quality gate CLI."""


@cli.command()
@click.argument("input_file", type=click.Path(path_type=Path))
@click.option(
    "--threshold",
    default=0.7,
    type=float,
    show_default=True,
    help="Minimum mean score required to PASS each metric.",
)
def score(input_file: Path, threshold: float) -> None:
    """Score an eval set and gate on quality thresholds.

    INPUT_FILE is a JSONL file where each line contains:
    {"question": "...", "context": "...", "answer": "..."}

    Exits with code 0 if all mean scores >= threshold, 1 otherwise.
    """
    samples = _load_jsonl(input_file)

    click.echo("Loading scorers...")
    faithful_scorer = FaithfulnessScorer()
    relevant_scorer = RelevanceScorer()
    precision_scorer = PrecisionScorer()
    click.echo(f"Scoring {len(samples)} samples...\n")

    faithful_scores: list[float] = []
    relevant_scores: list[float] = []
    precision_scores: list[float] = []

    for i, sample in enumerate(samples, start=1):
        q = sample["question"]
        c = sample["context"]
        a = sample["answer"]

        f_score = faithful_scorer.score(context=c, answer=a)
        r_score = relevant_scorer.score(question=q, answer=a)
        p_score = precision_scorer.score(question=q, context=c, answer=a)

        faithful_scores.append(f_score)
        relevant_scores.append(r_score)
        precision_scores.append(p_score)

        click.echo(
            f"Sample {i}: "
            f"faithful={f_score:.2f}  "
            f"relevant={r_score:.2f}  "
            f"precision={p_score:.2f}"
        )

    mean_f = sum(faithful_scores) / len(faithful_scores)
    mean_r = sum(relevant_scores) / len(relevant_scores)
    mean_p = sum(precision_scores) / len(precision_scores)

    pass_f = mean_f >= threshold
    pass_r = mean_r >= threshold
    pass_p = mean_p >= threshold
    overall_pass = pass_f and pass_r and pass_p

    click.echo(
        f"\nMEAN:      "
        f"faithful={mean_f:.2f}  "
        f"relevant={mean_r:.2f}  "
        f"precision={mean_p:.2f}"
    )
    click.echo(
        f"PASS/FAIL: "
        f"faithful={'PASS' if pass_f else 'FAIL'}  "
        f"relevant={'PASS' if pass_r else 'FAIL'}  "
        f"precision={'PASS' if pass_p else 'FAIL'}"
    )
    click.echo(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    if not overall_pass:
        sys.exit(1)
