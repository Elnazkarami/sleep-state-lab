"""Report tables, generated from saved predictions and from nothing else.

Every table in the README that carries a number was produced by this module and
pasted whole. Nothing is typed in by hand, which is the only way to guarantee
that the documentation and the predictions agree.
"""

from __future__ import annotations

import json
from pathlib import Path

from sleepstatelab.evaluation.metrics import (
    ABSENT_CLASS_RULE,
    EvaluationResult,
    evaluate_predictions,
)
from sleepstatelab.labels import STAGES


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def headline_table(results: list[EvaluationResult]) -> str:
    """Primary metric first, pooled beside it, and the class distribution behind both."""
    lines = [
        _row(
            [
                "model",
                "participant macro-F1",
                "pooled macro-F1",
                "balanced acc.",
                "Cohen's kappa",
                "accuracy",
                "epochs",
                "participants",
            ]
        ),
        _row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for result in sorted(results, key=lambda r: -r.participant_macro_f1_mean):
        lines.append(
            _row(
                [
                    result.model,
                    f"{result.participant_macro_f1_mean:.3f} ± {result.participant_macro_f1_sd:.3f}",
                    f"{result.pooled_macro_f1:.3f}",
                    f"{result.balanced_accuracy:.3f}",
                    f"{result.cohens_kappa:.3f}",
                    f"{result.pooled_accuracy:.3f}",
                    str(result.n_epochs),
                    str(result.n_participants),
                ]
            )
        )
    return "\n".join(lines)


def per_stage_table(result: EvaluationResult) -> str:
    """Precision, recall and F1 per stage, with the support each rests on."""
    lines = [
        _row(["stage", "precision", "recall", "F1 (pooled)", "F1 (participant mean)", "support", "participants"]),
        _row(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for index, name in enumerate(STAGES):
        precision = result.per_stage_precision[name]
        recall = result.per_stage_recall[name]
        denominator = precision + recall
        pooled = 2 * precision * recall / denominator if denominator else 0.0
        mean_f1 = result.per_stage_f1[name]
        lines.append(
            _row(
                [
                    name,
                    f"{precision:.3f}",
                    f"{recall:.3f}",
                    f"{pooled:.3f}",
                    "n/a" if mean_f1 != mean_f1 else f"{mean_f1:.3f}",
                    str(result.per_stage_support[name]),
                    str(result.per_stage_participants[name]),
                ]
            )
        )
        _ = index
    return "\n".join(lines)


def confusion_table(result: EvaluationResult) -> str:
    """Rows are truth, columns are prediction."""
    lines = [
        _row(["true \\ predicted", *STAGES]),
        _row(["---", *["---:"] * len(STAGES)]),
    ]
    for name, row in zip(STAGES, result.confusion, strict=True):
        lines.append(_row([name, *[str(value) for value in row]]))
    return "\n".join(lines)


def per_participant_table(results: list[EvaluationResult]) -> str:
    """One column per model, one row per held-out participant."""
    people = sorted({p for r in results for p in r.participant_macro_f1})
    models = [r.model for r in results]
    lines = [
        _row(["participant", *models]),
        _row(["---", *["---:"] * len(models)]),
    ]
    for person in people:
        cells = []
        for result in results:
            value = result.participant_macro_f1.get(person)
            cells.append("n/a" if value is None else f"{value:.3f}")
        lines.append(_row([person, *cells]))
    return "\n".join(lines)


def build_report(
    predictions: Path | str,
    *,
    split_part: str = "test",
    title: str = "Results",
    provenance: dict[str, object] | None = None,
) -> str:
    """A whole markdown report for one prediction file."""
    results = evaluate_predictions(predictions, split_part=split_part)
    if not results:
        raise ValueError(f"{predictions} holds no rows for split part {split_part!r}")

    sections = [
        f"# {title}",
        "",
        f"Generated from `{Path(predictions).name}` "
        f"({sum(r.n_epochs for r in results)} saved prediction rows), split part "
        f"`{split_part}`.",
        "",
        "Primary metric: **macro-F1 computed per participant, then averaged equally "
        "across people**. Pooled macro-F1 is over all epochs at once and is given "
        "beside it, never instead of it.",
        "",
        f"Absent-class rule: {ABSENT_CLASS_RULE}",
        "",
        "## Headline",
        "",
        headline_table(results),
        "",
        "## Per participant (primary metric)",
        "",
        per_participant_table(results),
        "",
    ]
    for result in sorted(results, key=lambda r: -r.participant_macro_f1_mean):
        sections += [
            f"## {result.model}",
            "",
            f"Run `{result.run_id}`, split `{result.split_id}`, seed {result.seed}. "
            f"Quality-control coverage {result.qc_coverage:.3f} "
            f"({result.qc_flagged_epochs} flagged epochs of {result.n_epochs}).",
            "",
            "### Per stage",
            "",
            per_stage_table(result),
            "",
            "### Confusion matrix",
            "",
            confusion_table(result),
            "",
        ]
    if provenance:
        sections += [
            "## Provenance",
            "",
            "```json",
            json.dumps(provenance, indent=2, default=str),
            "```",
            "",
        ]
    return "\n".join(sections)
