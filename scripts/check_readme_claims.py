"""Fail if the README claims something was not run that the evidence says was.

Written after the same defect was found twice by a reviewer. Result sections
get updated when a run lands; the prose above and below them does not, and a
README whose status contradicts its own results is worse than one with no
status at all -- it teaches the reader not to trust either.

This checks the cheap, mechanical half: for each model named in the committed
metrics, the README must not also assert that it has not been run. It cannot
check prose for sense, and is not a substitute for reading it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Phrases that assert something did not happen, paired with the evidence that
# would contradict them.
CLAIMS = {
    "the label-budget benchmark": (
        ("label-budget comparison — the thing this repository is\n  built to measure — has not been run",
         "label-budget comparison has not been run",
         "no benchmark results"),
        "docs/benchmark_metrics.json",
    ),
}


def models_with_results() -> set[str]:
    found: set[str] = set()
    for name in ("cohort_metrics.json", "benchmark_metrics.json", "pilot_metrics.json"):
        path = ROOT / "docs" / name
        if path.exists():
            found |= {entry["model"] for entry in json.loads(path.read_text())}
    return found


def main() -> int:
    readme = (ROOT / "README.md").read_text()
    problems: list[str] = []

    have = models_with_results()
    # A model with saved metrics must not be described as unrun.
    for model in sorted(have):
        pattern = re.compile(
            rf"{re.escape(model)}[^.\n]{{0,80}}(has not been run|not yet run|is not run)",
            re.I,
        )
        for match in pattern.finditer(readme):
            problems.append(f"{model}: README says {match.group(0)!r} but metrics exist")

    for phrase in ("no benchmark result exists", "no benchmark results"):
        if phrase in readme.lower() and have:
            problems.append(
                f"README says {phrase!r} but docs/ holds metrics for {len(have)} models"
            )

    if problems:
        print("README contradicts the committed evidence:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"README is consistent with metrics for {len(have)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
