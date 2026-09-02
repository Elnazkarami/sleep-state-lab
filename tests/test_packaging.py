"""The repository contains the package it claims to contain.

This file exists because of a real failure: a `.gitignore` line reading `data/`
also matched `src/sleepstatelab/data/`, so ten core modules were never committed
and never linted -- the pushed package could not import, and the lint that would
have said so was skipping the same directory for the same reason.

Two properties are asserted here, both cheap and both about the repository
rather than about sleep.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.synthetic

ROOT = Path(__file__).resolve().parents[1]


def _ignored(paths: list[Path]) -> list[str]:
    """Which of these paths git would refuse to track.

    `check-ignore` rather than `ls-files`, deliberately: the invariant is that no
    source file is *ignored*. A file that is merely not yet staged is a file
    someone is still writing, and failing on that would make the test noise.
    """
    found = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=ROOT,
        input="\n".join(str(p) for p in paths),
        capture_output=True,
        text=True,
        check=False,
    )
    if found.returncode not in (0, 1):
        pytest.skip("not a git checkout")
    return sorted(line for line in found.stdout.splitlines() if line)


def test_no_source_file_is_git_ignored():
    """A source file the repository ignores is a source file nobody else gets."""
    on_disk = [
        path.relative_to(ROOT)
        for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
        if "__pycache__" not in path.parts and ".egg-info" not in str(path)
    ]
    assert on_disk, "found no source files at all, which cannot be right"
    ignored = _ignored(on_disk)
    assert not ignored, (
        f"these source files are git-ignored: {ignored}. "
        "Check .gitignore -- a bare directory name matches at every depth, so "
        "`data/` also matches src/sleepstatelab/data."
    )


def test_every_subpackage_is_importable():
    """Catches a missing __init__.py, which a source distribution would drop."""
    import importlib

    for name in (
        "sleepstatelab",
        "sleepstatelab.data",
        "sleepstatelab.features",
        "sleepstatelab.models",
        "sleepstatelab.baselines",
        "sleepstatelab.training",
        "sleepstatelab.evaluation",
    ):
        assert importlib.import_module(name) is not None
