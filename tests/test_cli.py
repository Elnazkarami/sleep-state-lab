"""The command line exists, parses, and dispatches to real functions.

This file exists because of a real failure: an editing script wrote the
trainer's source over `cli.py` and every test still passed, because nothing
imported the command line. The package was, for one commit, a library with no
usable entry point and a green suite.

None of these tests run a pipeline -- that is what the smoke run is for. They
assert the shape of the interface, which is exactly what silently disappeared.
"""

from __future__ import annotations

import argparse

import pytest

from sleepstatelab import cli

pytestmark = pytest.mark.synthetic

EXPECTED = {
    "doctor",
    "audit",
    "prepare",
    "audit-report",
    "split",
    "baselines",
    "train-d1",
    "train-d2",
    "pretrain",
    "predict",
    "report",
    "smoke",
}


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    # argparse exposes no public way to reach the subparsers; this is the
    # documented-by-everyone private route, and it is why this helper exists
    # once rather than in each test.
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("the parser has no subcommands")


def test_the_module_is_the_command_line():
    """A one-line guard against the file being replaced by something else."""
    assert cli.__doc__ is not None
    assert "command line" in cli.__doc__.splitlines()[0].lower()
    assert callable(cli.main)
    assert callable(cli.build_parser)


def test_every_subcommand_is_present():
    found = set(_subparsers(cli.build_parser()))
    assert EXPECTED <= found, f"missing subcommand(s): {sorted(EXPECTED - found)}"


@pytest.mark.parametrize("command", sorted(EXPECTED))
def test_every_subcommand_dispatches_to_a_function(command: str):
    """`func` must be set and must be a function this module actually defines."""
    parser = _subparsers(cli.build_parser())[command]
    func = parser.get_default("func")
    assert callable(func), f"{command} dispatches to {func!r}"
    assert getattr(cli, func.__name__, None) is func


@pytest.mark.parametrize("command", sorted(EXPECTED))
def test_every_subcommand_has_help(command: str, capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exit_code:
        cli.main([command, "--help"])
    assert exit_code.value.code == 0
    assert capsys.readouterr().out.strip()


def test_the_training_commands_take_the_arguments_the_readme_uses():
    """The README's commands are the interface; a rename would break them."""
    parser = cli.build_parser()
    parsed = parser.parse_args(
        [
            "train-d2",
            "--config",
            "configs/pilot.yaml",
            "--split",
            "outputs/split.json",
            "--device",
            "cpu",
            "--epochs",
            "20",
            "--checkpoint",
            "runs/d2/checkpoint.pt",
        ]
    )
    assert parsed.func is cli.cmd_train_d2
    assert parsed.epochs == 20
    assert parsed.no_segments is False

    control = parser.parse_args(
        [
            "predict",
            "--split",
            "outputs/split.json",
            "--checkpoint",
            "runs/d2/checkpoint.pt",
            "--model-name",
            "D2-shuffled-context",
            "--shuffle-context",
        ]
    )
    assert control.func is cli.cmd_predict
    assert control.shuffle_context is True


def test_an_unknown_subcommand_is_refused():
    with pytest.raises(SystemExit):
        cli.main(["not-a-command"])
