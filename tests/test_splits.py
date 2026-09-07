"""Participants never cross a split, and a split is reproducible from its seed."""

from __future__ import annotations

import pytest

from sleepstatelab.data.splits import Split, SplitError, grouped_split, label_budget_subsets

pytestmark = pytest.mark.synthetic

PEOPLE = [f"SC4{i:02d}" for i in range(20)]


def test_no_participant_appears_twice():
    split = grouped_split(PEOPLE, seed=1)
    assert not set(split.train) & set(split.val)
    assert not set(split.train) & set(split.test)
    assert not set(split.val) & set(split.test)
    assert set(split.participants) == set(PEOPLE)


def test_a_split_that_shares_a_participant_cannot_be_constructed():
    with pytest.raises(SplitError):
        Split(
            name="bad",
            seed=0,
            strategy="manual",
            train=("SC400", "SC401"),
            val=("SC401",),
            test=("SC402",),
        )


def test_the_same_seed_gives_the_same_split():
    assert grouped_split(PEOPLE, seed=5).identity == grouped_split(PEOPLE, seed=5).identity
    assert grouped_split(PEOPLE, seed=5).identity != grouped_split(PEOPLE, seed=6).identity


def test_identity_is_of_the_people_not_the_name():
    first = grouped_split(PEOPLE, seed=5, name="a")
    second = grouped_split(PEOPLE, seed=5, name="b")
    assert first.identity == second.identity


def test_three_participants_is_the_floor():
    with pytest.raises(SplitError):
        grouped_split(["SC400", "SC401"], seed=0)


def test_label_budgets_are_nested_and_touch_only_training():
    split = grouped_split(PEOPLE, seed=2)
    budgets = label_budget_subsets(split, (0.1, 0.25, 1.0), seed=2)
    assert set(budgets[0.1]) <= set(budgets[0.25]) <= set(budgets[1.0])
    assert set(budgets[1.0]) == set(split.train)
    for people in budgets.values():
        assert not set(people) & set(split.val)
        assert not set(people) & set(split.test)


def test_written_split_detects_editing(tmp_path):
    split = grouped_split(PEOPLE, seed=3)
    path = tmp_path / "split.json"
    split.write(path)
    assert Split.read(path).identity == split.identity

    text = path.read_text().replace(split.test[0], split.train[0], 1)
    path.write_text(text)
    with pytest.raises(SplitError):
        Split.read(path)


def test_a_split_records_the_recordings_it_was_built_on():
    split = grouped_split(
        ["SC400", "SC401", "SC402"],
        seed=0,
        recordings=["SC400-n1", "SC400-n2", "SC401-n1", "SC402-n1"],
    )
    assert split.recordings == ("SC400-n1", "SC400-n2", "SC401-n1", "SC402-n1")


def test_a_cache_that_gained_a_recording_is_refused():
    """The failure this exists for: the epoch cache is keyed by preprocessing,
    so preparing more recordings later adds them to the same directory. A split
    naming SC404 then covers two nights where it covered one, and a result
    computed after is not comparable with one computed before."""
    split = grouped_split(
        ["SC400", "SC401", "SC402"],
        seed=0,
        recordings=["SC400-n1", "SC401-n1", "SC402-n1"],
    )
    part = split.part_of("SC400")
    split.check_recordings(part, ("SC400-n1",))
    with pytest.raises(SplitError, match="no longer matches"):
        split.check_recordings(part, ("SC400-n1", "SC400-n2"))


def test_a_split_without_recorded_recordings_skips_the_check():
    """Splits written before this was recorded stay usable, unchecked rather
    than guessed at."""
    split = grouped_split(["SC400", "SC401", "SC402"], seed=0)
    assert split.recordings == ()
    split.check_recordings("test", ("anything-n9",))


def test_recordings_do_not_change_the_split_identity():
    """Identity answers "same people, same sides", which is what a checkpoint
    needs. Adding a field to the file must not invalidate existing checkpoints."""
    plain = grouped_split(["SC400", "SC401", "SC402"], seed=0)
    with_recordings = grouped_split(
        ["SC400", "SC401", "SC402"], seed=0, recordings=["SC400-n1"]
    )
    assert plain.identity == with_recordings.identity


def test_recordings_survive_a_round_trip(tmp_path):
    split = grouped_split(
        ["SC400", "SC401", "SC402"], seed=0, recordings=["SC400-n1", "SC401-n1"]
    )
    path = tmp_path / "split.json"
    split.write(path)
    assert Split.read(path).recordings == split.recordings
