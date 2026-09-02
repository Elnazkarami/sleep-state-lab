"""Configuration, device selection and the record kept with every artefact."""

from __future__ import annotations

import pytest

from sleepstatelab.config import Config, from_dict, load
from sleepstatelab.devices import probe, resolve
from sleepstatelab.provenance import digest, make_run_provenance

pytestmark = pytest.mark.synthetic


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="lerning_rate"):
        from_dict({"train": {"lerning_rate": 0.1}})
    with pytest.raises(ValueError, match="unknown top-level"):
        from_dict({"trian": {}})


def test_identity_changes_with_the_settings():
    first = Config()
    second = from_dict({"train": {"learning_rate": 0.002}})
    assert first.identity != second.identity


def test_preprocessing_identity_ignores_training_settings():
    """The cache is keyed by preprocessing, so a changed learning rate must not
    invalidate it -- and a changed filter must."""
    base = Config()
    other_lr = from_dict({"train": {"learning_rate": 0.002}})
    other_filter = from_dict({"preprocess": {"lowpass_hz": 20.0}})
    assert base.preprocessing_identity == other_lr.preprocessing_identity
    assert base.preprocessing_identity != other_filter.preprocessing_identity


def test_samples_per_epoch_is_derived():
    assert Config().samples_per_epoch == 3000


def test_shipped_configs_parse():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted(root.glob("*.yaml")):
        assert load(path).samples_per_epoch == 3000


def test_cpu_is_always_available():
    assert resolve("cpu") == "cpu"
    assert probe().cpu is True


def test_an_absent_device_raises_rather_than_falling_back():
    report = probe()
    if not report.cuda:
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve("cuda")
    if not report.mps:
        with pytest.raises(RuntimeError, match="MPS"):
            resolve("mps")


def test_auto_resolves_to_something_real():
    assert resolve("auto") in {"cpu", "cuda", "mps"}


def test_provenance_records_the_contract():
    record = make_run_provenance(
        run_id="r",
        device="cpu",
        seed=3,
        config=Config().to_dict(),
        split_id="abc",
        channels=("EEG Fpz-Cz", "EEG Pz-Oz"),
        label_order=("Wake", "N1", "N2", "N3", "REM"),
        preprocessing_id="pre",
    )
    assert record.config_id == digest(Config().to_dict())
    assert record.channels == ("EEG Fpz-Cz", "EEG Pz-Oz")
    assert record.split_id == "abc"
    assert record.code_revision
