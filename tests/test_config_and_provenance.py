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


def test_a_device_can_be_checked_by_using_it():
    """CPU must pass a real operation, not merely claim to be available."""
    from sleepstatelab.devices import device_works

    assert device_works("cpu") is True


def test_an_unusable_device_reports_itself_as_such():
    """`torch.backends.mps.is_available()` returned True on a machine whose
    Metal compiler had become unreachable, and the first convolution aborted the
    process -- taking a cohort training run with it. Availability is a claim; a
    test operation is evidence."""
    from sleepstatelab.devices import probe

    report = probe(check=True)
    assert report.working["cpu"] is True
    for name in ("cuda", "mps"):
        if name in report.working:
            assert isinstance(report.working[name], bool)
    assert "test operation" in report.summary()


def test_resolve_refuses_a_device_that_fails_its_trial(monkeypatch):
    import sleepstatelab.devices as devices

    broken = devices.DeviceReport(
        torch_version="x",
        cpu=True,
        cuda=False,
        cuda_devices=(),
        mps=True,
        mps_built=True,
        working={"cpu": True, "mps": False},
    )
    monkeypatch.setattr(devices, "probe", lambda check=False: broken)
    with pytest.raises(RuntimeError, match="test operation"):
        devices.resolve("mps", check=True)
    # `auto` steps over a broken accelerator rather than refusing outright:
    # nothing was asked for by name, so nothing is being silently substituted.
    assert devices.resolve("auto", check=True) == "cpu"
