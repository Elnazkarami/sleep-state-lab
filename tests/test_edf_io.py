"""The file is read as the file says it should be read.

These are the tests that stop an entire experiment from being quietly wrong by
a factor of a thousand, or by one epoch.
"""

from __future__ import annotations

import numpy as np
import pytest

from sleepstatelab.data.discovery import discover
from sleepstatelab.data.edf_header import EDFHeaderError, read_header, to_microvolt_factor
from sleepstatelab.data.epochs import epoch_recording, read_channels_microvolts
from sleepstatelab.synthetic import write_edf

pytestmark = pytest.mark.synthetic


def test_header_reports_what_the_file_declares(synthetic_night):
    header = read_header(synthetic_night.psg_path)
    assert header.labels == ("EEG Fpz-Cz", "EEG Pz-Oz")
    assert header.sampling_rate("EEG Fpz-Cz") == 100.0
    assert header.duration_seconds == 24 * 30.0
    assert header.start is not None and header.start.year == 1985


def test_two_digit_year_uses_the_edf_convention(tmp_path):
    """85-99 mean the 1900s. Sleep-EDF was recorded in 1989; the wrong rule puts
    it in 2089 and the PSG/hypnogram alignment check then compares nonsense."""
    path = write_edf(
        tmp_path / "SC4991E0-PSG.edf",
        {"EEG Fpz-Cz": np.zeros(200)},
        100.0,
        start="24.04.89",
        start_time="16.13.00",
    )
    assert read_header(path).start.year == 1989


def test_microvolt_conversion_is_by_declared_unit(tmp_path):
    """A file declaring millivolts must come back a thousand times larger."""
    samples = 100 * 30
    # 5 Hz at 100 Hz puts a sample exactly on the peak, so the expected maximum
    # is the amplitude itself rather than the amplitude times whatever the
    # nearest sample happened to be.
    wave = 50.0 * np.sin(2 * np.pi * 5 * np.arange(samples) / 100.0)

    micro = write_edf(
        tmp_path / "SC4011E0-PSG.edf", {"EEG Fpz-Cz": wave}, 100.0, unit="uV"
    )
    milli = write_edf(
        tmp_path / "SC4021E0-PSG.edf",
        {"EEG Fpz-Cz": wave},
        100.0,
        unit="mV",
        physical_range=200.0,
    )
    in_uv, _ = read_channels_microvolts(micro, ("EEG Fpz-Cz",), expected_rate_hz=100.0)
    in_mv, _ = read_channels_microvolts(milli, ("EEG Fpz-Cz",), expected_rate_hz=100.0)

    assert np.abs(in_uv).max() == pytest.approx(50.0, rel=0.01)
    assert np.abs(in_mv).max() == pytest.approx(50_000.0, rel=0.01)
    # The ratio is the claim: the same numbers declared in millivolts are a
    # thousand times more microvolts.
    assert np.abs(in_mv).max() / np.abs(in_uv).max() == pytest.approx(1000.0, rel=1e-6)


def test_unknown_unit_raises_rather_than_guessing():
    with pytest.raises(EDFHeaderError):
        to_microvolt_factor("counts")


def test_wrong_sampling_rate_is_refused_not_resampled(tmp_path):
    path = write_edf(tmp_path / "SC4031E0-PSG.edf", {"EEG Fpz-Cz": np.zeros(6000)}, 200.0)
    with pytest.raises(ValueError, match="expected 100"):
        read_channels_microvolts(path, ("EEG Fpz-Cz",), expected_rate_hz=100.0)


def test_channel_order_is_the_order_asked_for(synthetic_night):
    forward, _ = read_channels_microvolts(
        synthetic_night.psg_path, ("EEG Fpz-Cz", "EEG Pz-Oz"), expected_rate_hz=100.0
    )
    reversed_order, _ = read_channels_microvolts(
        synthetic_night.psg_path, ("EEG Pz-Oz", "EEG Fpz-Cz"), expected_rate_hz=100.0
    )
    assert np.allclose(forward[0], reversed_order[1])
    assert np.allclose(forward[1], reversed_order[0])


def test_amplitudes_survive_the_round_trip(synthetic_night):
    """What was written in microvolts comes back in microvolts."""
    found = discover(synthetic_night.psg_path.parent)
    record = epoch_recording(found.pairs[0], channels=("EEG Fpz-Cz", "EEG Pz-Oz"))
    p95 = float(np.percentile(np.abs(record.signals), 95))
    assert 5.0 < p95 < 200.0, f"amplitudes of {p95} uV are not scalp EEG"
