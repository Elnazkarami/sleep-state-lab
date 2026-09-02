# Data audit: SC400-n1

One real recording, read from the file named below. Every number on this page was
computed from it by `sleepstatelab audit-report`; nothing is quoted from elsewhere.

## Files

| what | value |
| --- | --- |
| PSG | `SC4001E0-PSG.edf` |
| hypnogram | `SC4001EC-Hypnogram.edf` |
| PSG SHA-256 | `2b40a18adf76af69a42d6db1f30f31d26b369f6d27ca0050ef30147ef892b131` |
| hypnogram SHA-256 | `a4cf67694ade1b52a0ddd06d5817fd45d2d3e8bac5302f640f3e9cfbbf12a996` |
| published checksum | match |
| PSG start | 1989-04-24T16:13:00 |
| hypnogram start | 1989-04-24T16:13:00 |
| PSG duration | 22.08 h |
| EDF+ | False |

The PSG and the hypnogram start at the same wall-clock second. That is the check
that makes epoch *i* of the signal and epoch *i* of the scoring the same 30
seconds; without it, every label could be shifted and nothing downstream would
notice.

## Channels, as the header declares them

| channel | unit | Hz | physical range | prefiltering |
| --- | --- | ---: | --- | --- |
| EEG Fpz-Cz | uV | 100 | [-192, 192] | HP:0.5Hz LP:100Hz [enhanced cassette BW] |
| EEG Pz-Oz | uV | 100 | [-197, 196] | HP:0.5Hz LP:100Hz [enhanced cassette BW] |

Units are read from the header and converted to microvolts explicitly. The
declared physical range is also the amplifier's rail, and saturation is detected
against it rather than against a fixed threshold.

## Epochs

| what | value |
| --- | ---: |
| epochs in the signal | 2650 |
| epochs the hypnogram covers | 2880 |
| hypnogram epochs past the end of the signal | 230 |
| annotation coverage of the signal | 1.000 |
| epochs retained | 2650 |
| first/last retained epoch index | 0 / 2649 |
| discontinuities in the retained indices | 0 |

Retained epochs keep their original index. A discontinuity in that index is an
excluded epoch, not missing time, and no downstream code treats the two epochs
either side of one as neighbours.

## Retained stages

| stage | epochs | share |
| --- | ---: | ---: |
| Wake | 1997 | 75.4% |
| N1 | 58 | 2.2% |
| N2 | 250 | 9.4% |
| N3 | 220 | 8.3% |
| REM | 125 | 4.7% |

This is the primary preparation: every valid scored epoch, with no
sleep-stage-based cropping. Wake dominates because a Sleep Cassette recorder ran
for roughly twenty hours around one night's sleep. The optional sleep-window
sensitivity analysis changes this distribution substantially and is reported
separately wherever it is used.

## Exclusions

| reason | epochs |
| --- | ---: |
| beyond_signal | 230 |
| movement_time | 0 |
| no_annotation | 0 |
| outside_sleep_window | 0 |
| qc_clipped | 0 |
| qc_flatline | 0 |
| qc_high_amplitude | 0 |
| short_epoch | 0 |
| unknown_annotation | 0 |
| unscored | 0 |

## Quality control on the retained epochs

| code | epochs flagged |
| --- | ---: |
| qc_flatline | 0 |
| qc_clipped | 0 |
| qc_high_amplitude | 0 |
| qc_muscle | 0 |

95th-percentile absolute amplitude of the example epoch shown for each stage,
in microvolts: {"Wake": 47.8, "N1": 23.6, "N2": 30.6, "N3": 66.4, "REM": 19.4}.

## Figure

![audit figure](SC400-n1_audit.png)

Top row: raw waveforms for one epoch of each stage, in microvolts. Middle row:
their power spectra, log scale, to 32 Hz. Bottom: the retained hypnogram against
time.
