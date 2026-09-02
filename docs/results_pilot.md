# Pilot: six participants, real Sleep-EDF Cassette recordings

Generated from `predictions_pilot.csv` (15414 saved prediction rows), split part `test`.

Primary metric: **macro-F1 computed per participant, then averaged equally across people**. Pooled macro-F1 is over all epochs at once and is given beside it, never instead of it.

Absent-class rule: Within a participant, a stage with no true and no predicted epochs is omitted from that participant's macro average; a stage present in either the truth or the predictions scores F1 = 0 when it is missed.

## Headline

| model | participant macro-F1 | pooled macro-F1 | balanced acc. | Cohen's kappa | accuracy | epochs | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random_forest | 0.717 ± 0.000 | 0.717 | 0.705 | 0.757 | 0.859 | 2569 | 1 |
| D1 | 0.702 ± 0.000 | 0.702 | 0.748 | 0.743 | 0.847 | 2569 | 1 |
| D2 | 0.644 ± 0.000 | 0.644 | 0.718 | 0.679 | 0.807 | 2569 | 1 |
| D2-shuffled-context | 0.644 ± 0.000 | 0.644 | 0.713 | 0.680 | 0.807 | 2569 | 1 |
| logistic | 0.626 ± 0.000 | 0.626 | 0.644 | 0.689 | 0.815 | 2569 | 1 |
| class_prior | 0.150 ± 0.000 | 0.150 | 0.200 | 0.000 | 0.597 | 2569 | 1 |

## Per participant (primary metric)

| participant | D1 | D2 | D2-shuffled-context | class_prior | logistic | random_forest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SC404 | 0.702 | 0.644 | 0.644 | 0.150 | 0.626 | 0.717 |

## random_forest

Run `pilot6-baselines`, split `4b13bdea21d4fdad`, seed 20260901. Quality-control coverage 1.000 (0 flagged epochs of 2569).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.961 | 0.965 | 0.963 | 0.963 | 1534 | 1 |
| N1 | 0.315 | 0.651 | 0.424 | 0.424 | 166 | 1 |
| N2 | 0.932 | 0.797 | 0.859 | 0.859 | 620 | 1 |
| N3 | 0.921 | 0.660 | 0.769 | 0.769 | 53 | 1 |
| REM | 0.761 | 0.454 | 0.569 | 0.569 | 196 | 1 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 1481 | 44 | 5 | 0 | 4 |
| N1 | 35 | 108 | 7 | 0 | 16 |
| N2 | 17 | 98 | 494 | 3 | 8 |
| N3 | 0 | 0 | 18 | 35 | 0 |
| REM | 8 | 93 | 6 | 0 | 89 |

## D1

Run `pilot6-d1`, split `4b13bdea21d4fdad`, seed 0. Quality-control coverage 1.000 (0 flagged epochs of 2569).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.989 | 0.962 | 0.975 | 0.975 | 1534 | 1 |
| N1 | 0.335 | 0.524 | 0.408 | 0.408 | 166 | 1 |
| N2 | 0.980 | 0.698 | 0.815 | 0.815 | 620 | 1 |
| N3 | 0.708 | 0.868 | 0.780 | 0.780 | 53 | 1 |
| REM | 0.434 | 0.689 | 0.533 | 0.533 | 196 | 1 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 1475 | 50 | 0 | 1 | 8 |
| N1 | 16 | 87 | 2 | 1 | 60 |
| N2 | 0 | 62 | 433 | 17 | 108 |
| N3 | 0 | 0 | 7 | 46 | 0 |
| REM | 0 | 61 | 0 | 0 | 135 |

## D2

Run `pilot6-d2`, split `4b13bdea21d4fdad`, seed 0. Quality-control coverage 1.000 (0 flagged epochs of 2569).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.986 | 0.950 | 0.968 | 0.968 | 1534 | 1 |
| N1 | 0.297 | 0.542 | 0.384 | 0.384 | 166 | 1 |
| N2 | 0.963 | 0.585 | 0.728 | 0.728 | 620 | 1 |
| N3 | 0.575 | 0.943 | 0.714 | 0.714 | 53 | 1 |
| REM | 0.344 | 0.566 | 0.428 | 0.428 | 196 | 1 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 1458 | 58 | 3 | 0 | 15 |
| N1 | 21 | 90 | 6 | 0 | 49 |
| N2 | 0 | 72 | 363 | 37 | 148 |
| N3 | 0 | 0 | 3 | 50 | 0 |
| REM | 0 | 83 | 2 | 0 | 111 |

## D2-shuffled-context

Run `pilot6-d2`, split `4b13bdea21d4fdad`, seed 0. Quality-control coverage 1.000 (0 flagged epochs of 2569).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.985 | 0.950 | 0.967 | 0.967 | 1534 | 1 |
| N1 | 0.279 | 0.530 | 0.366 | 0.366 | 166 | 1 |
| N2 | 0.961 | 0.600 | 0.739 | 0.739 | 620 | 1 |
| N3 | 0.581 | 0.943 | 0.719 | 0.719 | 53 | 1 |
| REM | 0.352 | 0.541 | 0.427 | 0.427 | 196 | 1 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 1458 | 57 | 3 | 0 | 16 |
| N1 | 22 | 88 | 7 | 0 | 49 |
| N2 | 0 | 82 | 372 | 36 | 130 |
| N3 | 0 | 0 | 3 | 50 | 0 |
| REM | 0 | 88 | 2 | 0 | 106 |

## logistic

Run `pilot6-baselines`, split `4b13bdea21d4fdad`, seed 20260901. Quality-control coverage 1.000 (0 flagged epochs of 2569).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.982 | 0.956 | 0.969 | 0.969 | 1534 | 1 |
| N1 | 0.251 | 0.831 | 0.385 | 0.385 | 166 | 1 |
| N2 | 0.944 | 0.682 | 0.792 | 0.792 | 620 | 1 |
| N3 | 0.857 | 0.566 | 0.682 | 0.682 | 53 | 1 |
| REM | 0.837 | 0.184 | 0.301 | 0.301 | 196 | 1 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 1466 | 63 | 3 | 2 | 0 |
| N1 | 23 | 138 | 1 | 0 | 4 |
| N2 | 3 | 188 | 423 | 3 | 3 |
| N3 | 0 | 3 | 20 | 30 | 0 |
| REM | 1 | 158 | 1 | 0 | 36 |

## class_prior

Run `pilot6-baselines`, split `4b13bdea21d4fdad`, seed 20260901. Quality-control coverage 1.000 (0 flagged epochs of 2569).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.597 | 1.000 | 0.748 | 0.748 | 1534 | 1 |
| N1 | 0.000 | 0.000 | 0.000 | 0.000 | 166 | 1 |
| N2 | 0.000 | 0.000 | 0.000 | 0.000 | 620 | 1 |
| N3 | 0.000 | 0.000 | 0.000 | 0.000 | 53 | 1 |
| REM | 0.000 | 0.000 | 0.000 | 0.000 | 196 | 1 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 1534 | 0 | 0 | 0 | 0 |
| N1 | 166 | 0 | 0 | 0 | 0 |
| N2 | 620 | 0 | 0 | 0 | 0 |
| N3 | 53 | 0 | 0 | 0 | 0 |
| REM | 196 | 0 | 0 | 0 | 0 |
