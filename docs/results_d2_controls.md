# D2 and D3 at compute matched to D1, with both context controls

Generated from `predictions_matched.csv` (15414 saved prediction rows), split part `test`.

Primary metric: **macro-F1 computed per participant, then averaged equally across people**. Pooled macro-F1 is over all epochs at once and is given beside it, never instead of it.

Absent-class rule: Within a participant, a stage with no true and no predicted epochs is omitted from that participant's macro average; a stage present in either the truth or the predictions scores F1 = 0 when it is missed.

## Headline

| model | participant macro-F1 | pooled macro-F1 | balanced acc. | Cohen's kappa | accuracy | epochs | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D3 | 0.675 ± 0.000 | 0.675 | 0.703 | 0.740 | 0.847 | 2569 | 1 |
| D3-shuffled-context | 0.673 ± 0.000 | 0.673 | 0.701 | 0.738 | 0.846 | 2569 | 1 |
| D3-context-masked | 0.593 ± 0.000 | 0.593 | 0.669 | 0.707 | 0.830 | 2569 | 1 |
| D2 | 0.576 ± 0.000 | 0.576 | 0.631 | 0.614 | 0.767 | 2569 | 1 |
| D2-shuffled-context | 0.571 ± 0.000 | 0.571 | 0.625 | 0.612 | 0.765 | 2569 | 1 |
| D2-context-masked | 0.481 ± 0.000 | 0.481 | 0.622 | 0.594 | 0.760 | 2569 | 1 |

## Per participant (primary metric)

| participant | D2 | D2-context-masked | D2-shuffled-context | D3 | D3-context-masked | D3-shuffled-context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SC404 | 0.576 | 0.481 | 0.571 | 0.675 | 0.593 | 0.673 |

## D3

Run `pilot6-d3`, split `4b13bdea21d4fdad`, seed 0. Quality-control coverage 1.000 (0 flagged epochs of 2569).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.987 | 0.963 | 0.975 | 0.975 | 1534 | 1 |
| N1 | 0.339 | 0.494 | 0.402 | 0.402 | 166 | 1 |
| N2 | 0.910 | 0.766 | 0.832 | 0.832 | 620 | 1 |
| N3 | 0.656 | 0.792 | 0.718 | 0.718 | 53 | 1 |
| REM | 0.403 | 0.500 | 0.446 | 0.446 | 196 | 1 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 1478 | 36 | 3 | 0 | 17 |
| N1 | 20 | 82 | 10 | 0 | 54 |
| N2 | 0 | 49 | 475 | 22 | 74 |
| N3 | 0 | 0 | 11 | 42 | 0 |
| REM | 0 | 75 | 23 | 0 | 98 |

## D3-shuffled-context

Run `pilot6-d3`, split `4b13bdea21d4fdad`, seed 0. Quality-control coverage 1.000 (0 flagged epochs of 2569).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.986 | 0.963 | 0.975 | 0.975 | 1534 | 1 |
| N1 | 0.331 | 0.482 | 0.392 | 0.392 | 166 | 1 |
| N2 | 0.906 | 0.766 | 0.830 | 0.830 | 620 | 1 |
| N3 | 0.656 | 0.792 | 0.718 | 0.718 | 53 | 1 |
| REM | 0.408 | 0.500 | 0.450 | 0.450 | 196 | 1 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 1478 | 37 | 3 | 0 | 16 |
| N1 | 21 | 80 | 11 | 0 | 54 |
| N2 | 0 | 51 | 475 | 22 | 72 |
| N3 | 0 | 0 | 11 | 42 | 0 |
| REM | 0 | 74 | 24 | 0 | 98 |

## D3-context-masked

Run `pilot6-d3`, split `4b13bdea21d4fdad`, seed 0. Quality-control coverage 1.000 (0 flagged epochs of 2569).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.969 | 0.972 | 0.970 | 0.970 | 1534 | 1 |
| N1 | 0.333 | 0.036 | 0.065 | 0.065 | 166 | 1 |
| N2 | 0.857 | 0.724 | 0.785 | 0.785 | 620 | 1 |
| N3 | 0.539 | 0.906 | 0.676 | 0.676 | 53 | 1 |
| REM | 0.348 | 0.709 | 0.467 | 0.467 | 196 | 1 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 1491 | 7 | 8 | 1 | 27 |
| N1 | 36 | 6 | 21 | 0 | 103 |
| N2 | 1 | 1 | 449 | 40 | 129 |
| N3 | 0 | 0 | 4 | 48 | 1 |
| REM | 11 | 4 | 42 | 0 | 139 |

## D2

Run `pilot6-d2-matched`, split `4b13bdea21d4fdad`, seed 0. Quality-control coverage 1.000 (0 flagged epochs of 2569).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.980 | 0.954 | 0.967 | 0.967 | 1534 | 1 |
| N1 | 0.250 | 0.211 | 0.229 | 0.229 | 166 | 1 |
| N2 | 0.944 | 0.434 | 0.594 | 0.594 | 620 | 1 |
| N3 | 0.633 | 0.717 | 0.673 | 0.673 | 53 | 1 |
| REM | 0.278 | 0.837 | 0.417 | 0.417 | 196 | 1 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 1464 | 32 | 1 | 0 | 37 |
| N1 | 29 | 35 | 2 | 0 | 100 |
| N2 | 0 | 42 | 269 | 22 | 287 |
| N3 | 0 | 0 | 13 | 38 | 2 |
| REM | 1 | 31 | 0 | 0 | 164 |

## D2-shuffled-context

Run `pilot6-d2-matched`, split `4b13bdea21d4fdad`, seed 0. Quality-control coverage 1.000 (0 flagged epochs of 2569).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.980 | 0.954 | 0.967 | 0.967 | 1534 | 1 |
| N1 | 0.240 | 0.211 | 0.224 | 0.224 | 166 | 1 |
| N2 | 0.941 | 0.434 | 0.594 | 0.594 | 620 | 1 |
| N3 | 0.617 | 0.698 | 0.655 | 0.655 | 53 | 1 |
| REM | 0.277 | 0.827 | 0.415 | 0.415 | 196 | 1 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 1463 | 33 | 1 | 0 | 37 |
| N1 | 29 | 35 | 2 | 0 | 100 |
| N2 | 0 | 45 | 269 | 23 | 283 |
| N3 | 0 | 0 | 14 | 37 | 2 |
| REM | 1 | 33 | 0 | 0 | 162 |

## D2-context-masked

Run `pilot6-d2-matched`, split `4b13bdea21d4fdad`, seed 0. Quality-control coverage 1.000 (0 flagged epochs of 2569).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.953 | 0.971 | 0.962 | 0.962 | 1534 | 1 |
| N1 | 0.000 | 0.000 | 0.000 | 0.000 | 166 | 1 |
| N2 | 0.913 | 0.406 | 0.562 | 0.562 | 620 | 1 |
| N3 | 0.310 | 0.906 | 0.462 | 0.462 | 53 | 1 |
| REM | 0.282 | 0.827 | 0.421 | 0.421 | 196 | 1 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 1490 | 0 | 5 | 0 | 39 |
| N1 | 44 | 0 | 7 | 1 | 114 |
| N2 | 5 | 0 | 252 | 106 | 257 |
| N3 | 0 | 0 | 3 | 48 | 2 |
| REM | 25 | 0 | 9 | 0 | 162 |
