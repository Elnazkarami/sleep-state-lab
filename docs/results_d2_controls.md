# D2 at compute matched to D1, with both context controls

Generated from `predictions_matched.csv` (7707 saved prediction rows), split part `test`.

Primary metric: **macro-F1 computed per participant, then averaged equally across people**. Pooled macro-F1 is over all epochs at once and is given beside it, never instead of it.

Absent-class rule: Within a participant, a stage with no true and no predicted epochs is omitted from that participant's macro average; a stage present in either the truth or the predictions scores F1 = 0 when it is missed.

## Headline

| model | participant macro-F1 | pooled macro-F1 | balanced acc. | Cohen's kappa | accuracy | epochs | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D2 | 0.576 ± 0.000 | 0.576 | 0.631 | 0.614 | 0.767 | 2569 | 1 |
| D2-shuffled-context | 0.571 ± 0.000 | 0.571 | 0.625 | 0.612 | 0.765 | 2569 | 1 |
| D2-context-masked | 0.481 ± 0.000 | 0.481 | 0.622 | 0.594 | 0.760 | 2569 | 1 |

## Per participant (primary metric)

| participant | D2 | D2-context-masked | D2-shuffled-context |
| --- | ---: | ---: | ---: |
| SC404 | 0.576 | 0.481 | 0.571 |

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
