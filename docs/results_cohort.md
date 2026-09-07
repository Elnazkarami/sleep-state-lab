# Cohort: 78 participants, 15 held out

Generated from `predictions_cohort.csv` (162710 saved prediction rows), split part `test`.

Primary metric: **macro-F1 computed per participant, then averaged equally across people**. Pooled macro-F1 is over all epochs at once and is given beside it, never instead of it.

Absent-class rule: Within a participant, a stage with no true and no predicted epochs is omitted from that participant's macro average; a stage present in either the truth or the predictions scores F1 = 0 when it is missed.

## Headline

| model | participant macro-F1 | pooled macro-F1 | balanced acc. | Cohen's kappa | accuracy | epochs | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | 0.673 ± 0.097 | 0.715 | 0.757 | 0.750 | 0.874 | 81355 | 15 |
| D1+smoothing | 0.655 ± 0.104 | 0.700 | 0.772 | 0.710 | 0.851 | 81355 | 15 |

## Per participant (primary metric)

| participant | D1 | D1+smoothing |
| --- | ---: | ---: |
| SC400 | 0.775 | 0.787 |
| SC403 | 0.805 | 0.817 |
| SC407 | 0.831 | 0.829 |
| SC425 | 0.565 | 0.566 |
| SC428 | 0.721 | 0.705 |
| SC429 | 0.780 | 0.745 |
| SC430 | 0.585 | 0.539 |
| SC431 | 0.733 | 0.702 |
| SC434 | 0.591 | 0.535 |
| SC443 | 0.718 | 0.636 |
| SC455 | 0.664 | 0.632 |
| SC459 | 0.659 | 0.692 |
| SC461 | 0.552 | 0.516 |
| SC473 | 0.524 | 0.580 |
| SC476 | 0.597 | 0.545 |

## D1

Run `cohort-d1`, split `2a0313adb97b9da0`, seed 0. Quality-control coverage 1.000 (21 flagged epochs of 81355).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.995 | 0.948 | 0.971 | 0.969 | 56983 | 15 |
| N1 | 0.271 | 0.620 | 0.377 | 0.378 | 3556 | 15 |
| N2 | 0.865 | 0.690 | 0.768 | 0.758 | 12477 | 15 |
| N3 | 0.703 | 0.803 | 0.749 | 0.553 | 2892 | 15 |
| REM | 0.694 | 0.726 | 0.710 | 0.707 | 5447 | 15 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 54008 | 2585 | 42 | 23 | 325 |
| N1 | 183 | 2203 | 404 | 34 | 732 |
| N2 | 44 | 2316 | 8607 | 826 | 684 |
| N3 | 0 | 48 | 520 | 2322 | 2 |
| REM | 28 | 990 | 373 | 100 | 3956 |

## D1+smoothing

Run `cohort-d1`, split `2a0313adb97b9da0`, seed 0. Quality-control coverage 1.000 (21 flagged epochs of 81355).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.996 | 0.938 | 0.966 | 0.965 | 56983 | 15 |
| N1 | 0.240 | 0.769 | 0.366 | 0.366 | 3556 | 15 |
| N2 | 0.909 | 0.500 | 0.645 | 0.616 | 12477 | 15 |
| N3 | 0.658 | 0.853 | 0.743 | 0.553 | 2892 | 15 |
| REM | 0.765 | 0.797 | 0.780 | 0.774 | 5447 | 15 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 53474 | 3287 | 32 | 25 | 165 |
| N1 | 141 | 2734 | 133 | 35 | 513 |
| N2 | 42 | 4424 | 6240 | 1113 | 658 |
| N3 | 0 | 52 | 371 | 2468 | 1 |
| REM | 21 | 888 | 88 | 109 | 4341 |
