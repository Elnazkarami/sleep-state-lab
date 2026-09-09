# Cohort: 78 participants, 15 held out

Generated from `predictions_cohort.csv` (650840 saved prediction rows), split part `test`.

Primary metric: **macro-F1 computed per participant, then averaged equally across people**. Pooled macro-F1 is over all epochs at once and is given beside it, never instead of it.

Absent-class rule: Within a participant, a stage with no true and no predicted epochs is omitted from that participant's macro average; a stage present in either the truth or the predictions scores F1 = 0 when it is missed.

## Headline

| model | participant macro-F1 | pooled macro-F1 | balanced acc. | Cohen's kappa | accuracy | epochs | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D2 | 0.722 ± 0.094 | 0.766 | 0.791 | 0.812 | 0.907 | 81355 | 15 |
| D2-shuffled-context | 0.702 ± 0.093 | 0.747 | 0.767 | 0.797 | 0.900 | 81355 | 15 |
| D3 | 0.695 ± 0.094 | 0.743 | 0.785 | 0.774 | 0.887 | 81355 | 15 |
| D3-shuffled-context | 0.683 ± 0.093 | 0.731 | 0.770 | 0.765 | 0.883 | 81355 | 15 |
| D1 | 0.673 ± 0.097 | 0.715 | 0.757 | 0.750 | 0.874 | 81355 | 15 |
| D1+smoothing | 0.655 ± 0.104 | 0.700 | 0.772 | 0.710 | 0.851 | 81355 | 15 |
| D3-context-masked | 0.652 ± 0.102 | 0.691 | 0.720 | 0.742 | 0.871 | 81355 | 15 |
| D2-context-masked | 0.646 ± 0.106 | 0.686 | 0.696 | 0.760 | 0.883 | 81355 | 15 |

## Per participant (primary metric)

| participant | D1 | D1+smoothing | D2 | D2-context-masked | D2-shuffled-context | D3 | D3-context-masked | D3-shuffled-context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SC400 | 0.775 | 0.787 | 0.829 | 0.749 | 0.803 | 0.805 | 0.735 | 0.797 |
| SC403 | 0.805 | 0.817 | 0.824 | 0.762 | 0.797 | 0.789 | 0.761 | 0.767 |
| SC407 | 0.831 | 0.829 | 0.868 | 0.821 | 0.849 | 0.858 | 0.833 | 0.848 |
| SC425 | 0.565 | 0.566 | 0.659 | 0.572 | 0.640 | 0.647 | 0.626 | 0.633 |
| SC428 | 0.721 | 0.705 | 0.796 | 0.751 | 0.788 | 0.761 | 0.735 | 0.758 |
| SC429 | 0.780 | 0.745 | 0.823 | 0.750 | 0.797 | 0.796 | 0.774 | 0.773 |
| SC430 | 0.585 | 0.539 | 0.613 | 0.531 | 0.584 | 0.575 | 0.533 | 0.557 |
| SC431 | 0.733 | 0.702 | 0.776 | 0.700 | 0.748 | 0.735 | 0.700 | 0.733 |
| SC434 | 0.591 | 0.535 | 0.667 | 0.591 | 0.647 | 0.655 | 0.606 | 0.647 |
| SC443 | 0.718 | 0.636 | 0.777 | 0.700 | 0.769 | 0.732 | 0.696 | 0.715 |
| SC455 | 0.664 | 0.632 | 0.714 | 0.641 | 0.694 | 0.658 | 0.620 | 0.653 |
| SC459 | 0.659 | 0.692 | 0.685 | 0.591 | 0.660 | 0.687 | 0.620 | 0.674 |
| SC461 | 0.552 | 0.516 | 0.561 | 0.520 | 0.551 | 0.568 | 0.530 | 0.558 |
| SC473 | 0.524 | 0.580 | 0.607 | 0.447 | 0.589 | 0.630 | 0.483 | 0.609 |
| SC476 | 0.597 | 0.545 | 0.637 | 0.569 | 0.616 | 0.533 | 0.525 | 0.529 |

## D2

Run `cohort-d2`, split `2a0313adb97b9da0`, seed 0. Quality-control coverage 1.000 (21 flagged epochs of 81355).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.993 | 0.967 | 0.980 | 0.979 | 56983 | 15 |
| N1 | 0.397 | 0.656 | 0.495 | 0.501 | 3556 | 15 |
| N2 | 0.852 | 0.795 | 0.822 | 0.816 | 12477 | 15 |
| N3 | 0.730 | 0.751 | 0.740 | 0.522 | 2892 | 15 |
| REM | 0.798 | 0.787 | 0.792 | 0.795 | 5447 | 15 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 55126 | 1687 | 33 | 24 | 113 |
| N1 | 293 | 2331 | 498 | 51 | 383 |
| N2 | 53 | 1319 | 9913 | 613 | 579 |
| N3 | 0 | 10 | 703 | 2172 | 7 |
| REM | 33 | 521 | 491 | 117 | 4285 |

## D2-shuffled-context

Run `cohort-d2`, split `2a0313adb97b9da0`, seed 0. Quality-control coverage 1.000 (21 flagged epochs of 81355).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.993 | 0.968 | 0.980 | 0.979 | 56983 | 15 |
| N1 | 0.353 | 0.589 | 0.442 | 0.446 | 3556 | 15 |
| N2 | 0.839 | 0.776 | 0.806 | 0.800 | 12477 | 15 |
| N3 | 0.733 | 0.737 | 0.735 | 0.514 | 2892 | 15 |
| REM | 0.772 | 0.768 | 0.770 | 0.771 | 5447 | 15 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 55152 | 1649 | 44 | 17 | 121 |
| N1 | 321 | 2094 | 641 | 54 | 446 |
| N2 | 55 | 1486 | 9686 | 590 | 660 |
| N3 | 1 | 30 | 724 | 2130 | 7 |
| REM | 34 | 666 | 451 | 114 | 4182 |

## D3

Run `cohort-d3`, split `2a0313adb97b9da0`, seed 0. Quality-control coverage 1.000 (21 flagged epochs of 81355).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.995 | 0.958 | 0.976 | 0.975 | 56983 | 15 |
| N1 | 0.321 | 0.749 | 0.449 | 0.445 | 3556 | 15 |
| N2 | 0.876 | 0.682 | 0.767 | 0.758 | 12477 | 15 |
| N3 | 0.729 | 0.756 | 0.743 | 0.518 | 2892 | 15 |
| REM | 0.780 | 0.782 | 0.781 | 0.780 | 5447 | 15 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 54570 | 2263 | 6 | 18 | 126 |
| N1 | 244 | 2665 | 188 | 50 | 409 |
| N2 | 52 | 2628 | 8511 | 621 | 665 |
| N3 | 0 | 14 | 690 | 2187 | 1 |
| REM | 3 | 742 | 322 | 122 | 4258 |

## D3-shuffled-context

Run `cohort-d3`, split `2a0313adb97b9da0`, seed 0. Quality-control coverage 1.000 (21 flagged epochs of 81355).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.994 | 0.958 | 0.976 | 0.975 | 56983 | 15 |
| N1 | 0.301 | 0.711 | 0.423 | 0.418 | 3556 | 15 |
| N2 | 0.868 | 0.671 | 0.757 | 0.748 | 12477 | 15 |
| N3 | 0.729 | 0.742 | 0.735 | 0.512 | 2892 | 15 |
| REM | 0.763 | 0.768 | 0.766 | 0.764 | 5447 | 15 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 54563 | 2258 | 10 | 16 | 136 |
| N1 | 249 | 2530 | 286 | 53 | 438 |
| N2 | 51 | 2715 | 8377 | 612 | 722 |
| N3 | 0 | 59 | 687 | 2145 | 1 |
| REM | 4 | 850 | 292 | 118 | 4183 |

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

## D3-context-masked

Run `cohort-d3`, split `2a0313adb97b9da0`, seed 0. Quality-control coverage 1.000 (21 flagged epochs of 81355).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.991 | 0.947 | 0.969 | 0.967 | 56983 | 15 |
| N1 | 0.265 | 0.501 | 0.347 | 0.355 | 3556 | 15 |
| N2 | 0.827 | 0.756 | 0.790 | 0.785 | 12477 | 15 |
| N3 | 0.681 | 0.767 | 0.721 | 0.525 | 2892 | 15 |
| REM | 0.625 | 0.631 | 0.628 | 0.627 | 5447 | 15 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 53984 | 2545 | 48 | 45 | 361 |
| N1 | 297 | 1782 | 734 | 64 | 679 |
| N2 | 94 | 1148 | 9435 | 783 | 1017 |
| N3 | 5 | 29 | 635 | 2217 | 6 |
| REM | 88 | 1219 | 557 | 148 | 3435 |

## D2-context-masked

Run `cohort-d2`, split `2a0313adb97b9da0`, seed 0. Quality-control coverage 1.000 (21 flagged epochs of 81355).

### Per stage

| stage | precision | recall | F1 (pooled) | F1 (participant mean) | support | participants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wake | 0.989 | 0.966 | 0.977 | 0.976 | 56983 | 15 |
| N1 | 0.307 | 0.365 | 0.334 | 0.346 | 3556 | 15 |
| N2 | 0.767 | 0.805 | 0.785 | 0.781 | 12477 | 15 |
| N3 | 0.701 | 0.732 | 0.716 | 0.509 | 2892 | 15 |
| REM | 0.622 | 0.613 | 0.617 | 0.618 | 5447 | 15 |

### Confusion matrix

| true \ predicted | Wake | N1 | N2 | N3 | REM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wake | 55033 | 1519 | 119 | 16 | 296 |
| N1 | 364 | 1299 | 1151 | 39 | 703 |
| N2 | 102 | 600 | 10045 | 728 | 1002 |
| N3 | 2 | 18 | 730 | 2117 | 25 |
| REM | 140 | 793 | 1057 | 120 | 3337 |
