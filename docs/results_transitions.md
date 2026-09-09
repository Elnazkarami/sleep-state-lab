# Performance by distance from a stage change

Generated from `predictions_cohort.csv`, split part `test`.

Annotations are read here **only to decide which epochs to report separately**. Distance from a transition never entered training, sampling, class weighting or model selection; these are the same saved predictions every other table is built from, grouped after the fact.

A stage change is a change between two epochs whose original indices differ by exactly one. Two epochs either side of an excluded epoch are not a transition -- nothing is known about what happened between them.

**What this cannot show:** that stages are attractors, that transitions are bifurcations, or anything about neural dynamics. It shows where a classifier's errors are concentrated, which is a fact about the classifier.

## D1

| distance from a stage change | epochs | share | participant macro-F1 | accuracy | participants |
| --- | ---: | ---: | ---: | ---: | ---: |
| at a change | 6448 | 7.9% | 0.490 ± 0.084 | 0.545 | 15 |
| 1 epoch away | 3457 | 4.2% | 0.560 ± 0.109 | 0.617 | 15 |
| 2 epochs away | 2530 | 3.1% | 0.602 ± 0.110 | 0.661 | 15 |
| 3-5 epochs away | 5134 | 6.3% | 0.610 ± 0.128 | 0.713 | 15 |
| 6+ epochs away | 61592 | 75.7% | 0.652 ± 0.113 | 0.944 | 15 |
| no change in the run | 2194 | 2.7% | 0.232 ± 0.048 | 0.898 | 5 |

## D1+smoothing

| distance from a stage change | epochs | share | participant macro-F1 | accuracy | participants |
| --- | ---: | ---: | ---: | ---: | ---: |
| at a change | 6448 | 7.9% | 0.416 ± 0.073 | 0.478 | 15 |
| 1 epoch away | 3457 | 4.2% | 0.515 ± 0.105 | 0.565 | 15 |
| 2 epochs away | 2530 | 3.1% | 0.539 ± 0.111 | 0.595 | 15 |
| 3-5 epochs away | 5134 | 6.3% | 0.567 ± 0.131 | 0.650 | 15 |
| 6+ epochs away | 61592 | 75.7% | 0.655 ± 0.132 | 0.932 | 15 |
| no change in the run | 2194 | 2.7% | 0.350 ± 0.111 | 0.908 | 5 |

## D2

| distance from a stage change | epochs | share | participant macro-F1 | accuracy | participants |
| --- | ---: | ---: | ---: | ---: | ---: |
| at a change | 6448 | 7.9% | 0.492 ± 0.069 | 0.548 | 15 |
| 1 epoch away | 3457 | 4.2% | 0.633 ± 0.123 | 0.682 | 15 |
| 2 epochs away | 2530 | 3.1% | 0.656 ± 0.122 | 0.743 | 15 |
| 3-5 epochs away | 5134 | 6.3% | 0.686 ± 0.117 | 0.800 | 15 |
| 6+ epochs away | 61592 | 75.7% | 0.715 ± 0.103 | 0.972 | 15 |
| no change in the run | 2194 | 2.7% | 0.515 ± 0.261 | 0.955 | 5 |

## D2-context-masked

| distance from a stage change | epochs | share | participant macro-F1 | accuracy | participants |
| --- | ---: | ---: | ---: | ---: | ---: |
| at a change | 6448 | 7.9% | 0.484 ± 0.095 | 0.525 | 15 |
| 1 epoch away | 3457 | 4.2% | 0.566 ± 0.119 | 0.623 | 15 |
| 2 epochs away | 2530 | 3.1% | 0.583 ± 0.123 | 0.664 | 15 |
| 3-5 epochs away | 5134 | 6.3% | 0.613 ± 0.120 | 0.739 | 15 |
| 6+ epochs away | 61592 | 75.7% | 0.631 ± 0.113 | 0.954 | 15 |
| no change in the run | 2194 | 2.7% | 0.263 ± 0.052 | 0.930 | 5 |

## D2-shuffled-context

| distance from a stage change | epochs | share | participant macro-F1 | accuracy | participants |
| --- | ---: | ---: | ---: | ---: | ---: |
| at a change | 6448 | 7.9% | 0.476 ± 0.076 | 0.519 | 15 |
| 1 epoch away | 3457 | 4.2% | 0.593 ± 0.113 | 0.634 | 15 |
| 2 epochs away | 2530 | 3.1% | 0.629 ± 0.115 | 0.692 | 15 |
| 3-5 epochs away | 5134 | 6.3% | 0.666 ± 0.118 | 0.778 | 15 |
| 6+ epochs away | 61592 | 75.7% | 0.712 ± 0.105 | 0.972 | 15 |
| no change in the run | 2194 | 2.7% | 0.501 ± 0.270 | 0.955 | 5 |

## D3

| distance from a stage change | epochs | share | participant macro-F1 | accuracy | participants |
| --- | ---: | ---: | ---: | ---: | ---: |
| at a change | 6448 | 7.9% | 0.472 ± 0.075 | 0.527 | 15 |
| 1 epoch away | 3457 | 4.2% | 0.553 ± 0.126 | 0.601 | 15 |
| 2 epochs away | 2530 | 3.1% | 0.593 ± 0.116 | 0.651 | 15 |
| 3-5 epochs away | 5134 | 6.3% | 0.632 ± 0.119 | 0.728 | 15 |
| 6+ epochs away | 61592 | 75.7% | 0.706 ± 0.103 | 0.962 | 15 |
| no change in the run | 2194 | 2.7% | 0.364 ± 0.159 | 0.963 | 5 |

## D3-context-masked

| distance from a stage change | epochs | share | participant macro-F1 | accuracy | participants |
| --- | ---: | ---: | ---: | ---: | ---: |
| at a change | 6448 | 7.9% | 0.486 ± 0.086 | 0.535 | 15 |
| 1 epoch away | 3457 | 4.2% | 0.554 ± 0.119 | 0.620 | 15 |
| 2 epochs away | 2530 | 3.1% | 0.588 ± 0.100 | 0.662 | 15 |
| 3-5 epochs away | 5134 | 6.3% | 0.597 ± 0.112 | 0.706 | 15 |
| 6+ epochs away | 61592 | 75.7% | 0.636 ± 0.111 | 0.941 | 15 |
| no change in the run | 2194 | 2.7% | 0.229 ± 0.021 | 0.900 | 5 |

## D3-shuffled-context

| distance from a stage change | epochs | share | participant macro-F1 | accuracy | participants |
| --- | ---: | ---: | ---: | ---: | ---: |
| at a change | 6448 | 7.9% | 0.455 ± 0.076 | 0.504 | 15 |
| 1 epoch away | 3457 | 4.2% | 0.533 ± 0.133 | 0.577 | 15 |
| 2 epochs away | 2530 | 3.1% | 0.571 ± 0.128 | 0.616 | 15 |
| 3-5 epochs away | 5134 | 6.3% | 0.619 ± 0.121 | 0.710 | 15 |
| 6+ epochs away | 61592 | 75.7% | 0.705 ± 0.102 | 0.962 | 15 |
| no change in the run | 2194 | 2.7% | 0.377 ± 0.145 | 0.963 | 5 |
