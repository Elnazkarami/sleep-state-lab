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
