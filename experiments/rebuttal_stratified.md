## Per-language results (accuracy vs intended, default-capture, interaction rate)

| Model | Mode | n EN | EN Acc [95% CI] | EN Default | EN IR | n ZH | ZH Acc [95% CI] | ZH Default | ZH IR |
|---|---|---|---|---|---|---|---|---|---|
| gpt-5 | answer-only | 53 | 1.9 [0, 6] | 7.5 | 0 | 120 | 0.0 [0, 0] | 0.0 | 0 |
| gpt-5 | answer+search | 53 | 9.4 [2, 19] | 7.5 | 0 | 120 | 5.0 [2, 9] | 11.7 | 0 |
| gpt-5 | answer+search+interact | 53 | 24.5 [13, 36] | 0.0 | 100 | 120 | 18.3 [12, 26] | 17.5 | 98 |
| gpt-4o | answer-only | 53 | 1.9 [0, 6] | 1.9 | 0 | 120 | 0.0 [0, 0] | 0.0 | 0 |
| gpt-4o | answer+search | 53 | 15.1 [6, 25] | 7.5 | 0 | 120 | 10.8 [6, 17] | 50.0 | 0 |
| gpt-4o | answer+search+interact | 53 | 13.2 [6, 23] | 3.8 | 98 | 120 | 0.8 [0, 2] | 7.5 | 100 |
| gpt-5-mini | answer-only | 53 | 1.9 [0, 6] | 0.0 | 0 | 120 | 0.0 [0, 0] | 0.0 | 0 |
| gpt-5-mini | answer+search | 53 | 0.0 [0, 0] | 0.0 | 0 | 120 | 0.0 [0, 0] | 0.0 | 0 |
| gpt-5-mini | answer+search+interact | 53 | 0.0 [0, 0] | 0.0 | 100 | 120 | 0.0 [0, 0] | 0.0 | 94 |
| qwen3-4b | answer-only | 53 | 0.0 [0, 0] | 0.0 | 0 | 120 | 0.0 [0, 0] | 0.0 | 0 |
| qwen3-4b | answer+search | 53 | 17.0 [8, 28] | 5.7 | 0 | 120 | 4.2 [1, 8] | 70.0 | 0 |
| qwen3-4b | answer+search+interact | 53 | 20.8 [11, 32] | 9.4 | 89 | 120 | 8.3 [4, 13] | 60.8 | 1 |
| qwen3-8b | answer-only | 53 | 0.0 [0, 0] | 0.0 | 0 | 120 | 0.8 [0, 2] | 0.0 | 0 |
| qwen3-8b | answer+search | 53 | 17.0 [8, 28] | 7.5 | 0 | 120 | 13.3 [8, 20] | 61.7 | 0 |
| qwen3-8b | answer+search+interact | 53 | 20.8 [9, 32] | 5.7 | 49 | 120 | 26.7 [19, 34] | 49.2 | 0 |
| qwen3-14b | answer-only | 53 | 0.0 [0, 0] | 0.0 | 0 | 120 | 0.0 [0, 0] | 0.0 | 0 |
| qwen3-14b | answer+search | 53 | 20.8 [11, 32] | 5.7 | 0 | 120 | 17.5 [11, 24] | 51.7 | 0 |
| qwen3-14b | answer+search+interact | 53 | 15.1 [6, 26] | 5.7 | 96 | 120 | 25.8 [18, 34] | 52.5 | 9 |
| qwen3-32b | answer-only | 53 | 1.9 [0, 6] | 1.9 | 0 | 120 | 0.8 [0, 2] | 0.0 | 0 |
| qwen3-32b | answer+search | 53 | 24.5 [13, 38] | 3.8 | 0 | 120 | 28.3 [21, 37] | 43.3 | 0 |
| qwen3-32b | answer+search+interact | 35 | 14.3 [3, 26] | 0.0 | 100 | 120 | 6.7 [2, 12] | 71.7 | 50 |
| qwen3-30b-a3b | answer-only | 53 | 1.9 [0, 6] | 0.0 | 0 | 120 | 0.8 [0, 2] | 0.8 | 0 |
| qwen3-30b-a3b | answer+search | 53 | 20.8 [11, 32] | 5.7 | 0 | 120 | 31.7 [23, 40] | 52.5 | 0 |
| qwen3-30b-a3b | answer+search+interact | 53 | 0.0 [0, 0] | 3.8 | 11 | 120 | 16.7 [10, 23] | 6.7 | 98 |
| qwen3p5-35b-a3b | answer-only | 53 | 1.9 [0, 6] | 1.9 | 0 | 120 | 0.0 [0, 0] | 0.0 | 0 |
| qwen3p5-35b-a3b | answer+search | 53 | 20.8 [11, 32] | 11.3 | 0 | 120 | 42.5 [34, 52] | 40.0 | 0 |
| qwen3p5-35b-a3b | answer+search+interact | 53 | 15.1 [6, 26] | 3.8 | 57 | 120 | 35.0 [27, 43] | 23.3 | 28 |

## Single-gold inflation per language (answer+search: default-capture / intended accuracy)

| Model | EN intended | EN default | EN ratio | ZH intended | ZH default | ZH ratio | All ratio |
|---|---|---|---|---|---|---|---|
| gpt-5 | 9.4 | 7.5 | 0.8x | 5.0 | 11.7 | 2.3x | 1.6x |
| gpt-4o | 15.1 | 7.5 | 0.5x | 10.8 | 50.0 | 4.6x | 3.0x |
| gpt-5-mini | 0.0 | 0.0 | n/a | 0.0 | 0.0 | n/a | n/a (acc 0) |
| qwen3-4b | 17.0 | 5.7 | 0.3x | 4.2 | 70.0 | 16.8x | 6.2x |
| qwen3-8b | 17.0 | 7.5 | 0.4x | 13.3 | 61.7 | 4.6x | 3.1x |
| qwen3-14b | 20.8 | 5.7 | 0.3x | 17.5 | 51.7 | 3.0x | 2.0x |
| qwen3-32b | 24.5 | 3.8 | 0.2x | 28.3 | 43.3 | 1.5x | 1.1x |
| qwen3-30b-a3b | 20.8 | 5.7 | 0.3x | 31.7 | 52.5 | 1.7x | 1.3x |
| qwen3p5-35b-a3b | 20.8 | 11.3 | 0.5x | 42.5 | 40.0 | 0.9x | 0.9x |

Inflation ratio over 8 full-scale models with non-zero intended accuracy: min 0.9x, median 2.0x, max 6.2x; models where default-capture exceeds intended accuracy: 7/8.

## Per-category results, +interact mode (accuracy vs intended / interaction rate)

| Model | entity_scope (n) | metric_definition (n) | temporal_scope (n) | recognition_policy (n) |
|---|---|---|---|---|
| gpt-5 | 23.4 / IR 99 (94) | 12.7 / IR 98 (63) | 57.1 / IR 100 (7) | 11.1 / IR 100 (9) |
| gpt-4o | 5.3 / IR 99 (94) | 1.6 / IR 100 (63) | 14.3 / IR 100 (7) | 11.1 / IR 100 (9) |
| gpt-5-mini | 0.0 / IR 94 (94) | 0.0 / IR 98 (63) | 0.0 / IR 100 (7) | 0.0 / IR 100 (9) |
| qwen3-4b | 16.0 / IR 18 (94) | 6.3 / IR 30 (63) | 0.0 / IR 100 (7) | 22.2 / IR 56 (9) |
| qwen3-8b | 29.8 / IR 6 (94) | 20.6 / IR 24 (63) | 14.3 / IR 71 (7) | 11.1 / IR 0 (9) |
| qwen3-14b | 27.7 / IR 29 (94) | 19.0 / IR 30 (63) | 0.0 / IR 100 (7) | 11.1 / IR 100 (9) |
| qwen3-32b | 10.6 / IR 62 (85) | 6.3 / IR 56 (63) | 0.0 / IR 100 (5) | 0.0 / IR 100 (2) |
| qwen3-30b-a3b | 12.8 / IR 81 (94) | 12.7 / IR 70 (63) | 0.0 / IR 0 (7) | 0.0 / IR 44 (9) |
| qwen3p5-35b-a3b | 30.9 / IR 36 (94) | 28.6 / IR 30 (63) | 28.6 / IR 71 (7) | 11.1 / IR 56 (9) |

## Elicitation gap per language (context-oracle ceiling vs +interact)

| Model | EN ceiling | EN +interact | EN gap | ZH ceiling | ZH +interact | ZH gap |
|---|---|---|---|---|---|---|
| gpt-5 | 84.9 | 24.5 | 60.4 | 100.0 | 18.3 | 81.7 |
| gpt-4o | 77.4 | 13.2 | 64.2 | 100.0 | 0.8 | 99.2 |
| gpt-5-mini | 86.8 | 0.0 | 86.8 | 99.2 | 0.0 | 99.2 |
| qwen3-4b | 62.3 | 20.8 | 41.5 | 98.3 | 8.3 | 90.0 |
| qwen3-8b | 75.5 | 20.8 | 54.7 | 100.0 | 26.7 | 73.3 |
| qwen3-14b | 71.7 | 15.1 | 56.6 | 99.2 | 25.8 | 73.3 |
| qwen3-32b | 77.4 | 14.3 | 63.1 | 100.0 | 6.7 | 93.3 |
| qwen3-30b-a3b | 69.8 | 0.0 | 69.8 | 99.2 | 16.7 | 82.5 |
| qwen3p5-35b-a3b | 73.6 | 15.1 | 58.5 | 100.0 | 35.0 | 65.0 |
