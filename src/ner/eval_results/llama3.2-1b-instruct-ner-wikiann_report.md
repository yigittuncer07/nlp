
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.8833
- **Recall:** 0.8862
- **F1-Score:** 0.8837
- **Accuracy:** 0.8862

## Confusion Matrix
|  | B-LOC | B-ORG | B-PER | I-LOC | I-ORG | I-PER | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOC | 3419 | 380 | 531 | 30 | 30 | 7 | 617 |
| B-ORG | 906 | 2096 | 481 | 3 | 36 | 6 | 601 |
| B-PER | 601 | 243 | 3017 | 4 | 12 | 35 | 462 |
| I-LOC | 41 | 4 | 3 | 2393 | 379 | 88 | 228 |
| I-ORG | 33 | 29 | 16 | 380 | 6186 | 285 | 361 |
| I-PER | 5 | 5 | 18 | 98 | 218 | 5052 | 142 |
| O | 331 | 246 | 315 | 81 | 189 | 86 | 44546 |



## Dataset Summary
- **Training Set Size:** 20000
- **Validation Set Size:** 10000
- **Test Set Size:** 10000
- **Average Input Length:** 7.49

### Label Distribution (Training Set)
- **B-LOC**: 9679
- **B-ORG**: 7970
- **B-PER**: 8833
- **I-LOC**: 6501
- **I-ORG**: 14257
- **I-PER**: 11404
- **O**: 91142

