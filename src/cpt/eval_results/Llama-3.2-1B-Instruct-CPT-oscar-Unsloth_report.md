
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.9531
- **Recall:** 0.9526
- **F1-Score:** 0.9529
- **Accuracy:** 0.9752

## Confusion Matrix
|  | B-LOC | B-ORG | B-PER | I-LOC | I-ORG | I-PER | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOC | 4780 | 124 | 16 | 18 | 6 | 2 | 68 |
| B-ORG | 157 | 3834 | 56 | 1 | 24 | 3 | 54 |
| B-PER | 24 | 54 | 4246 | 0 | 1 | 16 | 33 |
| I-LOC | 23 | 0 | 2 | 2852 | 158 | 18 | 83 |
| I-ORG | 7 | 17 | 4 | 201 | 6895 | 61 | 105 |
| I-PER | 3 | 2 | 12 | 22 | 90 | 5363 | 46 |
| O | 65 | 70 | 61 | 39 | 84 | 39 | 45436 |



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

