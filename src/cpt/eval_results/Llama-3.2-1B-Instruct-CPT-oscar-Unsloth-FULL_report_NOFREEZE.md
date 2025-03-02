
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.9400
- **Recall:** 0.9365
- **F1-Score:** 0.9382
- **Accuracy:** 0.9671

## Confusion Matrix
|  | B-LOC | B-ORG | B-PER | I-LOC | I-ORG | I-PER | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOC | 4736 | 123 | 28 | 31 | 8 | 4 | 84 |
| B-ORG | 202 | 3701 | 90 | 2 | 36 | 3 | 95 |
| B-PER | 53 | 74 | 4161 | 0 | 5 | 28 | 53 |
| I-LOC | 32 | 0 | 1 | 2792 | 197 | 19 | 95 |
| I-ORG | 8 | 14 | 9 | 215 | 6780 | 101 | 163 |
| I-PER | 3 | 0 | 8 | 53 | 125 | 5275 | 74 |
| O | 93 | 66 | 64 | 46 | 108 | 61 | 45356 |



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

