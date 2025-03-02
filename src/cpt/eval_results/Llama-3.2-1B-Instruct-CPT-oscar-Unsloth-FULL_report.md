
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.9504
- **Recall:** 0.9478
- **F1-Score:** 0.9491
- **Accuracy:** 0.9736

## Confusion Matrix
|  | B-LOC | B-ORG | B-PER | I-LOC | I-ORG | I-PER | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOC | 4760 | 144 | 24 | 14 | 8 | 4 | 60 |
| B-ORG | 154 | 3839 | 54 | 1 | 21 | 3 | 57 |
| B-PER | 41 | 66 | 4205 | 0 | 2 | 9 | 51 |
| I-LOC | 32 | 2 | 1 | 2813 | 178 | 17 | 93 |
| I-ORG | 9 | 18 | 3 | 195 | 6875 | 74 | 116 |
| I-PER | 5 | 1 | 12 | 44 | 103 | 5323 | 50 |
| O | 70 | 65 | 54 | 37 | 71 | 25 | 45472 |



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

