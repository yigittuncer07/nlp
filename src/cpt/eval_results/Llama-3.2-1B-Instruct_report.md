
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.9338
- **Recall:** 0.9230
- **F1-Score:** 0.9283
- **Accuracy:** 0.9612

## Confusion Matrix
|  | B-LOC | B-ORG | B-PER | I-LOC | I-ORG | I-PER | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOC | 4664 | 157 | 24 | 25 | 7 | 1 | 136 |
| B-ORG | 256 | 3600 | 75 | 2 | 35 | 4 | 157 |
| B-PER | 50 | 78 | 4128 | 0 | 3 | 19 | 96 |
| I-LOC | 30 | 2 | 1 | 2693 | 247 | 24 | 139 |
| I-ORG | 10 | 25 | 6 | 194 | 6731 | 115 | 209 |
| I-PER | 4 | 2 | 25 | 60 | 117 | 5227 | 103 |
| O | 100 | 71 | 67 | 59 | 123 | 62 | 45312 |



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

