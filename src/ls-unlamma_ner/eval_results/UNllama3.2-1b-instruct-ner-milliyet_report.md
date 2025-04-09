
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.9896
- **Recall:** 0.9897
- **F1-Score:** 0.9896
- **Accuracy:** 0.9897

## Confusion Matrix
|  | B-LOCATION | B-ORGANIZATION | B-PERSON | I-LOCATION | I-ORGANIZATION | I-PERSON | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOCATION | 852 | 18 | 8 | 4 | 0 | 0 | 18 |
| B-ORGANIZATION | 14 | 762 | 6 | 0 | 5 | 0 | 44 |
| B-PERSON | 4 | 7 | 1260 | 1 | 0 | 12 | 29 |
| I-LOCATION | 6 | 0 | 0 | 89 | 0 | 1 | 10 |
| I-ORGANIZATION | 1 | 5 | 2 | 3 | 535 | 4 | 35 |
| I-PERSON | 1 | 0 | 4 | 0 | 5 | 640 | 12 |
| O | 31 | 36 | 48 | 8 | 39 | 14 | 37486 |



## Dataset Summary
- **Training Set Size:** 22338
- **Validation Set Size:** 2482
- **Test Set Size:** 2751
- **Average Input Length:** 18.80

### Label Distribution (Training Set)
- **B-LOCATION**: 8821
- **B-ORGANIZATION**: 8316
- **B-PERSON**: 13290
- **I-LOCATION**: 1284
- **I-ORGANIZATION**: 5860
- **I-PERSON**: 6314
- **O**: 376111

