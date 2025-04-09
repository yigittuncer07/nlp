
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.7769
- **Recall:** 0.7393
- **F1-Score:** 0.7558
- **Accuracy:** 0.9571

## Confusion Matrix
|  | B-LOCATION | B-ORGANIZATION | B-PERSON | I-LOCATION | I-ORGANIZATION | I-PERSON | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOCATION | 593 | 52 | 71 | 2 | 2 | 1 | 179 |
| B-ORGANIZATION | 82 | 450 | 64 | 0 | 7 | 1 | 227 |
| B-PERSON | 45 | 44 | 915 | 0 | 3 | 11 | 295 |
| I-LOCATION | 5 | 0 | 0 | 61 | 22 | 8 | 10 |
| I-ORGANIZATION | 4 | 1 | 0 | 16 | 447 | 9 | 108 |
| I-PERSON | 1 | 0 | 2 | 4 | 13 | 630 | 12 |
| O | 98 | 92 | 146 | 12 | 127 | 29 | 37158 |



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

