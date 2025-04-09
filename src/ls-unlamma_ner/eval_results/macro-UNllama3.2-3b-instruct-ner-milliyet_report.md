
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.9405
- **Recall:** 0.9522
- **F1-Score:** 0.9462
- **Accuracy:** 0.9921

## Confusion Matrix
|  | B-LOCATION | B-ORGANIZATION | B-PERSON | I-LOCATION | I-ORGANIZATION | I-PERSON | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOCATION | 864 | 10 | 6 | 4 | 2 | 0 | 14 |
| B-ORGANIZATION | 7 | 788 | 4 | 0 | 6 | 0 | 26 |
| B-PERSON | 3 | 7 | 1279 | 0 | 0 | 6 | 18 |
| I-LOCATION | 5 | 0 | 0 | 93 | 0 | 1 | 7 |
| I-ORGANIZATION | 2 | 4 | 0 | 5 | 547 | 0 | 27 |
| I-PERSON | 0 | 0 | 2 | 0 | 6 | 645 | 9 |
| O | 27 | 33 | 28 | 7 | 42 | 14 | 37511 |



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

