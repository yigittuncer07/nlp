
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.9406
- **Recall:** 0.9349
- **F1-Score:** 0.9377
- **Accuracy:** 0.9663

## Confusion Matrix
|  | B-LOC | B-ORG | B-PER | I-LOC | I-ORG | I-PER | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOC | 4702 | 157 | 31 | 21 | 7 | 2 | 94 |
| B-ORG | 207 | 3706 | 78 | 0 | 27 | 4 | 107 |
| B-PER | 37 | 60 | 4193 | 0 | 0 | 16 | 68 |
| I-LOC | 27 | 1 | 1 | 2735 | 237 | 23 | 112 |
| I-ORG | 7 | 22 | 3 | 175 | 6806 | 106 | 171 |
| I-PER | 3 | 2 | 13 | 46 | 103 | 5289 | 82 |
| O | 97 | 69 | 70 | 52 | 134 | 63 | 45309 |



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

