
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.9275
- **Recall:** 0.9186
- **F1-Score:** 0.9229
- **Accuracy:** 0.9583

## Confusion Matrix
|  | B-LOC | B-ORG | B-PER | I-LOC | I-ORG | I-PER | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOC | 4619 | 185 | 32 | 28 | 13 | 2 | 135 |
| B-ORG | 249 | 3591 | 78 | 0 | 48 | 3 | 160 |
| B-PER | 41 | 84 | 4147 | 1 | 3 | 18 | 80 |
| I-LOC | 31 | 2 | 0 | 2627 | 318 | 26 | 132 |
| I-ORG | 7 | 42 | 7 | 198 | 6738 | 79 | 219 |
| I-PER | 4 | 1 | 16 | 60 | 142 | 5219 | 96 |
| O | 103 | 95 | 98 | 69 | 168 | 69 | 45192 |



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

