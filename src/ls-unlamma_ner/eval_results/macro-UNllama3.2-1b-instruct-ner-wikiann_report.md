
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.9287
- **Recall:** 0.9183
- **F1-Score:** 0.9233
- **Accuracy:** 0.9585

## Confusion Matrix
|  | B-LOC | B-ORG | B-PER | I-LOC | I-ORG | I-PER | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOC | 4622 | 159 | 27 | 37 | 18 | 2 | 149 |
| B-ORG | 246 | 3567 | 85 | 1 | 52 | 5 | 173 |
| B-PER | 46 | 75 | 4150 | 0 | 2 | 20 | 81 |
| I-LOC | 29 | 1 | 0 | 2628 | 294 | 29 | 155 |
| I-ORG | 8 | 33 | 5 | 206 | 6734 | 103 | 201 |
| I-PER | 4 | 1 | 15 | 59 | 131 | 5231 | 97 |
| O | 115 | 83 | 82 | 66 | 161 | 66 | 45221 |



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

