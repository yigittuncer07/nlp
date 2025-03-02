
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.9339
- **Recall:** 0.9249
- **F1-Score:** 0.9293
- **Accuracy:** 0.9615

## Confusion Matrix
|  | B-LOC | B-ORG | B-PER | I-LOC | I-ORG | I-PER | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOC | 4645 | 167 | 19 | 33 | 13 | 4 | 133 |
| B-ORG | 224 | 3631 | 70 | 1 | 38 | 4 | 161 |
| B-PER | 30 | 72 | 4146 | 0 | 0 | 31 | 95 |
| I-LOC | 32 | 0 | 1 | 2695 | 255 | 17 | 136 |
| I-ORG | 7 | 27 | 4 | 210 | 6748 | 105 | 189 |
| I-PER | 3 | 3 | 10 | 52 | 130 | 5244 | 96 |
| O | 84 | 74 | 82 | 65 | 159 | 65 | 45265 |



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

