# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** 0.9518
- **Recall:** 0.9498
- **F1-Score:** 0.9508
- **Accuracy:** 0.9739

## Confusion Matrix
|  | B-LOC | B-ORG | B-PER | I-LOC | I-ORG | I-PER | O |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-LOC | 4780 | 127 | 13 | 19 | 8 | 3 | 64 |
| B-ORG | 159 | 3812 | 54 | 0 | 28 | 2 | 74 |
| B-PER | 30 | 63 | 4215 | 0 | 2 | 10 | 54 |
| I-LOC | 16 | 3 | 0 | 2837 | 189 | 11 | 80 |
| I-ORG | 6 | 15 | 4 | 161 | 6934 | 73 | 97 |
| I-PER | 3 | 1 | 11 | 46 | 105 | 5324 | 48 |
| O | 75 | 67 | 47 | 51 | 108 | 39 | 45407 |



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
~                                                                                                                                                                                                          
~                                          
