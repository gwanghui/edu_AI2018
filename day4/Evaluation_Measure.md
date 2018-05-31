# Evaluation Measure
## Confusion Matrix
Actual/Predicted|Cp|~CP
-|-|-
Ca | True Positive (TP) | False Negative (FN)
~Ca | False Positive (FP) | True Negative (TN)

- 내가 예측한거 기준으로 읽자



### Accuracy
qwe|Play_Game=Yes|Play_Game=No|total
-|-|-|-
Play_Game=Yes | 6100 | 900 | 7000
Play_Game=No | 500 | 2500 | 3000
total | 6600 | 3400 | 10000

- 내가 잘 예측한건 (Yes, Yes), (No, No)
- 6100 + 2500 / 10000

### Error rate 
- 1 - accuracy


### Precision
- Exactness : true로 판단한 모수에서 True로 잘 판단한 경우
- 관심있는 true값만 보자
- TP / TP + FP

### Recall
- Completeness : True여야 하는 모수에서 얼마나 True로 계산했냐?
- TP = TP / TP + FN

- 보통의 경우 서로 상호 보완 관계이다.

### F-Measure ( F-Score )
- Harmonic Mean (조화평균) of precision and recall

- F = 2 * precision * recall / precision + recall
- FB = weighted Measure of precision and recall
  - Assign B times as much weight to recall as to precision.

### Sensitivity
- True Positive recognition rate
  - Sensitivity = TP / TP + FN

### Specificity 
- True Ngative recognition rate
  - Specificity = TN / FP + TN
  

  
