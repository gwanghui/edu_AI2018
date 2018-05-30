# Logistic Regression
- classification Problem 
  - 선형이다 보니 x값에 의해 0~1을 넘을 수 있다.
  - Outlier에 취약하다

- f(x) = w0x0 + w1x1 + w2x2 + ... + wdxd
```text
         d
f(x) = ∑ wixi = w.Trans()x
        i=0
``` 
 
- Sigmoid Fuction

```text
           L
σ(x) = ------------
       1+e^-k(x-x0)
```

- Odd
  - 이길 확률과 질 확률에 베팅
  - ln을 취해준다 ln(odds) = ln(p / (1-p))
 
 - Formulation of Logistic Regression
 
 ## Negative Conditional Log Lidelihood
 
 
  
