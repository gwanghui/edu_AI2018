# Evaluating Classfication Model
## Linear Basis Function Models
- 선형식을 기준으로 만든 
- y(w,x) = w0+ w1x1 + ㆍㆍㆍ + wdxd

## Polynomial Curve Fitting
- 다항식으로 만든 non-liearfunction
- y(w,x) = w0 + w1x + w2x² + w3x³ + ㆍㆍㆍ + wMxⁿ 

## Overfitting
- Training data에 대해 너무 특화되어 있어 실제 Test data의 정확도가 떨어진다.
  
### Check Point
- 같은 모집합에서 데이터에서 왔는가
- Overfitting 되었는가?

### 복잡한 모델
- 파라미터의 개수가 많은가?
- 같은 파라미터의 개수 안에서 내부 값 차이가 얼마나 큰가?

## Overfitting 을 없애보자
### Generalization
#### Regularization
- Shrinking to zero
- loss + regularization
- weight decay, ridge regression

## Holdout Method?
- data = traningSet + Test set

## Cross-Validation (k-fold)
- Data D is randomly partitioned into k mutually exclusive subsets {D₁, D₂, D₃, D₄...,Dk)
- each approximately equal size.
- 서브셋을 바꿔가면서 돌려보고 검증해본다

### Leave-one-out
- 한개만 빼서 테스트 해본다.
- 극단적으로 n과 k를 같게한다.
- n = 100
- k = 100
