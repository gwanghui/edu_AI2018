# Decision Tree
- 결과를 설명 가능한 트리
- purity
  - 분류의 결과값이 큰게 purity가 크다고 한다
  
## How to Measure Purity
- 하나의 종류로 되어 있는게 더 좋은 구분이다
- entropy로 판단한다

### Information entropy
- entropy : 1940년의 정보의 아버지 샤론이 정의한 measure
- number of minimum bits
  - ex) A(1) B(10) C(01) 
  - A A A A A A A A B B C (6 4 2)
- 현재 데이터가 얼마나 섞여있느냐?  
  
### Information Gain


- Gain(A) = H(D) - I~a~(D)
