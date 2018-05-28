## Data Cleaning

- Incomplete : 직업란에 아무것도 써져 있지 않은 경우
- Noisy : 연봉이 -1000만원일 경우
- Inconsistent : 2개의 데이터가 같은 것을 표현해야 하는데 아닐경우 

ex) Rating was "1,2,3", now it is "A, B, C" 일부는 123 일부는 ABC가 존재한다

### Incomplete Data

- How to handle missing data?
  - 열이나 행을 삭제
  - Default 값을 추가

- Probability-based approach
  - Find the most probable value.
  - Inference-based method e.g. linear regression, K nearest neightbor

### Noisy Data
- Binning
  - sorted data for price : 4 8 15, 21, 21, 24, 25, 28, 34
    - bin 1 : 4, 8, 15
    - bin 2 : 21, 21, 24
    - bin 3 : 25, 28, 34
    
  - Smoothing by bin means : 평균으로 치환해버린다. bin1 : 9 9 9
  - Smoothing by bin boundaries : 선으로 밀어버린다. bin1 : 4 4 15
  
- Regression
- Clustering

## Data Tranformation
### Min-Max Normalization
- Transforming [min, max] into [new_min, new_max]

### Z-score Normalization
- Using mean and standard deviation

### Decimal Scaling Normalization
- 모든값을 1보다 작은 수로 바꾸기

## Data Preprocessing : Data Reduction
- 고차원 데이터를 저차원 데이터로 변경!

### Feature Selection
- 3차원 데이터를 가지고 선택한다면 부분집합에서 공집합과 전체 집합을 제외한 6개 (A, B, C, AB, AC, AB)

Greedy 하게
#### Forward Selection
```text
Initial Attribute Set
[A1,A2,A3,A4,A5,A6]
[]
=> [A1]
=> [A1,A4]
=> Reduced Attribute Set [A1,A4,A6]
```
#### Backward elimination
```text
Initial Attribute Set
[A1,A2,A3,A4,A5,A6]
=> Initial Attribute Set
=> [A1,A3,A4,A5,A6]
=> [A1,A4,A5,A6]
=> [A1,A4,A6]
```
#### Decision tree induction
```text
A4 if Y A1 
        if Y Class 1
        if N Class 2
   if N A6
        if Y Class 1
        if N Class 2
 
=> [A4,A1,A6]
```
### Feature Extraction
#### Singular Value Decomposition
- 각점들과의 수직거리가 가장 짧은 선을 긋자
#### Properties of SVD
- A[M*N] = U[m*m]∑[m*n](V[n*n])^T
- 행렬 분해~
- Diagonal matrix : 대각선으로 되어 있는 값이 정렬되어 있는 경우
```text

3 * 5 Diagonal Matrix : min(m,n)

| 0 X X X X |
| X 0 X X X |
| X X 0 X X |

특정 행렬을 투영 할때 0에 들어가는 값이 가중치 값이 된다.
```

#### Properties of SVD
U,∑,V : Unique

m*n을 K*k Matrix로 표현하겠다

## LSH : Finding Similar Data
### Scene Completion Problem
- 검색결과로 나온 이미지를 적절히 찾아서 채워주자!

http://graphics.cs.cmu.edu/projects/scene-completion/

- Problem Motivation
```text
| 1 2 1 |
| 0 2 1 | -> |1 2 1 0 2 1 0 1 0|
| 0 1 0 |
```
- Chanllenge : 고차원인 데이터를 어떻게 효율적으로 비슷한 형태의 사진을 찾을 수 있을까?

#### Common Metaphor
- 비슷한 경향을 가지는 친구들


#### Review : Document Similarity
- 편의상 Jaccard Similarity를 써서 계산해보자

#### Shingling
- k-shingle( or k-gram) : 단어를 이어서 생각해보기
- 장점 : 구절의 표현력이 좋아짐
- 단점 : Feature가 너무 많아져!

#### Problem Formulation
Goal : Jaccard Similarity 한정 Min Hashing 가능 d(xi,xj) >= s
document --> Shingling --> **Min Hashing** -> Locality Sensitive Hashing

요새 하는건 Semantic Hashing
(d1,d2) = 0.5
(d1,d3) = 0.7
... 
(d1, dx) = ?

#### Min Hashing
- Signatures : short integer vectors that represent the sets and relect their similarity ( 저차원으로 표시 )
  - Minhash = 임의의 순열을 기준으로 가장 처음 나온 1의 index를 표현한다. Idea
  - 근데 이게 Jaccard랑 같음!
 
 Pr[sim(sig(ci),sig(cj))] = sim(ci,cj)

#### Locality-Sensitive hashing
- First Cut of LSH
  - Problem
  - General idea : Candidate pairs를 잘 찾기!
  - Min-Hash matrices : 
  
- Assumption for LSH
  - if h(x1) == h(x2) 
     ==> x1 == x2
  - 지역에 같은 Hash값을 가진다면 Bucket에 넣어준다.
