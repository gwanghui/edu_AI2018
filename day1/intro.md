# Day 01

## Intro

### BigData 3V (IBM이 정의한 3V)
Volume, Velocity, Variety

#### Volume

#### Variety의 종류
- Unstructured data : Text
- semistructured data 
- structured data : RDS Table Data

#### Velocity
- 실시간 처리
- 데이터의 수명

#### Problem of Data Flooding
- 의미 있는 데이터를 찾아야한다.

### Typical View From ML and Statistics
- Data : 쌓여있는 정보들, 의미 있던 없던 쌓여있는 정보들
- Information : 정리되어 있는 정보들
- Knowledge : **알려지지 않았지만 안다면 도움이 될만한 정보들** (non-trivial, implicit, unknown and potentially useful)
  - ex) 월마트 사례
  
  
### Correlation(상관관계) vs Causality(인과관계)

- Correlation(상관관계) : A와 B는 연관이 있다
- Causality(인과관계) : A === B이다.

## Marchine Learning

### Types of Learning

**Feedback 관점에서의 차이점**

- Supervised Learning
  - Direct Feedback이 주어진 경우
- Unsupervised Learning
  - Feedback 자체가 주어지지 않는 경우
  - Feature를 표현하는 모델을 만드는!
- Reinforcement Learning
  - Relative Feedback이 주어진 경우 (Reward 개념)
    - reward가 쌓이면 의미가 있다
- Semi-supervised Learning
  - 일부는 레이블이 존재하고, 일부는 레이블이 존재하는 데이터 세팅
  
### Classification

- class label
- Predict the lable of a new record using the learned model.
- ex) Document Classification, Handwritten Digits Classification

## Unsupervised Learning
- Clustering : discover groups of similar examples within the data.
  - K-means clustering ex) 남녀로 구분해봐 , 3그룹으로 분류해봐.
- Dimensionality reduction : project the data from a high-dimensional space to a lower dimension space
  - Principle Component Analysis (PCA)
- Density estimation : determine the distribuion of data within the input space
  - Gaussian Mixture Model (GMM)
- Matrix completion : fill the missing entries of a partially observed matrix
  - low-rank approximation (LRA)
  
### Clustering

- 장점 : Labal이 필요가 없어서 데이터를 다루기가 쉽다.
- 단점 : 

**Non-linear similarities면 뷴류가 힘듬**
- similarities를 정의를 어떻게 정의할 것이냐.

- similar(or related)
- Dissimilar(or unrelated)

### Outlier Analysis
- Outlier
- Useful in fraud detection, rare events analysis

### Dimensionality Reduction

**가장 중요한 개념**

- 가정
  - 3차원 데이터 (3-dimensinal Data) 보통 x,y,z로 표현
  - 데이터 손실 없이 차원을 낮춰야 한다.
  - ex) (1,2,3)

```text
단위벡터에서의 표현
(1,2,3)
[
  1 * [1,0,0],
  2 * [0,1,0],
  3 * [0,0,1]
]

새로운 축에서의 표현
[
  x1 * [1,2,3],
  y1 * [2,3,4],
  
]
```

- 단위벡터

| /    | Mon | Tue | Wed | Thu | Fri |
| Alice| 2   | 3   | 0   | 0   | 0   |
| Bob  | 4   | 6   | 0   | 0   | 0   |
| Carol| 6   | 9   | 0   | 0   | 0   |
| David| 0   | 0   | 2   | 4   | 2   |
| Eve  | 0   | 0   | 3   | 6   | 3   |
| Frank| 0   | 0   | 1   | 2   | 1   |



- Feature Extraction

- Discover hidden correlations/topics
- Remove redundant and noisy features
- Interpretation and visualization

###  Density Estimation
https://www.youtube.com/watch?v=kPEIJJsQr7U&feature=youtu.be

- ex)저차원에서 숫자를 표현함에 있어서 우리가 생각하는 의미 공간에 표현을 한다.

### Matrix Completion

- ex)이미지 모자이크나 손상된 곳을 복원
- 공식 추후 
  
## Getting to Know Your Data
### Data Object
- 행하나하나를 

- ROws -> tuple
- column -> attribute
  - categorical 
    - Binary  : {0 , 1}, Symmetric, 
    - **Nominal** : states or "names of things"
    - **ordinal** : ex) size = {small, medium, large}, grades, army rankings
  - Numeric 
    - Interval
    - ratio 
    
### Statistical Description of Data
- Data Central tendency : mean, median, mode
  - mean :
  - median 
  홀수는 가운데, 짝수인 경우 가운데 있는 값의 평균으로 
  - median interpolation
  median 공식 삽입 예정
  - mode : 데이터의 빈도
  
- Data dispersion :  
  - variance
  - standard deviation


### Anscombe's Quartet

https://en.wikipedia.org/wiki/Anscombe%27s_quartet

### Quantile and Percentile

#### Quartiles and outliers
- the 4-quantiles are called quartiles.
- Inter-quartile range : IQR = Q3 - Q1
- Outlier : Values higher / lower than 1.5 * IQR

#### Five number summary
- min, Q1, median, Q3, max

### Boxplot

- 그림 삽입
- 
### Histogram

Boxplot 보다 정보량이 많다.
- D1 = {1,2,2,2,3,4,4,4,5}
- D2 = {1,2,2,3,3,3,4,4,5}

### Scatter Plot

- 산점도
- 2차원 데이터에 한해서 Positively, Negatively Correlated Data 확인 가능

### Heatmap

- 데이터 각각의 관계를 전체적으로 보고자 함

### Parallel Coordinates

- high dimensinal geometry
- Scalability of many objects
- Scalability of many attributes

### Multi-dimensional Scalings(MDS)
- T-SNE 


## Similarity Analysis
### Similarity and Dissimilarity

- Similarity 
  - 
- Dissimilarity
  - 편의상 1 - similarity (similarity를 최대 1로 지정할 경우)

### Simple Similarity Measure
    
A = 1 0 1 1 0
B = 1 0 0 1 0

- Simple matching coefficient (SMC)
SMC(A,B) = # of matching attributes / # of attributes = (M00 + M11) /(M11+M01+M10+M00)

- Jaccard similarity coefficient
Jaccard(A,B) = M11 / (M11 + M01 + M10)

- Dice coefficient
  - harmonic mean : 식 유도 과정 보기
  
### Minkowski Distance
- h = 1 : Manhattan distance 잔차의 절대값들을 전부 더한 값, 노드 네트워크에서 많이 쓰임
- h = 2 : Euclidean distance 대각선길이
- h = infinity : supremum distance = max(잔차의 절대값)

- h가 커질수록 distance가 작아짐

### How to represnet a Docunment?
#### Motivation : Bag of Words
- 형태소 분석

- 단어간 중요도나 반복수를 챙기지 못한다.
- 경향성을 판단하기 힘들다. 
- 의미론적으로 같은 단어를 비교하지 힘들다.

### Vector Space Model

- **Cosine Similarity** : 교수님 曰 문서화는 닥치고 Cosine Similarity

- 아직도 똑같은 단어가 문장에서 중요도가 다르다
- 아직도 긴 문서가 유리하다!

#### Term Frequency (TF)
- 특정한 문서가 단어가 몇번 존재하냐

- weighting scheme
  - binary
  - raw frequency
  - log
  - double normalization 0.5 : 단어가 존재하면 0.5로 시작하기
  - double normalization K : K 이상
  
#### Inverse Document Frequency (IDF)
- 문서에서 정말 의미 있는 단어냐! 모든 문서에 같은 단어가 존재한다면 패널티를 주자! ex) is, the, a, an

#### TF-IDF Weighting
- TF-IDF weighting scheme
  - TF-IDF(t,d) = TF(t,d) * IDF(t)

- Okapi BM25 is the most popular TF-IDF model
https://en.wikipedia.org/wiki/Okapi_BM25

#### Hamming Distance
- String Sequence, Bit 

ex) 1011101 vs 1001001
    C**G**GT vs C**C**GT
    **trip**le vs **ridd**le
    
#### Levenshtein distance
example : GUMBO vs. GAMBOL

- The minimum number of single-character operations
Dynamic Programming 


## 분류
### Categorical 
  - Jaccard, Dice, SMC
### Nummerical
  - Eucliddean, Manhattan, Supream
### document
  - cosine
### term(word)
- Edit Distance (Levenshtein distance)

### Correlation Analysis
#### Chi-squared Test
- 있다 없다. (Independent or not)
- Definition
관찰한 데이터에 대한 우리의 가설은 둘이 상관관계가 없다 라고 가정하고 0으로 가깝다면 독립, 크면 상관

- Degree of freedom 자유도
  - p-value is 0.05
  - 8.34 > 3.84 they are not independent.
### Numerical Analysis
#### Covariance
- 2개의 상관관계의 대한 값
- 있다 없다. 있으면 양수(positive), 음수(negative) , ==0
- ==0 이면 독립
- Positive, Negative의 크기가 크다고 해서 더 큰 상관관계가 있는게 아니다.

#### Pearson Correlation Coefficient
- 평균을 빼고 표준편차로 나눠준 정규화 값을 COV를 실행
- 그 뒤에 CCOV(A,B) ㄱㄱ

#### Rank Correlation
- 웹에서 요즘 많이 씀
- D1, D2, D3 등수를 맞추기!

#### Kendall's Tau Distance
- Rank inversion 
- 순위가 뒤집어진 갯수!
- O(n^2)

#### Spearman's Footrule Distance
- 등수 차이의 절대값

#### Kendall vs Spearman Relationship
- K(a) <= F(a) <= 2K(a)

