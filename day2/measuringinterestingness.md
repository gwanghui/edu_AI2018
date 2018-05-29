# Measuring Interestingness

## intuition 1
  - Rule : buying a game -> buying a video
  - support(G- > V) = P(G,V) = 0.4
  - confidence(G -> V) = P(G,V) / P(G) = 4000 / 6000 = 0.66
  - Video를 산사람이 75%인데 game이랑 같이 산게 66프로니까 negatively correlated

## intuition 2
  - Rule : buying a game -> not buying a video
  - support(G- > V) = P(G,~V) = 0.2
  - confidence(G -> V) = P(G,~V) / P(G) = 2000 / 6000 = 0.33
  - Video를 사지 않은 사람이 25%인데 game을 사고 비디오를 사지 않은 사람이 33프로니까 positively correlated
  
## Example of Interestingness Measure
  - Milk가 많은걸 고려하지 않은 confidence
  
## Discussion on Interestingness Measure
### X^2 
  - 카이 스퀘어도 봐줘야 한다.
### Lift
- lift(A,B) = P(A,B) / P(A)P(B)
- lift < 1.0 : Negatively correlated
- lift > 1.0 : Positively correlated
  
### Lift도 문제가 있습니다. !?
- Null-invariance : 우리가 보통 measuring을 할때 Independent한 데이터가 존재할때 lift의 값이 다르다!
- confidence는 Null-invariance한 measure가 된다. 어?
#### Four Null-Invariant Measure
- all_conf(A,B) = sup(AUB) / max{sup(A),sup(B)} = min{P(A|B),P(B|A)}
- max_conf(A,B) = max{P(A|B),P(B|A)}
- kulc (A,B) = 1/2 (P(A|B) + P(B|A))
- cosine (A,B) = sqrt(P(A|B) * P(B|A))

- if < 0.5 negatively
- if > 0.5 positively

#### Inbalanced Ratio
- IR = | sup(A) - sup(B)| / (sup(A) + sup(B) - sup(AUB))
