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
  
