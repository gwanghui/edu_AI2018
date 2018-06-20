### Stacking Linear Functions
- Linear Function은 Layer를 깊게 쌓아도 수식을 풀어쓰면 1차로 표현 가능하기 때문에 깊게 쌓는게 의미가 없다.

### Adding Non-Linearity
- Linear layer의 아웃풋에 activation function을 조합함으로써 어려운 문제를 표현할 수 있을것 이다. 라는 가정
- 다루는 함수는 인풋 아웃풋은 어떻게 돌아갈지 모르지만 결정된거를 줘라
- Linear가 Optimal이면 Linear로 학습해
- 자유도 있는 학습

### Learning Deep Neural Network
#### Loss function
- 얼마나 네트워크가 잘 동작하고 있나
- Loss를 무엇무엇하게 할 것 (minimize, maximize, sometimes)
- Loss가 합리적으로, 미분가능하게 잘 디자인이 되어야 한다.
- Multiclass SVM Loss for classification : http://ishuca.tistory.com/378

#### Auto Encoder
- input과 Output의 형태가 같은 결과를 도출
- ex) input 100dim , output 100dim, layer1 < 100dim , layer2 < layer1 , layer2 < layer3
- encoding 이 중요하다
  - ex) 100dim의 데이터를 20dim으로 줄인 encoder가 전체를 잘 표현할 경우 20dim만 보전하면 된다는 결과를 얻는다.
- 학습을 통해서 정말 필요한 데이터만 뽑아 내기 위해서 필요하다.  
- L2 Loss가 0이 되게끔 만드는게 Auto Encoder이다.
  
#### optimizer
- (Stochastic: 확률적) Gradient Descent : Mini Batch로 Local optimal 구하면 Global 접근 가능?
- Momentum , NAG, Adagrad, Adadelta, Rmsprop
- Local minimum으로 안빠지게 하는 기법들을 추가
- 속도 ?
- ADAM optimizer를 자주 씀

### Deep Learning
- complex model is vulnerable to overfitting.
- in order to prevent overfitting
  - 데이터를 더 준다....
  - Network를 더 간단하게 만든다.
  - Batch normalization : 한 배치 안에서 Intra variation을 방지하기 위해
    - ex) 하나의 배치 안에서 너무 밝은 이미지만 분석
  - drop out : 매번 임의로 nn에서 노드를 지워 보는 것, 특정 뉴런과 특정 인풋에 완벽하게 종속 되는것을 방지
  - data argumentation : 밝기를 바꾸고, Rotation, local crop 등등
  - regularization : Weight들이 너무 크거나 너무 작지 않게 만드는 것
    - ex) w1x1 + w2y2...  w1이 너무 크면 x1에 영향을 많이 받기 때문에 weight들을 조정
    
  
  
