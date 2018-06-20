# CNN
- CNN에서 Feature Learning이 이루어질때 초반에는 작은 단위로 보게 된다.
- CNN뒤로 갈수록 중첩이 되기 때문에 큰 단위로 보게 된다.
- Local 한 부분부터 Global 한 부분으로 Feature를 모은다.

## 왜 Hot 해졌을까
- 1980s 에 CNN 개념이 나왔는데 AlexNet이 2012년에 나왔다!
- Data, Computing Power(GPU)

## Main components of CNN
- convolutaion layer
- activation function
- pooling layer

### convolution layer
- Sliding window
- 32 * 32 * 3 image , 5 * 5 * 3 filter *2
- 초반, 중반, 마무리 Convolution 
- 독립적으로 Local feature를 추출한다.

#### Forward pass

#### Backpropagation
- 각각의 Weight가 Loss를 줄이는 방향으로
- 모든 Weight에 대한 기울기를 편미분해서 구해놓는다.
  - 1. weight update
  - 2. 다음 뉴런에 gradient 전달
  
#### Spatial dimension
- stride가 1일 때 7(N)*7 3(F)*3 -> 5*5
- ((N - F)/ stride) +1
- 1 * 1 convolution layer는 reduces the number of channells

#### Activation function
- sigmoid, tanh, ReLU, Leaky ReLU(평균이 0이 안된다!)
- Maxout , ELU(적당한 알파를 찾아서 평균을 0으로 만드는..)
