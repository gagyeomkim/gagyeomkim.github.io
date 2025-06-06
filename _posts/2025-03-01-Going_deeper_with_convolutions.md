---
title: "[GoogleNet 논문 리뷰] - Going deeper with convolutions"

categories: ["Computer Vision Paper Review"]
tags:
  - ["Deep Learning", "Paper Review", "Computer Vision"]
---

InceptionV1 paper : <https://arxiv.org/abs/1409.4842>

Inception을 위한 논문 리뷰 글입니다.

> 본 논문의 코드 구현은 [깃허브](https://github.com/gagyeomkim/paper-review/blob/main/GoogLeNet.ipynb)에서 확인 가능합니다.



## 1. Simple Introduction
Convolutional network의 발전으로 이미지 인식과 객체 탐지의 품질이 극적인 속도로 발전했다.

이러한 발전은 단지 더 좋은 하드웨어, 큰 데이터셋, 더 큰 모델 덕분이 아니라, 새로운 아이디어, 알고리즘, 그리고 향상된 network architecture 덕분이며, 이 논문에서의 architecture의 설계 또한, 알고리즘의 효율성(전력이나 메모리 사용의 측면에서)을 많이 고려하였다고 한다.

해당 논문에서는 Computer Vision에 대해서, Inception이라는 효율적인 deep neural network를 소개한다.



Inception이라는 아키텍처의 이름은 'we need to go deeper'이라는 인터넷 밈에서 유래하였는데, 여기서 deep은 아래와 같은 2가지 의미를 가진다.

1. 'Inception 모듈의 형태'의 새로운 수준의 구조를 도입한다.
2. network의 depth(레이어의 깊이)를 증가시킨다.

Inception 구조의 이점은 ILSVRC 2014에서 검증되었고, 당시 SOTA(state of the art)의 수준을 상회하는 결과를 보여주었다.

이제 Inception이 무엇인지 알고, 하나씩 살펴보자. 

## 2. Background Knowledge
> CNN을 정확하게 모르는 경우에도 이해할 수 있도록 기초적 개념들을 몇가지 다루었습니다.

### Image Data
신경망에 데이터를 이용하기 위해서는 이미지를 **정형 데이터화**하는 과정이 필요하다.
- 정형 데이터화: 컴퓨터가 식별가능한 형태로 데이터를 변환하는 것

이미지는 픽셀 단위로 구성되며, 각 픽셀은 RGB값으로 구성된다.(즉, 아주 작은 네모가 여러개 모여 이미지를 이루는 것으로 생각할 수 있다.) 

이미지를 정형 데이터화하는 방법 중 하나는 흑백 이미지의 경우 **'가로x세로'**에 흑색의 강도가 들어간 배열로, 컬러이미지의 경우 **'가로x세로x3'**의 배열로 나타낼 수 있고, 마지막 3에는 각각 R, G, B강도의 값으로 구성할 수 있다. 
![fig 1. 이미지 데이터](https://i.imgur.com/8wkKBNK.png)
>  fig 1. 이미지 데이터. 빨간색 동그라미 쳐진 부분이 Channel이다

이미지 해상도 외에 겹쳐지는 부분을 **Channel**(채널)이라고 하며, 흑백의 경우 1, 컬러의 경우 3이다.(RGBA의 경우 4이며 마지막 A는 이미지 밝기인 Alpha)

> kernel의 경우에는, Channel이 3이라고 해도 필터의 수가 3개인 것이 아니라, 1개임에 주의하자.

> 채널쪽으로 feature map이 여러개 있다면 입력데이터와 필터의 합성곱 연산을 채널마다 수행하고, 그 결과를 더해서 하나의 출력을 얻는다.

### CNN의 기본구조
LeNet-5부터 CNN은 일반적으로 표준 구조를 가진다.

![fig 2. CNN](https://i.imgur.com/ofmX12x.png)

CNN(Convolutional neural net)은 기본적으로 

**Convolutional layer - Pooling layer - FC(Fully Connected) layer** 순서로 진행된다,

#### Convolutional Layer 
입력 이미지를 특정 kernel을 이용하여 이미지의 특징들을 추출하고, 추출한 특징들을 Feature Map으로 생성함.
![fig 3 Convolutional Layer](https://i.imgur.com/h2LvQYZ.png)

#### pooling Layer 
이미지 크기(Feature Map)를 적절히 줄이면서 특정 Feature을 강조할 수 있음

#### FC Layer
Convolution/pooling 프로세스의 결과를 취하여 이미지를 정의된 라벨로 분류하는 데 사용함. 아래의 3가지 과정을 보통 FC Layer에서 수행함
1. 2차원 배열 형태의 이미지를 1차원 배열로 평탄화
2. 활성화 함수(Relu, Leaky Relu, Tanh,등)뉴런을 활성화
3. 분류기(Softmax) 함수로 분류

### NIN(Network in Network)
Network-in-Network는 신경망의 표현력(represnetational power)을 높이기 위해 제안한 접근 방식이다. 

Convolutional Layer에 적용할 경우, 이것은 활성화함수로 ReLU함수가 뒤따르는 추가적인 1x1 Convolutional Layer로 생각할 수 있다.

해당 논문에서도 이 방식을 아키텍처에 많이 사용했는데, 1x1 Convolutions를 2가지 목적으로 사용했다고 한다.
1. computational bottleneck을 제거하기 위한 dimension reduction modules
2. network의 크기를 제한하기 위해서

> 이것은 depth뿐 아니라 width(unit 수)의 증가를 성능 저하없이 이끌어냈다.

## 3. Method
### Motivation and High Level Considerations
deep neural network의 성능을 개선하는 가장 간단한 방법은 크기를 늘리는 것이며, 이것은 depth(the number of layer)와 width(the number of units)를 늘리는 것을 모두 포함한다.

그러나, 이 solution에는 2가지 결점이 존재한다.
1. 확대된 network가 overfitting되기 쉽다.
	- 큰 사이즈는 일반적으로 많은 수의 파라미터를 의미하기 때문
	- training set에서 label이 지정된 example의 수가 제한되어 있을 경우 그럴 가능성이 높아진다.
2. computational resources의 사용이 상당히 증가한다.
	- 2개의 convolutional layer가 결합된(나란히 놓여져 있는 경우를 말하는 거라고 판단함)경우, 필터 수의 증가는 계산의 2차적 증가를 가져옴
	- 대부분의 가중치가 0에 가까워지는 것 같이, 추가된 용량이 비효율적으로 사용되는 경우에는 많은 계산이 낭비됨

이를 해결하는 가장 근본적인 방법은 fully connected architecture로부터 **sparsely connected**(모든 unit이 서로 연결되지는 않은) architecture로 전환하는 것이며, 이것에 Hebbian 원리(함께 활성화되는 뉴런은, 서로 연결되어있다.)를 적용하여 이해할 수 있다고 한다.

이러한 아이디어를 바탕으로 다양한 spatial domain에서 sparsity가 활용되었다.

저자들은 Inception 아키텍쳐를 sparse structure를 근사화하고, 쉽게 사용할 수 있는 dense component에 의한 예상 결과를 평가하기 위해 시작했는데, hyperparameter를 조정하고 실험한 결과 꽤 좋은 성과가 나왔다고 한다.

### Architectural Details
Inception 아키텍쳐의 주요 아이디어는 convolutional network에서 **최적의 local sparse structure**를 쉽게 이용할 수 있는 **dense component**로 커버하는 방법을 찾는 것이었다.

translation invariance이 이루어져야한다는 것은, **우리의 네트워크가 convolutional building block으로 구축되어야함**을 의미한다. (CNN이 pooling layer를 거치게되면서 translation-invariance해지기 때문이다.)
- translation invariance : **input의 위치가 달라져도 output이 동일한 값을 갖는것을 말한다**



즉, **우리가 해야하는 것은 최적의 local 구성(convolutional 블록으로)을 찾고, 그것을 반복해서 쌓는 것**이다.

이 논문에서는 이전 layer(입력에 가까운 layer)의 각 unit(neuron)이 이미지의 일부 영역에 해당하고, 이러한 unit이 filter에 의해 그룹화된다고 가정한다. **즉, 많은 neuron의 무리(cluster)를 한 영역에 집중시켜서, 다음 layer에서 1x1 convolution layer로 덮을 수 있는 것이다.**



그러나, 이는 더 큰 patch로 갈수록, cluster가 넓게 퍼지지 않고, 점점 더 큰 영역으로 갈수록 patch의 수가 감소할 것으로도 예상할 수 있다.(patch의 크기가 커질수록 patch가 커버할 수 있는 영역이 넓어져 전체 patch의 개수가 줄어들기 때문)

또한, patch alignment 문제를 피하기 위해서 1x1, 3x3, 5x5로 필터크기를 제한했다.
- patch alignment 문제 : 필터 사이즈가 짝수일 때 patch(필터)의 중간을 어디로 두어야할지 결정해야함.  

이후에는, 모든 레이어와 각 filter bank(하나의 필터가 차지하는 영역을 뜻하는 용어로 이해하면 될 것 같다.)의 출력을 concatenate시켜 다음 단계에 대한 입력을 형성한다고 한다. 또한, pooling 연산이 당시 sota인 convolutional networks의 성공에 필수적이었기 때문에, 병렬 pooling을 추가하는 것이 추가적으로 이로운 효과를 가져올 것이라고 저자들은 판단했다.



이에 대한 Inception module의 초기 접근(naive version)은 아래 사진과 같다.
![fig 4. Inception module, naive version](https://i.imgur.com/5rqP4WJ.png)

naiver form의 문제는 5x5 convolution같은 적은 수 조차도, 많은 수의 필터가 있는 convolutional layer 위에 놓이면 엄청나게 expensive하다는 것이다.
(pooling unit이 추가되면 더욱 그러한데, pooling unit의 output filter수는 이전 stage의 필터수와 같다.)

convolutional layer의 출력과 pooling layer의 출력을 결합하면 각 stage의 출력 수가 불가피하게 증가하며, 조금의 stage 이후 계산량이 폭발적으로 증가하게 된다.

따라서, computational requirements가 너무 많이 증가할 수 있는 곳에 dimension reduction과 projection(고차원의 벡터를 저차원 공간에 투영하는 것)을 적용하는 것이 하나의 방법이 된다.

-> **즉, 1x1 convolution을 expensive한 3x3과 5x5 convolutions전에 사용해 계산 축소를 이끌어낸다.**

> 또한, 때때로 stride가 2인 max pooling을 사용했다고 한다.

이를 적용한 module 구조는 아래와 같다.
![fig 5. Inception module with dimension reductions](https://i.imgur.com/aScvg24.png)

학습 중 메모리 효율성과 같은 기술적 이유로 인해 하위 레이어는 기존의 합성곱 방식으로 유지하면서 상위 레이어에서만 inception 모듈을 사용하는 것이 유익해보인다고 한다.

---
#### 1x1 convolution 추가설명
그렇다면 1x1 convolution은 어떻게 dimension reduction에 영향을 주는 걸까?

1x1 convolution은 필터 사이즈가 1x1이라는 것을 의미하는데, 이는 feature map의 feature 하나(Image input 기준으로 픽셀 하나)에 대해 convolution 연산을 진행한다는 것을 뜻한다. 아래의 사진에서는 1x1x192의 convolution filter를 하나 통과시켰을 때의 결과를 시각화한 사진이다. 
![1x1 Convolution 예시 1](https://i.imgur.com/j9myqWp.png)

> ref : <https://sjkoding.tistory.com/73>

여기서 filter의 channel이 192라고 표현되어 있는데, 1x1 convolution을 진행할 때, channel의 수는 우리가 원하는 만큼 지정할 수 있다. 위 그림에서는 input의 채널이 192이므로, 만약 128채널로 줄이고 싶다면 1x1 convolution filter를 128개 만들어 연산하면 된다. 

밑의 그림에서는 이와 같이 1x1 convolution을 활용하여 우리가 원하는 수의 Channel 수를 가지는 층을 구성하고 있다. 또한, Channel 수의 조절이 계산량 감소에 직접적으로 영향을 주는 것도 확인할 수 있다.
![1x1 Convolution 예시 2](https://i.imgur.com/wihwRbK.png)

> ref : <https://sjkoding.tistory.com/73>
> 노란색 직육면체 부분이 1x1 필터를 한번 적용하는 부분이다.
> 28x28은 실제로 파라미터를 뜻하진 않지만, 계산의 편의성을 위해 사용했다.

빨간색 글씨는 해당 과정에서 사용되는 전체 파라미터의 개수를 의미하며, 1x1 convolution을 적용하지 않았을 때와 약 4배의 차이가 나는 것을 볼 수 있다.

이처럼, 1x1 convolution을 이용하여 feature map의 사이즈는 동일하게 출력하면서, parameter의 수를 획기적으로 감소시킬 수 있다.

### GoogLeNet
이제 Inception Module이 적용된 전체 GoogleNet 구조에 대해서 알아보자.

Inception module 내부를 포함해서, 모든 Convolution Layer에는 ReLU가 적용되어있다.
또한, receptive field의 크기는 224 x 224이며, RGB 컬러 채널을 가지고, mean subtraction을 적용한다.
![fig 6. GoogLeNet incarnation of the Inception architecture](https://i.imgur.com/OPc4GIZ.png)

'#3x3 reduce', '# 5x5 reduce': 3x3과 5x5 Convolutional Layer 앞에 사용되는 1x1 필터의 채널 수
pool proj : max pooling layer 뒤에 오는 1x1 필터의 채널 수

GoogLeNet을 4부분으로 나누어서 살펴보자.
> ref : <https://phil-baek.tistory.com/entry/3-GoogLeNet-Going-deeper-with-convolutions-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0>


#### part 1.
![fig 7. GoogLeNet part 1](https://i.imgur.com/NfoUThW.png)
part 1에서는 입력 이미지가 들어오는 부분이다. 위에서 설명했듯이, 효율적인 메모리 사용을 위해 낮은 layer에서는 기본적인 CNN 모델을 적용하고, 높은 layer에서 Inception module을 사용하라고 했던 조언을 따르고 있는 것을 볼 수 있다.

#### part 2.
![fig 8. GoogLeNet part 2](https://i.imgur.com/sw3btfx.png)
위에서 살펴본 Inception module이 적용된 것을 확인할 수 있다. 
#### part 3.
![fig 9. GoogLeNet part 3](https://i.imgur.com/RYT6XLZ.png)
part 3부터는 auxiliary classifier(보조분류기)가 적용되었다.

network가 상대적으로 깊은 깊이를 가질 경우, 모든 레이어를 통해 gradient를 backprop하는 능력에 문제가 생길 수 있다. 

논문의 저자들은 상대적으로 얕은 신경망의 강한 성능을 통해서 신경망의 중간 layer에서 생성된 특징이 매우 차별적이라는 것을 알 수 있었고, 이러한 중간 layer에 auxiliary classifier를 추가하면 중간중간에 결과를 출력해서 추가적인 역전파를 일으켜 gradient가 전달될 수 있게끔 하면서도 정규화 효과가 나타날 것으로 예상하였다고 한다.

> 지나치게 영향을 주는 것을 막기 위해서 auxiliary classifier의 loss에는 0.3을 곱하였고, inference time(추론)때는 auxiliary network를 삭제하였다.




#### part 4.
![fig 10. GoogLeNet part 4](https://i.imgur.com/Fst4OCK.png)

part 4는 예측 결과가 나오는 모델의 마지막 부분이다.

여기서 average pooling layer가 사용되었는데, 이는 GAP(Global Average Pooling)이 적용된 것이다. FC(fully connected)방식과 GAP 방식의 차이점은 아래와 같다.

![fig 11. GAP](https://i.imgur.com/Umrqfy0.png)
> ref : <https://bskyvision.com/539>

GAP는 이전 레이어에서 나온 feature map에서 각 채널마다 전체 평균을 구하고, 1차원 벡터로 만들어준다.

즉, i번째 feature map의 평균값을 구해서 i번째 출력 노드에 입력하는 것이라고 이해하면 된다.

이렇게 GAP를 이용하면 가중치의 개수가 상당히 많이 줄어들게 되고, GAP를 적용할 시 fine tuning이 쉬워진다.

## 4. Conclusions
Inception 구조는 **최적의 sparse structure를 dense 블록으로 근사화하였음**에 그 의의가 있다. sparse 구조를 근사함으로써, 연산량을 줄이고 성능을 높이는 결과를 가져왔다.

## Reference
- <https://yjjo.tistory.com/8>

- <https://ganghee-lee.tistory.com/43>

- <https://www.youtube.com/watch?v=uQc4Fs7yx5I&t=39s>

- <https://sjkoding.tistory.com/73>

- <https://hwiyong.tistory.com/45>

- <https://phil-baek.tistory.com/entry/3-GoogLeNet-Going-deeper-with-convolutions-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0>

- <https://bskyvision.com/539>