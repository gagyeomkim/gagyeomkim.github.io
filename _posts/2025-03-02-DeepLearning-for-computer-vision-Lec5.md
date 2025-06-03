---
title: "[EECS 498-007] Lecture5: Neural Networks"

categories: ["EECS 498-007 Deep Learning for Computer Vision"]
tags:
  - ["Deep Learning", "Computer Vision"]

date: 2025-03-25
last_modified_at: 2025-03-25
---


![](https://i.imgur.com/qzjhrka.jpeg)
Lecture 5에서는 Neural Networks(신경망)에 대해서 배운다. 
![](https://i.imgur.com/OpflrhC.jpeg)
지난시간까지 우리는 다음과 같은 내용을 다루었다.
1. image classification problem을 해결하기 위한 Linear Model
2. Loss Function을 통해 가중치 선택에 대한 우리의 preference를 표현하는 방법
3. Loss function을 minimize하고 model을 훈련시키는 SGD방법

이제 1번에서 등장한 linear model로 다시 돌아가서, classifier의 관점으로 Linear model에서부터 Neural Network 모델이 만들어지는 과정을 살펴보려고 한다.

> 5장 목표 : linear model의 성능이 나쁜 것을 neural networks를 사용하여 해결하고자 한다.

![](https://i.imgur.com/EDSPthJ.jpeg)
우리는 Linear Classifier가 가지는 한계점에 대해서 다루고, Linear Classifier의 Geometric View와 Visual Viewpoint를 배웠었다.

Geometric Viewpoint에서 Linear Classifier는 high dimensional한 (feature의) 공간을 분리하는 hyperplane을 만드는 것이었다. Linear Classifier로는 좌측의 그림과 같은 파란점과 초록점을 선형으로 분류하는 것이 불가능하다.

Visual Viewpoint에서 Linear Classifier는 클래스 당 하나의 템플릿만 학습할 수 있었기 때문에, 실제로 같은 class여도 정해진 템플릿과 비슷한 생김새가 아니라면 그 class로 인식할 수 없게 된다. 즉, 같은 물체에 대한 multiple mode를 표현할 수 없었다.(이를 "표현력에 한계가 있다"고 말한다.)

Neural Network를 바로 보기 전에, 이러한 Linear Classifier의 한계를 해결할 수 있는 방법들에 대해서 조금 알아보자.
## Feature transform
![](https://i.imgur.com/6Fvhmab.jpeg)
첫번째는 Feature transforms이다. 
- Feature transform : Original Space의 input들을 classification하기 위해서, feature를 어떤 수학적 공식으로 변환하는 것

Feature transform를 이용하면 Linear classifier로도 non-linear한 decision boundary를 만들 수 있다.

위와 같은 예시는 직교 좌표계를 극 좌표계로 변환하면 classification의 목적에 더 잘 맞는 데이터로 표현할 수 있다는 것을 나타낸다.
![](https://i.imgur.com/q6W6K3m.jpeg)
직교 좌표계에서 극 좌표 변환을 통해 만든 Feature Space에서는 두 종류의 point들이 linear 하게 잘 나뉘게 된다.
![](https://i.imgur.com/ebyDvzw.jpeg)
이후 linear classifier를 만들고

(->$f(x,W)=0$의 함수를 만들고)
![](https://i.imgur.com/cShj5j5.jpeg)
그 linear classifier를 Original Space로 가지고 오면 nonlinear decision boundary를 만들 수 있게 된다.

즉, feature space의 linear decision boundary가 original space의 nonlinear decision boundary에 대응된다.

이처럼 data에 잘 맞는 변환을 적절하게 선택하면 linear classifier로도 충분히 nonlinear decision boundary를 만들 수 있게 된다.

feature transform을 위해서는 두가지 조건에 대한 고려도 이루어져야한다.
1. 데이터의 구조를 파악하고 있어야함
2. 데이터에 맞는 feature transform을 알고 있어야함

![](https://i.imgur.com/cGYis8v.png)
## Feature transform in Computer Vision
### Color Histogram
이러한 feature transform에 대한 컴퓨터 비전의 예시 중 하나는 **Color Histogram**이다.

***Corlor histogram 방법***
1. RGB feature space를 discrete(이산적인) bins의 개수로 나눈다.(스펙트럼화)
2. 사진의 픽셀 각각이 어느 color bin의 위치에 있는걸 확인한다.
3. pixel들의 color 분포를 histogram으로 표현한다.(feature representation)

Color Histogram방법은 그림의 질감과 이미지의 공간적 정보(spatial position)은 무시하고, 색깔 정보에만 초점을 맞춘다.

-> 이러한 점 때문에 spatially invariant한 특징을 가진다고 한다.(개구리가 이미지 어디에 있든지 상관없이 색상만 고려함)
### HoG
![](https://i.imgur.com/CExTWHS.jpeg)
또 다른 feature transform 방법은 Histogram of Oriented Gradients(HoG)이다. HoG는 Color histogram과 반대의 방식이라고 할 수 있는데, 색상의 정보를 사용하지 않고, local edge(각 region 내 edge)의 방향과 강도에 관한 정보를 사용한다.

![](https://i.imgur.com/Xz3Rlds.jpeg)

***HOG 방법***
1. 각 픽셀의 edge 방향과 강도를 계산한다.
2. 이미지를 8x8 크기로 잘라서, edge detection을 진행한다. (HoG는 매 input image마다 edge의 방향과 강도를 알려준다.)

HoG는 각 input image마다 edge의 방향(direction)과 강도(strength)를 알려준다고 한다. 
![](https://i.imgur.com/MysbwGr.jpeg)
예시를 보면 블러처리가 된 배경 부분에는 edge가 약하게 표현되며, 빨간 박스 부분은 대각방향의 edge가, 개구리의 눈 부분은 모든 방향의 edge가 표현된 것을 볼 수 있다.

![](https://i.imgur.com/EUuChtP.jpeg)
> 이러한 feature representation에서 연구자들이 고려해야하는 것은 아래와 같은 것들이 있다고 한다.
> - feature representation을 통해서 무엇을 얼마만큼 capture 하기를 원하는지
> - 어떠한 유형의 feature representation을 디자인 할 수 있는지

### Bag of Words
![](https://i.imgur.com/lrFwkkR.jpeg)
data-driven한 방식의 feature transform 방식도 존재하는데, **Bag of Words**라고 불리는 방법이다.

1.주어진 image를 랜덤한 크기와 비율의 patch로 나눈 후, 이들을 몇개의 cluster(무리)로 묶는다. 

- 이러한 cluster를 **codebook** 혹은 set of visual words라고 부르며, 우리의 이미지 속에 어떤 feature가 표현되어야하는지를 나타낼 수 있다.

> Bag of Words의 아이디어는 많은 training data 속에서 흔하게 관찰되는 feature structure가 있는데, 이것을 통해서 일종의 visual word representation을 학습할 수 있다는 것이다.

![](https://i.imgur.com/AG08gb4.jpeg)
2.학습된 codebook을 통해서 이미지를 인코딩한다. 

이것은 일종의 histogram representation을 계산하는 것인데, input으로 사용된 이미지에서 미리 학습된codebook이 얼마나 나타났는지를 확인하는 것이다.

즉, 입력된 이미지에서 **Visual Word**가 얼마나 등장하였는지 살피는 것과 같다.

![](https://i.imgur.com/Qoj2BND.jpeg)
Image feature를 만들 때 여러개의 방식으로 feature transform을 한 뒤, 이것들을 모두 조합해서 long-high dimmensional feature로 만드는 방법도 존재한다. 예를 들면 color histogram-Bag of Words - HoG 순서대로 feature를 만들고 연결할 수 있다.
![](https://i.imgur.com/zvQQiq8.jpeg)
2011년 ImageNet 챌린지를 살펴보면, 앞서 소개한 방식으로 다양한 feature들을 결합하여 사용한 우승자가 있는 것을 확인할 수 있다.
![](https://i.imgur.com/FPpCe1C.jpeg)
machine learning 파이프라인의 관점에서 지금까지 배운 걸 다시 한번 더 생각해보자.

image classification은 다음의 2가지 부분으로 나뉜다고 할 수 있다.

1. feature extraction
2. 추출된 feature를 학습한 model을 통한 classification

하지만, feature extraction 단계가 Bag of words 방식처럼 data-driven 하더라도, feature 자체가 그들의 가중치를 스스로 업데이트 하는 것은 아니다.(=tuning이 안된다.)

즉, 지금까지의 feature는 전체 loss를 minimize하려는 목표로 변환된 것이 아니라, 그저 이미지를 잘 표현할 수 있는 feature들을 추출한 것 뿐이다.

![](https://i.imgur.com/lpGqkU3.jpeg)
Neural Network의 motivation 중 하나가 바로 이러한 점으로부터 왔다. 입력부터 끝까지 스스로 학습하는 end-to-end 파이프라인을 만드는 것이다.
- end-to-end : 처음부터 끝까지 라는 의미로. 데이터(입력)에서 목표한 결과(출력)을 사람의 개입없이 얻는다는 뜻을 가지고 있다.

즉, Neural Network를 사용하면 feature extraction과 분류를 모두 해낼 수 있다. 이때 중요한 점은 **전체 loss를 최소화하는 방향으로 스스로 feature extraction 방법을 학습한다**는 것이다.
## Neural Networks
![](https://i.imgur.com/2Yzbqx7.jpeg)
지금까지의 linear score function은 $f(x,W)=Wx$의 식을 가지고 있었다.
- $x$: column vector로 input data의 정보가 담겨져 있다.
- $W$ : learnable matrix

![](https://i.imgur.com/upCBpqG.jpeg)
가장 간단한 형태의 Neural Network인 2-layer Neural Network에 대해 먼저 알아보자. 
- $C$ : class 개수
- $H$ : hidden size; 표현하고자 하는 feature representation의 차원 수
- $W_1x$ : matrix-vector multiplication(행렬-벡터의 곱셈)
- $max(0,W_1x)$ : 0과 elementwise하게 max
	- $W_1x$의 음수 값은 0으로 대체
	- shape : $H$x1
- $W_2max(0,W_1x)$ : matrix-vector multiplication

$H$의 shape을 가지고 있는 weight matrix $W_1$이 있고, 

$W_1$과 column vector $x$와 multiplication을 진행한 vector와 $W_2$간에 matrix-vector multiplication이 다시 일어나는 것을 알 수 있다.

> Bias도 삽입될 수 있지만, 간단하게 하기 위해서 해당 예시에서는 생략했다.



![](https://i.imgur.com/u9Ha66K.jpeg)
더욱 복잡한 Neural Network라고 해도, 선형 함수와 비선형 함수의 합성 함수를 계속 반복해서 적용할 뿐이다.

![](https://i.imgur.com/uiHBJe8.jpeg)
이 과정을 시각화 해서 나타내면 다음과 같다.

가장 좌측에는 input image를 flatten한 (3072, )차원의 column vector x가 존재하고 / 중간에는 100차원을 가지는 hidden vector h가 / 마지막에는 10개의 카테고리에 대한 score를 담고 있는 output vector s가 존재한다.

이때, layer란 **실제로 가중치를 갖는 layer**를 뜻한다.

즉, x와 h 사이, h와 s 사이이므로 2-layer neural network라고 부른다.

![](https://i.imgur.com/PXCI1hA.jpeg)
Weight matrix는 이전 단계의 layer가 다음 단계의 layer에 얼마나 많은 영향력을 끼치는지를 나타낸다.
ex. 
- 성분 $(w_1)_{ij}$는 h의 성분 $h_i$에 대해서 성분 $x_j$가 얼마만큼 contribute 했는지를 결정한다.
- 성분 $(w_2)_{ij}$는 s의 성분 $s_i$에 대해서 성분 $h_j$가 얼마만큼 contribute 했는지를 결정한다.

![](https://i.imgur.com/DE5AmRV.jpeg)
즉, 다른 element들까지 생각해보면, x의 모든 element가 h의 모든 element에 영향을 미치게 된다. 또한, h의 모든 element도 s의 모든 element에 영향을 미치게 된다.

일반적으로 이러한 dense connectivity한 패턴을 신경망에서 **Fully-connected Neural network**라고 일컫는다.

- Fully-Connected Neural Network : 이전레이어의 모든 벡터들이 다음 레이어의 모든 벡터에 영향을 미친다.

> 일부 문헌에서는 퍼셉트론의 관점에서 Multi-Layer Perceptron(MLP)라고 부르기도 한다.



![](https://i.imgur.com/607DoYA.jpeg)
지금까지의 Linear Classifier는 각 클래스 당 1개의 템플릿만을 학습할 수 있었다.
- $w$의 row 하나(class 하나에 대응됨)를 그림처럼 나타낸 것이 template이다.

![](https://i.imgur.com/v7PWQYV.jpeg)
$w_1$ matrix 또한 역시 어떤 하나의 템플릿을 학습한다. 

위에 보이는 템플릿 이미지는 알아보기 쉽지 않아서 이것이 어떤 것을 예측할지를 알아내긴 어렵지만, 2-layer neural network system에서 첫번째 layer는 클래스별로 무언가 구별되는 어떤 **구조적인** 템플릿을 학습한다는 것을 알 수 있다. 

![](https://i.imgur.com/5chZPow.jpeg)
가끔 빨간색 박스처럼 뚜렷하게 class의 feature가 구별되는 것을 볼 수 있다.

(아마도, 3강 예시에서의 오른쪽을 바라보는 말과 왼쪽을 바라보는 말처럼 보인다.)

이것은 앞서 알아본 linear classifier의 한계점이기도 했다.
![](https://i.imgur.com/kPLNkpd.jpeg)
이후, 두번째 layer($W_2$)는 앞선 첫번째 layer($W_1$)의 정보들을 조합하여 class score를 만든다. h 하나를 만들기 위해 input에 템플릿($W_1$)을 매칭한 후, h와 새로운 템플릿($W_2$)을 다시 한번 매칭하는 것이다.

즉, 오른쪽과 왼쪽을 보고 있는 말의 **정보를 조합한 하나의 template을 만드는 것**이며, 레이어가 2개가 되면, horse에 대한 템플릿이 여러 개일 때, 이를 조합할 수 있게 된다.

이것은 때로 **distributed representation**이라고도 한다.
- distributed representation : 템플릿들도 말의 예시는 알아보기는 쉬웠지만, 대부분의 박스친 부분들(개별 템플릿)은 해석하기가 어렵다. 템플릿은 단지 어떠한 구별되는 spatial(공간) 정보만을 distributed 하게 담고 있는 representation이다.

![](https://i.imgur.com/xq5GH51.jpeg)
다음은 layer가 6개인 6-layer Neural Networks를 그림으로 나타낸 것이다.

**Depth** : layer의 수(=**weight matrix $W$의 개수**)

**width** : 각 layer의 사이즈; hidden layer의 dimension에 대응; 노드의 개수

### Activation function
![](https://i.imgur.com/dZcpn9k.jpeg)
신경망의 수식을 보면 max함수가 있다. 사실 이 함수는 **ReLU(Rectified Linear Unit)**라는 함수이며, 이는 **activation function**(활성화 함수)라고 불린다. 

$$ ReLU(z) = max(0, z)$$

$$f=W_2 \cdot max(0,W_1x)=W_2\cdot ReLU(W_1x)$$

활성화 함수로 ReLU를 사용할 경우, 양수는 그대로 살리고 음수인 부분은 모조리 차단하는(값을 0으로 만드는) 효과를 갖는다.

> ReLU에서 Rectified란 '정류된'이라는 뜻이다. 정류는 전기회로쪽 용어로, +/-가 반복되는 교류에서 -흐름을 차단하는 회로이다. 아래의 그래프와 비교하면 x가 0 이하일때를 차단하여 아무값도 출력하지 않는(0을 출력하는)것으로 이해할 수 있다. 단, 이런 유래에 대해 집중하기보다는, 개념에 집중하자.



![](https://i.imgur.com/99yu6gv.jpeg)
Neural Network에서 Weight matrix W외에 **nonlinear한 activation function을 두는 이유**는 뭘까? 만약 activation function이 없다면 수식은 다음과 같이 나타난다.

$$s=W_1W_2x$$

이러한 matrix-matrix multiplication은 서로 연관되어 있기 때문에 2개의 matrix를 하나의 그룹으로 묶어서 계산할 수 있다. 즉, x에 대한 선형 변환일 뿐이며, **activation function이 없다면 신경망이 그저 Linear Classifier로 남게된다.** 

따라서, ReLU와 같은 non-linear <font color="#c00000">activation function이 있어야</font> 신경망에 어떠한 **추가적인 표현력**(additional representation power)이 생기게 된다.

![](https://i.imgur.com/yMfRIVQ.jpeg)
ReLU뿐 아니라 다양한 activation function이 있다.
![](https://i.imgur.com/uL8Sjxs.jpeg)
오늘날에는 **ReLU**를 default choice로 선택하는 것이 가장 일반적이다.
> ReLU는 도함수를 취하면 negative 값 -> exactly 0 / positive 값 -> exactly 1이므로 미분 계산이 편하다.
> 
> 역사적으로는 sigmoid, tanh를 많이 사용했지만, numerical problem이 발생하기도 하고, 컴퓨터로 계산할 때 gradient vanishing(기울기소실) problem이 발생하기도 한다.



![](https://i.imgur.com/paBHpJi.jpeg)
실제로 Neural Network를 파이썬 코드로 짜는 것은 매우 긴 코드를 요구하지는 않는다.

Initialize weights : 맨 처음의 초기점을 무엇으로 할건지 정해주는 단계.

-> w에 대한 loss의 함수에서, 한 점에서 시작하여 global minimum을 찾아 내려갈 때, 이 시작점을 구하는 단계이다.

Data : data 로드

1. weight를 10000번 업데이트하고, 그 동안 loss를 계산한다.
  - 여기서는 sigmoid activation function과 L2 loss를 이용하였다.
2. gradient를 계산한다.
3. SGD로 W를 업데이트한다.



![](https://i.imgur.com/Zo3S3jn.jpeg)

## Biological vs Artificial neuron
biological neuron과 artificial  neuron에 대해서 조금 다루어보자.

![](https://i.imgur.com/lrug92O.jpeg)
우리의 뇌세포인 뉴런은 축삭돌기 / 축삭 / 수상돌기로 이루어진다.

중앙에는 cell body가 있고, Axon(축삭)은 전기 자극의 통로 역할을 한다.

전기자극은 Dendrite(수상돌기)를 통해서 받아들여진다.
![](https://i.imgur.com/LqZIyJG.jpeg)
여러개의 뉴런이 **Axon을 통해 전기 자극을 흘려보내고**, Dendrite를 통해서 받게 된다.

뉴런의 Presynaptic terminal(축삭돌기)와 다른 뉴런의 Dendrite간의 간극을 **Synapse**(시냅스)라고 부른다. 시냅스는 Axon으로부터 흘러들어온 전기적 자극을 Dendrite를 위해 모듈화하거나 변환한다.
![](https://i.imgur.com/xV89DSZ.jpeg)
모아진 전기적 자극이 뉴런의 중앙인 cell body에 모이고, 일정한 수치값을 넘으면 downstream 뉴런으로 또다시 자극을 흘려보낸다. 
![](https://i.imgur.com/P5Aspdf.jpeg)
이때, 신경망에서 firing rate라는 개념을 사용한다.

- **firing rate** : 활성화 속도(뇌의 뉴런이 전기적 자극을 생성하는 속도)
- 받은 자극을 그대로 전달하지 않고, 특정 수치값(그림 상에서는 y=0.5정도)를 넘기면 자극이 전달되는 것을 확인할 수 있다.

![](https://i.imgur.com/CaqDeG2.jpeg)
Biological neuron의 이 현상을 수학적으로 나타낸 것이 (Artifical) **Neural** **network**이다.

input된 전기 자극인 $x_0$이 시냅스를 통해 신호가 조금 바뀌어 $w_0 * x_0$가 출력된다.

cell body에서는 모은 정보를 종합하고, activation function인 $f$를 적용하여 다음 뉴런으로 전달해준다.

![](https://i.imgur.com/Pv0OdEy.jpeg)
하지만, 생물학적 뉴런은 인공신경망과 달리 뉴런 사이가 매우 복잡한 것을 알 수 있다. 즉, 인공신경망은 생물학적 뉴런의 매우 단순화된 버전이다.
![](https://i.imgur.com/SpjLDgM.jpeg)
이러한 생물학적 뉴런을 모방한 random connection을 도입한 neural network도 연구되고 있다고 한다.
![](https://i.imgur.com/CaCTsDs.jpeg)
해당 강좌에서 생물학적 뉴런에 대한 정보를 조금 소개했지만, neural network를 구축할 때, 이 내용을 너무 진지하게 받아들이진 않아도 된다고 한다.
## Space Warping
![](https://i.imgur.com/05rtdPT.png)
Neural Network가 강력한 이유는 앞서 알아본 distributed representation이란 특징 외에도, Space Warping이란 개념 때문인 것도 있다.

우선, $h=Wx$라는 선형 변환과, $x_1$, $x_2$로 이루어진 2차원 공간에 대해서 알아보자.

이 경우, 각 linear classfier는 하나의 hyperplane이 된다.
![](https://i.imgur.com/epaDhSk.png)
$x_1$,$x_2$는 선형변환 $h=Wx$에 의해서 $h_1$, $h_2$로 축이 옮겨지게 되고(기저 변환), Original space에서 두 직선으로 나눠진 4개의 공간을 rotate하는 효과를 가져오게 된다.

![](https://i.imgur.com/thEEatj.jpeg)
이번에는 Original space에서 두 직선으로 나누어진 4개의 공간에 data point들이 그림처럼 분포해있다고 생각해보자.
![](https://i.imgur.com/cuXKtVc.jpeg)
동일한 방식으로 Feature transform을 적용해도($Wx$로 변환), linear classifier로는 좋은 boundary를 만들 수 없다.(linearly separable하지 않다.)

![](https://i.imgur.com/c1irV06.png)
따라서, ReLU함수를 적용해보자.

ReLU는 음수 값을 0으로 만들기 때문에, 각 사분면마다 다르게 적용된다.

![](https://i.imgur.com/5X4W4ZC.png)

- A 영역 : A 공간의 data point들은 h1, h2축에서 모두 양수값이므로 그대로 둔다.
- B 영역 : B공간의 data point들은 h1축 기준 negative, h2기준 positive 값이므로, h2축으로 붙는다.
	- $(-, +)$ -> $(0, +)$
- D 영역 : D 공간에 있는 data point들은 h1축 기준 positive, h2 기준 negative 값이므로, h1축으로 붙는다. 
	- $(+, -)$ -> $(+, 0)$
- C영역 : C 공간에 있는 data point들은 h1, h2축 기준 모두 negative 값이므로 원점으로 붙게 된다.
	- $(-, -)$ -> $(0, 0)$

![](https://i.imgur.com/grolBRJ.jpeg)
다시 돌아가서, 선형변환을 한 이후, **ReLU**를 적용하면 이들이 어떻게 변환되는 지 살펴보자.
![](https://i.imgur.com/jmFq9yQ.jpeg)
ReLU를 적용하면 Linear Classifier를 통해 classification할수 있는 것을 확인할 수 있다.  

![](https://i.imgur.com/B4E5Aoe.jpeg)
feature space에서 linear classifier를 찾고, 다시 Original space로 돌아가면,  decision boundaryrk non-linear 하게 형성되며, 이는 ReLU를 기반으로 한 Neural Network Classifier에 대한 설명이 된다.

즉, activation function의 중요한 역할은 2가지가 있다.
1. 다양한 함수 표현력
2. Space Warping

![](https://i.imgur.com/HF7JtAK.jpeg)
만약 Neural Network의 hidden unit(neuron)의 수를 계속 증가시키는 경우에는, 선형 변환을 거친 feature space에서 영역을 가르는 decision boundary가 늘어나고, 이를 다시 원래의 input space로 가져오면 매우 복잡한 non-linear decision boundary에 대응된다.

즉, Hidden layer, neuron의 개수가 많아지면 꺾일 수 있는 방향이 많아지고 더 유연한 decision boundary를 갖게 된다.

![](https://i.imgur.com/ujHIsSr.jpeg)
하지만, 이것은 그렇게 좋은 아이디어가 아니며 일반적으로는 Regularization의 parameter를 조절하는 방법을 사용한다. $\lambda$의 세기를 조정함으로써 decision boundary의 complexity를 조정할 수 있게 된다.

## Universal Approximation
![](https://i.imgur.com/nPZbyuE.jpeg)
이처럼 Neural Network는 linear classifier와 달리 매우 복잡한 decision boundary를 만들 수 있다. Neural Network의 이러한 성질을 **Universal Approximation**이라고 한다.

Universal Approximation의 아이디어는 단 하나만의 히든 레이어로 어떠한 함수의 형태든 근사할 수 있다는 것이다.

arbitary precision으로 근사한다는 것은
$|f(x;W)-f^*(x)|$의 값(true root와 prediction의 차이, 즉 error(오차))이 우리가 정한 허용범위인 $\epsilon$보다 작다는 것을 의미한다.
![](https://i.imgur.com/spVzJfZ.jpeg)
ReLU를 기반으로한 신경망에서 Universal Approximation이 어떻게 작동하는지를 살펴보기 위해서 간단한 예시를 들어보자.

- input : single value
- output : single value
- hidden layer : 3개의 unit(neuron)

> unit은 neuron과 같은말이다.

![](https://i.imgur.com/6huPg6O.jpeg)
output인 y를 표현하면 다음과 같다. 이는 weight matrix에 의해 scale되고, bias에 의해 shift된 합으로 나타난다.
![](https://i.imgur.com/rwchfiJ.jpeg)
또한, 이에 ReLU함수를 적용하게 되면 기하학적으로는 오른쪽과 같이 나타날 것이다. 

![](https://i.imgur.com/AKKVHIk.jpeg)
이를 활용하여, 우리는 ReLU함수 4개를 조합함으로써 bump function이라는 형태를 만들 수 있게 된다.
![](https://i.imgur.com/xEa14TC.jpeg)
bump function의 경사 $m1, m2$를 위와 같이 가정할 경우 아래와 같이 4개의 형태를 조합할 수 있다.
![](https://i.imgur.com/L6cM69p.jpeg)

![](https://i.imgur.com/ViCgHlw.jpeg)

![](https://i.imgur.com/Z6QWPYj.jpeg)

![](https://i.imgur.com/3lUf8ha.jpeg)

![](https://i.imgur.com/vuG9jDI.png)
$f^*(x)$(true function)가 단조증가 함수일때, hidden layer를 증가시켜서 $w_1$(1층의 weight)과 nonlinear activation으로 계단 하나하나를 만들고, $w_2$(2층의 weight)를 통해 계단을 모으도록 디자인할 수 있다. 이 경우, $f^*(x)$를 근사할 수 있게 된다.


![](https://i.imgur.com/FkUFnWQ.png)
관련해서 궁금증이 생기는 부분은 아래의 reference를 참고하자.
> [reference](https://neuralnetworksanddeeplearning.com/chap4.html)



![](https://i.imgur.com/0einP4g.png)
Universal approximation은 neural network 기반인 $f(x;W)$가 x의 함수로써 매우 다양한 형태의 함수를 가질 수 있음을 증명해낸다. 즉, $f^*(x)$가 어떤 형태이든, $f(x;W)$로 매우 근사치를 만들어낼 수 있다.

단, 한계점이 존재한다.
1. 근사할 수는 있지만, 어떻게 근사할지에 대해서는 말해주고 있지 않다.
2. $f^*(x)$를 근사하기 위해, 얼마나 많은 training 데이터가 필요한지에 대
해서도 언급하지 않는다.

universal approximation은 단지 어떠한 함수를 근사할 수 있는 weight의 setting이 있다는 사실을 증명할 뿐이다.

> KNN역시 universal approximation의 성질을 가지고 있었다.

## Neural Network는 대부분 convex하지 않다!
![](https://i.imgur.com/llQ3fw2.jpeg)
Neural Network의 Optimization의 과정에 대해서 살펴보자.

Neural Network가 최적 해를 찾는 방법에는 convex function을 사용할 수 있다.

input은 n차원의 벡터이고, output은 scalr이다.
![](https://i.imgur.com/2200jU5.jpeg)
convex function이 말하는 바를 기하학적으로 이해해보자.

$f(x)=x^2$이라고 가정한다.
![](https://i.imgur.com/BzXATZb.png)
$tx_1$+$(1-t)x_2$ : $x_1$과 $x_2$사이의 선분

부등식의 왼쪽 부분은 두 벡터에 의한 선형 조합이 input으로 들어가며, 어떠한 함수의 chunk(부분; 곡선의 일부분)를 나타낸다.


부등식의 오른쪽은 함숫값의 선형 조합으로, 위와 같은 직선(secant line)을 의미한다.

**즉, 임의의 $x_1$, $x_2$를 찍은 후에, 주어진 곡선이 두 점의 평균변화율보다 항상 아래에 있다면 해당 함수는 convex함수라고 판단할 수 있다.**
![](https://i.imgur.com/yUyWo6Z.png)
cos함수처럼 적어도 하나의 $(x_1, x_2)$에 대해서 두점을 연결하는 선분이 주어진 곡선보다 위에 있다면, 해당 함수는 convex함수가 아니다.

![](https://i.imgur.com/JdLqLCG.jpeg)
이를 일반화해서 고차원에서 살펴보면, convex function은 bowl shape을 갖게 되며, 동일한 성질이 성립한다.

> 따라서 다차원 함수에 대해서도 똑같이 convexity를 확인할 수 있다.

![](https://i.imgur.com/rSul4Rd.jpeg)
convexity를 만족하기 위해서는 함수 자체가 bowl shape으로 생겨야한다. 이는 **<font color="#c00000">최솟값이 단 1개여야한다는 뜻이다.(global minimum이 1개이다; local minimum = global minimum)</font>**

따라서, convex function은 경사를 따라 내려가다 보면 global minimum에 반드시 도달하게 된다는 특징이 있다.

> Gradient는 local property, Convexity는 global property이다. 따라서 Gradient descent알고리즘이 보장하는 것은 local minimum일 뿐이지만, 만약 f라는 함수가 convex함수라면, 전체 함수가 bowlshape임을 convexity가 보장한다. 즉, 해당 상황에서는 **local minima가 곧 global minimum**이다.

![](https://i.imgur.com/YsG33iH.jpeg)
우리가 지금까지 배운 linear classifier가 사용하는 Softmax와 SVM Loss는 convex optimization의 주체이다. linear classifier는 Convex하기 때문에, linear model이 자주 사용되는 것이다.
![](https://i.imgur.com/dexgYkZ.jpeg)
neural network에서는 불행히도 loss surface에 대해서 이러한 수렴을 보장하지는 않는다. 예를 들어 neural network에 대한 고차원의 loss surface 중 일부는 convex하게 보이는 형태를 가지고 있지만
![](https://i.imgur.com/QNHlCCl.jpeg)
일부 loss surface의 단면은 nonconvex하다.
![](https://i.imgur.com/CjYP9qT.jpeg)
또한, 정반대의 loss surface를 가지고 있기도 하다.
![](https://i.imgur.com/svoh1YN.jpeg)
wild한 형태(울퉁불퉁한 형태)도 가지고 있다.
![](https://i.imgur.com/3KGYaOC.jpeg)
따라서, Neural netowkr는 non-convex optimization을 필요로 한다.

즉, **대부분의 neural network의 (w에 대한) loss function은 convex 함수가 아닐 확률이 크다**. 

따라서 global minimum으로 수렴한다는 것을 보장하지는 않지만, 실전에서는 어떻게든 작동하고 있기 때문에 사용한다고 한다.
![](https://i.imgur.com/AKwzvNo.jpeg)

![](https://i.imgur.com/PghXB8N.jpeg)

![](https://i.imgur.com/vtgbENC.jpeg)

![](https://i.imgur.com/zLSKSov.jpeg)

![](https://i.imgur.com/xGiG4KU.jpeg)
그렇다면 gradient를 어떻게 계산할 수 있을까? 이를 위해 backpropagation을 사용할 것이다.
![](https://i.imgur.com/AbJLdKK.jpeg)
다음 시간에는 backpropagation에 대해서 다룬다.

## Ref
- <https://ploradoaa.tistory.com/76>
- <https://sites.google.com/view/statml-smwu-2020s>
