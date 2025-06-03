---
title: "[EECS 498-007] Lecture3: Linear Classifiers"

categories: ["EECS 498-007 Deep Learning for Computer Vision"]
tags:
  - ["Deep Learning", "Computer Vision"]

date: 2025-03-07
last_modified_at: 2025-03-07
---

## Linear Classifier 개념
![](https://i.imgur.com/NEziGfX.jpeg)
Lecture3에서는 2장에 배운 image classification 문제에 적용할 수 있는 Linear Classifiers에 대해서 다룬다.

![](https://i.imgur.com/5s11NAs.jpeg)
Linear Classifier는 Neural Network를 만들 때, 가장 기본적인 building block으로 사용된다. 
![](https://i.imgur.com/6DhAbwQ.jpeg)
CIFAR10 데이터셋을 기준으로 설명을 진행한다.
- 10개의 label
- 50000개의 training images
- 10000개의 test images
- 각 이미지는 32x32x3의 3차원으로 구성되며, 이를 flatten 시켜서 vector로 펼쳐내면 3072차원을 가지게 된다.

![](https://i.imgur.com/FjJKj4j.jpeg)
Learnable parameter인 $W$가 추가되었는데, 이러한 방법을 Parametic Approach라고 한다.
우리는 parameter(또는 weight(가중치))라고 불리는 $W$에training data에 대한 모든 정보를 담기를 바란다.

즉, **Linear Classifier를 만든다**는 의미는, $f(x,W)$라는 함수를 하나 만드는 것이다.
- $x$ : image
- $W$ : weights
- output : 각 class에 대한 10개의 score


![](https://i.imgur.com/RMXEric.jpeg)
Linear Classifier(=$f(x,W)$)는 $Wx$라는 식으로 나타낼 수 있다. 
- $x$ : 이미지를 flatten 시켜서 3072 size(32x32x3)의 vector로 간주할 수 있음. 이러한 과정에서 이미지의 spatial structure가 파괴됨
- $W$ : 파라미터 matrix로 나타낼 수 있다. 
  $W$의 row는 우리가 구별하기를 원하는 category를 나타낸다.

즉, 이는 matrix-vector multiplication의 과정이다.

![](https://i.imgur.com/0jL2kYT.jpeg)
때로, Linear Classifier에 bias term을 포함시킨 형태를 볼 수 있다. 
- $b$ : bias term(vector); 우리가 recognize해야하는 class에 대해 약간의 offset을 제공함.

즉, <font color="#c00000">x라는 input image에 대해서 (xW +b)의 일차변환을 하는 것</font>이 linear classifer의 핵심이다.

![](https://i.imgur.com/MNfKgAO.jpeg)
예를 들어, 2x2(=4개의 픽셀) 형태의 Input Image와, 3개의 category(cat/dog/ship)를 가지는 경우를 생각해보자.

![](https://i.imgur.com/xNORPcM.jpeg)
weight matrix의 **각 row가 category에 대한 하나의 분류기**가 된다. (category 개수만큼의 row를 가진다.)

score : 확률로 변화되기 이전의 출력값

1행 : ship에 대한 score

2행 : cat에 대한 score

3행 : dog에 대한 score

이 경우,  cat에 대한 score가 가장 높기 때문에 이 이미지는 cat에 배정한다.
## Linear classifier: 3가지 Viewpoints
### 1. Algebraic Viewpoint
![](https://i.imgur.com/gbYr9rd.jpeg)
**Algebraic Viewpoint**는 위와 같이 Linear Classifier를 matrix-vector의 multiplication의 관점에서 이해하는 것이다.

![](https://i.imgur.com/EClCsMY.jpeg)
Algebraic Viewpoint의 이점 중 하나가 **Bias trick**이다.
1. x의 차원을 1개 늘리고, 항상 1의 값을 갖게 한다.
2. W의 마지막 column 뒤에 b를 merge해서, b를 하나의 matrix로 합친다. 

이 trick을 이용하면 단 한 번의 matrix-vector multiplication 연산으로 나타낼 수 있다.

하지만, bias term은 initialize방법과 regularization방법이 weight와 별개로 동작해야 하는 부분이 많기 때문에 이러한 방식이 사용되지는 않는다고 한다.

![](https://i.imgur.com/ZYNMpzE.jpeg)
Linear Classifier의 prediction은 linear 형태이므로, Input image x에 상수를 곱할 때는 f(x, W)의 함수 밖으로 상수를 빼낼 수 있다.
![](https://i.imgur.com/rawQWw0.jpeg)
즉, input 이미지에 $0.5$ 정도의 상수를 곱한다고 가정하면, 약간 흐릿해지는 결과를 얻을 수 있다.

이는 $f(x,W)$의 output인 score에 0.5를 곱한 것과 동일한 효과를 가져오며, 이때 score 자체도 0.5배가 되지만, score의 순서는 그대로 유지된다.
### 2. Visual Viewpoint
![](https://i.imgur.com/sFdpvkE.jpeg)
algebraic 관점으로 봤을때의 Linear Classifier를 이미지적 관점에서는 어떻게 이해할 수 있을까?
![](https://i.imgur.com/NBtAPpL.jpeg)
W의 row를 하나씩 떼어서, 이미지와 동일한 형태인 2x2 구성으로 만든다. 

즉, 기존의 algebraic 관점에서 x와 W를 element-wise multiplication하는 것은 **image가 W라는 필터를 거치는 것**과 같다.
![](https://i.imgur.com/1QyfGQF.jpeg)
각 category에 대한 필터를 거쳐서 나온 결과는 각 category(=class)에 대한 score이다.
![](https://i.imgur.com/WHPmjz9.jpeg)
image가 W라는 필터를 거치는 것으로 나타나므로, **W의 각 row는 각 class별 템플릿으로 이해**할 수 있다.

이처럼 Linear Classifier를 Template Matching(템플릿 매칭)으로 이해하는 관점을 **Visual Viewpoint**라고 한다.

각 템플릿들을 이미지와 내적을 통해 비교하여 score를 계산하고, 이 score를 기준으로 가장 잘 매칭되는 것이 무엇인지 정한다.

즉, Linear Classifier가 결국 템플릿 매칭을 하고 있고, 각 템플릿이 학습을 통해 배워지는 것이다.
![](https://i.imgur.com/0coNjbu.jpeg)
그러나 linear classifier는 각 클래스 당 딱 한 장만의 템플릿만 활용할 수 있다.(하나의 row만 사용하므로)
![](https://i.imgur.com/6fWXhqi.jpeg)

한 장의 템플릿은 데이터의 다양한 모드를 포착할 수는 없다.

CIFAR10 데이터셋에서 horse image들은 오른쪽을 보고있거나, 왼쪽을 보고있는 등 다양한 방향을 보고 있다.

Linear Classifier가 이러한 점들을 학습하려고하다 보니, horse template의 시각화를 보면 마치 말들의 머리가 왼쪽, 오른쪽에 모두 있는 것처럼 보이게 된다.

이처럼 single template으로 모든 모습들을 포착하려하다 보니, 다양한 모양을 가진 동일 label의 image들을 우리가 원하는 label에 assign하지 못할 수 있다.

### 3. Geometric Viewpoint
![](https://i.imgur.com/jEDS5jE.jpeg)
Gemetric Viewpoint는 Linear Classifier를 기하학적 관점으로 이해하는 것이다.

보이는 그래프의 x축은 픽셀값이고, y축은 class score에 해당한다.

Linear Classifier는 Linear function이기 때문에 각 Category의 score function은 linear한 형태로 나타날 것이다.

![](https://i.imgur.com/DHFMTRl.jpeg)
픽셀 2개로 확장하는 경우에는 2차원의 형태가 되기 때문에linear classifier는 면의 형태로 나타나게 된다.
![](https://i.imgur.com/jbb92op.jpeg)
car template(빨간색 직선)이 공간상에서 'car' class에 대한 score 값이 0이 되는 점들의 집합이라 하자.
-> $score = w_1x+b_1=0$ ; $x$는 vector로 2차원의 값을 가짐

화살표 방향은 score가 증가하는 방향으로, car의 score가 증가하면 car로 분류할 확률이 높아진다.
![](https://i.imgur.com/XPbJFDk.jpeg)
다른 모든 cateogry에 대해서도 동일하게 공간 상에서 분류하는 평면을 만들어낼 수 있다.

이러한 관점은 더욱 high-dimensional한(고차원의; 모든 픽셀을 생각한다고 볼 수 있을 것이다.) 이미지에도 적용할 수 있다.
![](https://i.imgur.com/BGZ2lOy.jpeg)
즉, high-demensional space로 갔을 때, 기하학적 관점에서 $Wx+b$는 하나하나의 hyperplane을 나타낸다. 즉, 각각의 hyperplane은 classifier의 역할을 하게 된다.
![](https://i.imgur.com/qpCUdp6.jpeg)
위의 예시들은 Linear Classifier를 활용하기 어려운 경우들이다. 이 3가지 예시와 같은 data 분포는 single hyper plane으로 Class1(파란색 영역)과 Class2(빨간색 영역)를 한번에 자를 수 없기 때문에, linear classifier로 분류할 수 없다고 말할 수 있다.

> 세번째 three modes와 같은 경우는 템플릿 매칭(말)의 경우와 유사하다.

![](https://i.imgur.com/T3Mkmj1.jpeg)
Lec1에서 언급했던 XOR problem이다. 퍼셉트론은 Linear model이기때문에 XOR를 구분지을 수 없다.

![](https://i.imgur.com/9r4Rk25.jpeg)
Linear classifier를 이해하는 세 가지 관점을 다시 정리해보자.
1. Algebraic viewpoint : matrix-vector multiplication으로 이해
2. Visual viewpoint : 템플릿 매칭으로 이해
3. Geometric viewpoint : 고차원 공간상에서 초평면으로 공간을 나누어 이미지를 분류하는 것으로 이해

![](https://i.imgur.com/EiflZUa.jpeg)
이제까지, Linear Classifier에 대해서 배웠다. 위의 그림처럼 3개의 image가 들어간다면, matrix-vector multiplication 연산을 통해서 10개의 score값을 가진 vector를 3개 출력할 것이다.

**Classifier의 궁극적인 목표는 true class에 대한 점수를 높게 설정하는 것**이다. 예시를 보면, '고양이' 이미지를 dog으로, '자동차' 이미지를 automobile로,  '개구리' 이미지를 truck으로 분류한다. 즉 3개의 예시 중에서도 2개를 오분류 하고 있음을 알 수 있다.

이는, 우리가 만든 Linear Classifier인 $f(x,W)=Wx+b$의 $W$가 좋지 않았기 때문이다. 그렇다면 **어떻게 좋은 $W$를 찾을 수 있을까?**

![](https://i.imgur.com/QMSVEb4.jpeg)
1. W값이 얼마나 좋은지를 나타내기 위해서 Loss function을 이용한다.
2. loss function을 최소화하는 W를 찾는다.(이러한 과정을**optimization**(최적화)라고 한다.)

## Loss Function
![](https://i.imgur.com/w0DSQwb.jpeg)
Loss function은 W가 어느 값으로 고정 되었을때, Linear classifier가 얼마나 잘 수행하고 있는지를 말해주는 함수이다.
- 낮은 loss = 좋은 classifier
- 높은 loss = 좋지 않은 classifier

loss function은 **objective function**(목적 함수), **cost function**(비용함수)라고도 불린다.

![](https://i.imgur.com/ztmS5vs.jpeg)
**Negative loss function**은 minimize하는게 아니라, **maximize**할 때 사용한다. 즉, loss function의 부호만 바꾸면 구할 수 있다. 이를 reward function, profit function, utility function, fitness function..등등으로 나타낸다.
![](https://i.imgur.com/S47tyXT.jpeg)
datatset은 보통 1~N장까지의 ${(x_i, y_i)}$의 형태로 구성되어 있다.

- $x_i$ : 이미지
- $y_i$ : (integer) label

![](https://i.imgur.com/YjkYbD8.jpeg)
하나의 sample을 이용하는 예시에 손실함수는 위와 같이 나타난다. 이 방식을 요약하면 아래와 같다.
1. i번째 sample($x_i$)에 대하여 linear classifier를 적용한다.($f(x,W)$를 만든다.)
2. 계산한 score를 Loss function을 이용하여 i번째 이미지의 $y_i$(true label)와 비교한다. 

![](https://i.imgur.com/XNumjRk.jpeg)
전체 dataset에 대한 loss는 각 data sample의 loss들을 평균낸 것을 사용한다.
### Multiclass SVM Loss
![](https://i.imgur.com/QFP3loH.jpeg)
그럼 Loss Function으로는 어떤 것이 사용될 수 있을까?

Loss function의 첫번째로 **multiclass SVM Loss**에 대해 알아보자.

multiclass SVM Loss의 아이디어는 **Correct class에 대한 스코어가 다른 class에 대한 스코어보다 높아야 한다**는 것이다.

이때, x축을 correct class에 대한 score로, y축을 Loss로 나타낼 수 있다.
![](https://i.imgur.com/vgvZiiy.jpeg)
correct class를 제외하고, 2번째로 높은 score(=Highest score among other classes)를 찾는다.
![](https://i.imgur.com/1cYztg2.jpeg)
Correct Class의 score가 2번째로 높은 score를 가진 class와 **margin 만큼의 차이가 날 때** '전체' Loss가 0이 된다.

또한, 왼쪽으로 갈수록(두 클래스간 차이가 줄어들수록) loss가 커진다.

> Hinge Loss : 문에 있는 hinge와 유사하여 SVM Loss는 Hinge Loss라고도 불린다.

![](https://i.imgur.com/ASylC27.jpeg)
$x_i$ : image

$y_i$ : label; 1~c까지의 integer 형태로 주어졌다고 가정

$s$ : c차원의 vector(각 클래스에 대응되는 점수값)

$L_i$ : SVM loss

- $s_j$ :  sample label score
- $s_{y_i}$ : true label score
- $+1$ : margin

SVM Loss의 식은 위와 같으며,  true class($y_i$)를 제외한 class에 대한 스코어를 true class에 대한 스코어와 비교하는 것이다. 

$s_j - s_{y_i}$이라는 **두 값의 차이가 1보다 커야** (= $s_j - s_{y_i}$ < -1) loss가 0이된다.

![](https://i.imgur.com/aAhQwYo.jpeg)
SVM Loss에 대한 예시를 들어보자.

class가 cat / car / frog 세개로 분류된다고 하자.

i = 1일 때 고양이 이미지, i=2일때 car 이미지, i=3일때 frog 이미지가 주어진다.

현재 matrix-vector multiplication으로 각 이미지별로 class score가 도출된 상황이다.

![](https://i.imgur.com/eBYzz76.jpeg)
i = 1일 때(이미지가 고양이일때), $y_i$(true label)는 'cat'이다.

즉, $s_{y_i}$ = cat score =  $3.2$

true class를 제외한 class에 대한 스코어를 true class에 대한 스코어와 비교해야한다. 이 경우, SVM loss의 공식에 따라 위와 같이 계산할 수 있고,  loss = 2.9라는결과를 얻는다. 

![](https://i.imgur.com/7EK2GSc.jpeg)
i = 2일때 자동차 이미지에 대해서도 똑같이 계산할 수 있다. 이 경우, correct class(car)에 대한 score가 두번째로 높은 score(frog)보다 1(margin)이상 크므로, loss는 0이된다.
![](https://i.imgur.com/WrYgj1x.jpeg)
i=3일때 frog의 경우에도 같은 방식으로 계산한다.
![](https://i.imgur.com/aecaGhq.jpeg)
전체 dataset에 대한 loss를 구하기 위해, 각 data sample의 loss들을 평균낸 것을 사용했다.
![](https://i.imgur.com/llqUU3J.jpeg)
Q. car 이미지( i=2일때 )에 대해서 score가 조금(0.01정도) 바뀐다면, Loss값이 어떻게 될까?

A. 그대로 0일 것이다. correct class와 다른 class들 간의 차이가 1이상 차이나는 것은 변함 없기 때문이다.
![](https://i.imgur.com/SRMomKP.jpeg)
Q.  각 i에 대해, 가능한 loss값의 최솟값과 최댓값은?

A. 

최솟값 : 0($s_j - s_{y_i} + 1$가 음수가 되면 max함수에 따라 0이 출력되기 때문)

최댓값 :  inf(단 하나의 sample에 대해서 loss 값이 inf라면 전체 loss도 inf가 된다.)

![](https://i.imgur.com/VUX8RuC.jpeg)
Q. W matrix가 랜덤으로 초기화된다면, 각 data sample에서 어떠한 loss가 나올까?

A. 일반적으로 랜덤으로 초기화 할 때는, 표준편차 0.01과 같은 값으로 초기화를 시킨다. 이 경우, weight matrix는 0에 가까운 값을 가지게 되고, 이러한 weight matrix와의 연산을 통해서 얻은 class score또한 0에 가까운 값을 가지게 된다.
따라서, correct class와 incorrect class간의 score 차이($s_j - s_{y_i}$)도 0에 매우 가까운 값이므로, 초기 loss는 1에 가까운 값을 가진다.

즉, 전체 category의 수를 $C$라고 하면, $C-1$(true label 제외)의 loss를 가지게 될 것이다.

이 것을 추후 디버깅에 활용할 수 있다.랜덤으로 초기화 시 전체 loss가 c-1값이 나오지 않으면 어딘가 잘못된 것이다.

![](https://i.imgur.com/z4LVgBG.jpeg)
Q. SVM loss에서는 correct class를 제외하고 전체 loss를 구하였는데 만약 correct class 포함하여 계산하게되면 어떻게 되는가? 

A. $j = y_i$일 때, max(0, $s_j-s_{y_i}+1$) = max(0, 1) = 1

따라서, 원래 loss + 1의 값이 된다.
![](https://i.imgur.com/JSz657y.jpeg)
Q. 만약 각 sample의 loss에서, sum이 아닌 mean(평균)으로 바꾼다면?

A. 각 image의 loss의 크기에 대한 순서의 변동은 없다. 


![](https://i.imgur.com/XPF4xp6.jpeg)
Q. loss값을 구할 때, max(0, $s_j-s_{y_i}+1$)을 제곱해서 사용하면 어떻게 될까?

A. 더이상 multiclass SVM loss가 아니고, 다른 loss가 된다.

![](https://i.imgur.com/5CetCBO.jpeg)
Q. 그렇다면, 어떤 W matrix로 부터 zero Loss를 달성했다면, 이것은 W matrix는 zero Loss를 달성할 수 있는 유일한 W인가?

A. 유일하지 않다. W matrix에 2배를 곱해도 $L = 0$이 된다.


![](https://i.imgur.com/FpgniLp.jpeg)
위의 사진에서 2W또한 Loss가 0이 나옴을 확인할 수 있다.
![](https://i.imgur.com/fT4rLcY.jpeg)

그렇다면 W나 2W나 여전히 트레이닝 데이터에 대해서 동일한 loss를 갖는다면 W를 어떻게 선택해야 할까?

-> 우리의 최종 목표는 training set이 아닌 test set에서 classifiy 하는 것이므로, 대체적으로 잘 작동하는 model이 필요하다. 이러한 모델의 일반화를 위해 정규화(regularization)을 사용할 수 있다.

### Regularization
![](https://i.imgur.com/AY0UQ7h.jpeg)
위와 같은 기본적인 형태를 **Data loss**라고 부른다.

기본적으로, training data에 점점 잘 맞추도록 W가 update 된다.
![](https://i.imgur.com/4qIhulX.jpeg)
Data loss에 Regularization term을 추가할 수 있다. 

이는 training data를 잘 맞추려고 하는 과정에서, training data에 너무 의존하는 것(overfitting)을 방지하는 역할을 한다.
![](https://i.imgur.com/ajqAWT5.jpeg)
이때, regularization을 얼마나 강하게 둘 것인지에 대해(regularization의 강도를 조절) $\lambda$라는 hyperparameter를 사용할 수 있다.
![](https://i.imgur.com/iyXEgai.jpeg)
Regularization의 예시로는 L2 regularization과 L1 regularization, L1과 L2를 합친 Elastic net 등이 있다. (단순히 $R(W)$ 식에 위의 식들을 사용하면 된다.)

Data loss에 regularization term을 추가하는 방식 이외에도 Dropout, Batch normalization 등의 방법도 존재한다.
![](https://i.imgur.com/hEoU327.jpeg)
Regularization의 목적은 아래와 같다.

1. training error를 최소화시키기 위해 만든 model들 사이에서 preference를 나타낸다.(즉, 하이퍼 파라미터 조정같은 인간의 지식도 반영되도록 하기 위함)
2. overfitting을 방지(너무 적합한 것보다는 단순한 모델을 사용하게 한다.)
3. 곡률(낙폭)을 더함으로써, gradient가 더 경사지게 하여 최적점을 빨리 찾게 한다.(optimization과정을 향상시킨다.)

![](https://i.imgur.com/E4icmVK.jpeg)
예를 들어, Regularization은 Preference를 나타낼 수 있다.
> Preference : 강의에서, preference는 '연구자' 관점에서 '어떤 것에 조금 더 중점을 두는지'를 나타내는 용어 정도로 이해할 수 있을 것 같다. 

주어진 예시에서 weight와 $x$의 dot product는 둘 다 1이지만, $w_1$과 $w_2$의 data 분포에는 차이가 있다.

![](https://i.imgur.com/jJonJRw.jpeg)
L2 Regularization을 사용한다고 가정할 때, 고르게 퍼진 값의 $R(W)$가 더 작으며, L2 Regularization은 각 weight 값들을 spread out(늘려 퍼뜨리는) 성질을 가진다.  즉, 하나의 weight에 몰려있는 것보다 weight가 전반적으로
퍼진 것을 선호할 때, 이런 정규화를 진행할 수 있는 것이다.

이처럼 모든 feature들이 서로 다른 feature와 연관이 있다면 L2 regularization을 사용하는 편이 좋다. <- 이런 preference를 반영할 수 있다.

> L1 Regularization을 사용하면, 하나의 feature에만 집중하게 만들 수도 있다.

![](https://i.imgur.com/YxYjaQM.jpeg)
Regularization은 overfitting을 방지하기 위해서 학습시 간단한 모델을 preference하도록 만든다.
x가 input으로 들어오고, y를 예측하는 regression task에서, 5개의 data sample이 있다고 가정해보자.
![](https://i.imgur.com/7x4oGfM.jpeg)
f1: polynomial model 적합 | f2: linear model 적합

f1은 training data에 대해 완벽하게 들어맞는다.

f2는 training error가 존재한다.

따라서, training data의 loss로만 보면 f2의 loss가 더 크다고 판단할 수 있다.
![](https://i.imgur.com/r2clWyd.jpeg)
하지만, 새로운 input data로 회색 포인트들(test data)이 들어왔을 때,  noise가 더해져 있다면 f1은 보다 큰 오차를 가지게 된다.

- noise : 이 경우에서는, f1에서 벗어난 정도
- 
따라서, noise가 끼어있을 수 있는 traininig set에 너무 정확하게 맞추면, 새로운 data(test data)의 nosie에 잘 대처할 수 없기 때문에 Regularization은 simple한 model을 preference하도록 만들어준다.

### Cross-entropy loss
![](https://i.imgur.com/13SmO8Q.jpeg)
Loss Function의 두번째로 cross-entropy loss를 알아보자.

SVM Loss는 score에 대해서 어떠한 해석도 제공하지 않았다. 하지만, cross-entropy loss는 score의 점수를 확률(Probability)로 이해할 수 있게 해준다.
![](https://i.imgur.com/n7tj1yl.jpeg)
SVM Loss에 대한 예시를 들어보자.  

class가 cat / car / frog 세개로 분류된다고 하자.  

i = 1일 때 고양이 이미지, i=2일때 car 이미지, i=3일때 frog 이미지가 주어진다.  

현재 matrix-vector multiplication으로 'cat' 이미지에 대한 class score가 도출된 상황이다
![](https://i.imgur.com/BRO69pJ.jpeg)
우리에게 주어진 각 score들은 unnormalized log-probabilities / logits(확률로 변경되기 이전의 값)이다.

이제부터, 이 score들을 확률로 해석할 수 있도록 바꿔보자.
![](https://i.imgur.com/fsD6jKo.jpeg)
1.exp()함수를 취해준다.
   - 각 score가 $e$의 지수로써 들어가게 되는 것이다.

지수함수는 항상 양수의 값을 가지므로, 모든 score가 이제 양수의 값을 가지게 된다. <- 확률은 모두 양수이므로, 이 성질을 만족시켜주기 위해 지수함수를 취했다.

이 과정이 끝난 값들을, unnormalized probabilities라고 부른다.
![](https://i.imgur.com/gJyBWEw.jpeg)
2.이후, 확률값으로 만들어주기 위해서 지수함수를 취한 모든 score값의 합을 각 score별로 나누어준다.

1, 2의 과정을 score에 **softmax function**을 사용하는 것으로 이해할 수 있다.
이 과정을 normalize 과정이라고 한다.
- normalize : 값의 범위(scale)를 0과 1사이의 임의의 값으로 바꾸는 것

![](https://i.imgur.com/V41xl4W.jpeg)
최종 Loss값은, true class일 확률 값(=$P_{y_i}$에 $-log$를 씌운 형태로 이해할 수 있다._

-> $-log(P_{y_i})$

**즉, $P_{y_i}$가 1에 가까울수록(잘 맞추었을수록) loss의 값이 0에 가까워진다.**
![](https://i.imgur.com/UnKll7L.jpeg)
해당 아이디어는 **최대우도추정법**으로부터 왔다고 한다.
![](https://i.imgur.com/9Mp83dJ.jpeg)

- correct probs : true class는 1로, 나머지 class는 0으로 표현한 형태.

> max 함수의 경우 최댓값에만 1을 반환하고 나머지는 모두 0을 반환한다. 이러한 극단적인 방식이 아니기 때문에 **softmax**라는 이름이 붙었다고 한다.

![](https://i.imgur.com/9kMiADD.jpeg)
위의 과정에서 보라색 박스(correct probs)와 초록색 박스(probabilities)를 비교해야한다. 이러한 확률 분포를 비교하는 방법이 Kullback-Leibler divergence (KL-Divergence) 라는 방법이며, 그 식은 위의 마지막 식과 같다.
![](https://i.imgur.com/b9KyfMJ.png)
식을 전개하여 보면, true label일 때의 확률만 고려하면 됨을 알 수 있다.
![](https://i.imgur.com/mn8osQw.jpeg)
> 참고 : 두 분포간의 차이를 최소화 하는 것 (KL-Divergence를 최소화 하는 것)은  Cross entropy H(P,Q)를 최소화 하는 것과 같다고 한다.
>
> 즉 두 분포간의 차이를 보는 것이 Cross entropy와 관련이 있기 때문에 이러한 loss function을 Cross Entropy Loss라고 부르게 되었다.

![](https://i.imgur.com/lJU70fj.jpeg)
다시 정리하면 cross entropy loss는 input image가 true class에 속할 확률을 구해준다.

따라서, cross entropy loss를 사용할 때는 input image가 correct class에 속할 확률(=$P_{y_i}$)을 maximizing해야한다.(그래야 loss가 0에 가까워지므로)
![](https://i.imgur.com/kaQKiAt.jpeg)
Q. 가능한 loss값의 최소/최대값은 무엇일까?
![](https://i.imgur.com/iPWM0fH.jpeg)
A. 

최대 : +inf

최소 : 0
![](https://i.imgur.com/43TphOY.jpeg)
Q. 모든 스코어가 랜덤이라면, 로스값은 어떻게 될까?
![](https://i.imgur.com/bJRw48C.jpeg)
std=0.01정도일 때, 모든 class score를 랜덤으로 초기화하면 score들이 0~1사이의 값들을 가지게 된다. 즉, 각 클래스마다 score의 값이 거의 차이가 나지 않으며, 이 경우에는 correct class에 속할 확률인 $P_{y_i}$가 $\frac{1}{C}$이라는 확률을 가지게 된다.($C$는 전체 카테고리 개수)

따라서, loss값은 $log(C)$의 값을 가지게 된다.

즉, 이것은 중요한 insight로 활용된다.

만약 10개의 category의 loss를 줄이기 위해서 테스트 하는 과정에서, $-log(10)$(대략 2.3)의 값보다 loss가 크다는 것은, W matrix를 랜덤하게 초기화한 것보다 성능이 더 안좋다는 뜻과 같고, 이는 잘못된 방법으로 model을 training 시키고 있다는 뜻이 되므로, 학습을 중단해야된다.

![](https://i.imgur.com/kpCP3qf.jpeg)
correct class가 score=10을 가진다고 하자.

Q. 해당 예시에서 cross-entropy loss의 값은? SVM loss의 값은? 
![](https://i.imgur.com/I9blECY.jpeg)
A. 

cross-entropy-loss > 0

SVM loss = 0
![](https://i.imgur.com/8hfJP3Q.jpeg)
Q. 맨 마지막 데이터 포인트의 score가 조금(+0.1정도) 바뀐다면, Loss값이 어떻게 될까?
![](https://i.imgur.com/HXlB82e.jpeg)
A. 

Cross-entropy loss는 분모에서 모든 score를 사용하므로, score가 조금 변하더라도 loss가 변화한다.

SVM loss는 그대로일 것이다.(margin이 다른 score와 1이상 차이가 나므로)
![](https://i.imgur.com/eSgTxJq.jpeg)
Q) true class에 대해서 스코어값이 2배 (10에서 20)가 되면, 각 loss값이 어떻게 될까?

A. 

이전 문제와 비슷한 논리로서, SVM Loss는 여전히 zero loss일것이나 Cross-Entropy loss는 감소할 것이다.

Cross-Entropy loss가 증가하지 않고 감소하는 이유를 살펴보자면, true class 스코어가 커지면서 true class에 대한 proportion($P_{y_i}$)이 커진다면, 전체적인 loss값($-log(P_{y_i})$)이 작아질 것이기 때문이다.

## Summary
![](https://i.imgur.com/mOUjqkf.jpeg)
3강에서 다룬 내용들을 간단히 요약해보자.
1. Algebraic viewpoint : matrix-vector multiplication으로 이해
2. Visual viewpoint : 템플릿 매칭으로 이해
3. Geometric viewpoint : 고차원 공간상에서 초평면으로 공간을 나누어 이미지를 분류하는 것으로 이해

![](https://i.imgur.com/XUXwUQM.jpeg)
Loss function을 이용함으로써 preference를 수치적으로 나타낼 수 있다는 것을 이해하자.

**Full loss** : Data Loss + Regularization term
![](https://i.imgur.com/vj7eGrG.jpeg)
가장 좋은 W를 찾는 방법을 optimization이라고 하며, 이는 다음 강의에서 알아본다.
![](https://i.imgur.com/oJkTU4o.jpeg)


## Ref
- <https://ploradoaa.tistory.com/76>
- <https://sites.google.com/view/statml-smwu-2020s>

