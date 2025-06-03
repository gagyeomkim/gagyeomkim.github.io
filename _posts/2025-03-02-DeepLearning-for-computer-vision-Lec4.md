---
title: "[EECS 498-007] Lecture4: Optimization"

categories: ["EECS 498-007 Deep Learning for Computer Vision"]
tags:
  - ["Deep Learning", "Computer Vision"]

date: 2025-03-19
last_modified_at: 2025-03-19
---

![](https://i.imgur.com/h8iBcZX.jpeg)
Lecture 4에서는 Optimization(최적화)에 대해 배운다.

optimization : loss function을 최소화하는 W값을 찾는 과정

![](https://i.imgur.com/YHVTQGZ.jpeg)
Optimization의 목적은 output인 $L(W)$(=loss)를 최소화하는 $w$를 찾는 것이다.

이때의 $w$를 $w^*$이라 한다.

그렇다면 이 $w^*$를 어떻게 구할까?
![](https://i.imgur.com/H3ibCbQ.jpeg)
Optimization은 앞이 보이지 않는 사람이 산비탈을 탐험하는 것과도 같다.

(산비탈은 high dimensional landscape이라고 표현할 수 있다.)

$L(W)$는 산비탈과 같이 주로 전체 개형을 알 수 없다. 그렇다면 자신이 서있는 지점만이 주어졌을 때, 어떻게 global minimum을 찾아갈 수 있을까?

![](https://i.imgur.com/QVEHicr.jpeg)
우선, 아래로 내려가야하는 것이, global minimum을 찾아가는 방법임을 알 수 있다. 하지만 한번에 explicit solution을 얻을 순 없으므로, Iterative method를 이용할 수 있다.
- Iterative method : $w^*$에 도달하기 위해서 반복적으로 $w$를 개선시켜나가는 방법

> 단, 이는 global minimum으로 가는 것을 보장하지는 않는다.
> local minimum이나 saddle point(안장점)에 갇힐 가능성이 있다.

![](https://i.imgur.com/rNYyHdO.jpeg)
$w^*$를 찾는 아이디어의 첫번째는 **Random Search**이다.

random search는 $W$ matrix의 값을 랜덤으로 초기화해서, 각각의 W matrix에 대응하는 Loss값을 모두 구하고, 그 크기를 비교하는 방법이다. 즉, Loss를 반복적으로 계산하면서 Loss가 개선되면 W를 업데이트한다.

![](https://i.imgur.com/qhRkM5Z.jpeg)
하지만, Random Search는 좋은 아이디어가 아니다. 랜덤으로 W값을 개선시켜나가는 방법은 15.5%의 accuracy를 달성한다.(SOTA의 정확도는 95%이다.)
## Computing Gradient
![](https://i.imgur.com/NNzzA5R.jpeg)
두번째로 생각할 수 있는 아이디어는, 남자의 감각을 통해서 주변의 경사를 따라 내려가는 방법이다. 전체 shape을 알지는 못할테지만, 현재 지점보다 아래로 경사진 곳으로 내려가는 것이 낮은 곳으로 간다는 것은 확실할 것이다.

즉, **국소적** gradient를 구해서 함수가 감소하는 방향으로 내려가는 방법을 사용하는 것이다.

이 방법을 사용할 때 단순히 가파른 방향으로 조금의 step만큼만 가보는 것을 반복함으로써, $w^*$에 도달할 수 있게 된다.
![](https://i.imgur.com/GHlZEV4.jpeg)
 gradient는 $x$가 조금 변할 때, 전체 $f(x)$가 얼마나 변하는지를 나타내며,
이는 곧 앞에서 나온 경사도(slope)를 의미한다.

gradient의 기하학적 의미에서 중요한 점은,  gradient의 방향은 함수가 가장 증가하는 방향(greatest increase)을 나타낸다는 것이다. 그래서 gradient의 반대 방향은 **그 지점에서 가장 가파르게 감소하는 방향**이 된다.

즉, gradient는 $f(x)$(=loss function)의 '증가' 방향을 나타내므로, **negative gradient** 방향으로 가야 $L(W)$의 값을 가장 많이 감소시킬 수 있다.

> multiple dimension(다변수)에서는 gradient가 vector의 형태로 나타난다.

### Numeric Gradient
![](https://i.imgur.com/xHKEHQc.jpeg)
그런데,  gradient의 값은 어떻게 계산하는 걸까?
- gradient dL/dW : 각 방향에 대한 편미분
- 우리가 구하는 loss는 단순히 single scalar값이다.
- gradient의 shape은 w matrix의 shape과 같다는 것을 주의하자.

첫번째 아이디어는 **Numeric** gradient(수치적 gradient)이다.

이는, 현재 $W$의 모든 dimension에 대해서 해당 방향으로 아주 조금(h) 이동했을 때, Loss값이 얼마나 변하는지를 구하는 것이다. (수치적으로 접근하여 근사하는 방식)

![](https://i.imgur.com/AJP6efV.jpeg)
$h$ : 0.0001

현재 $W$의 첫번째 dimension에서 h만큼 이동시켰다.

우리가 아는 gradient 식으로 계산하면, 첫번째 gradient 값은 $-2.5$이다.

![](https://i.imgur.com/c1zsbmb.jpeg)

![](https://i.imgur.com/UESbF4S.jpeg)

![](https://i.imgur.com/Y4iVXqZ.jpeg)

![](https://i.imgur.com/s0TPCvk.jpeg)

![](https://i.imgur.com/1RrtoYt.jpeg)
두번째, 세번째에도 같은 과정을 반복할 수 있다.
![](https://i.imgur.com/ujFx2PK.jpeg)
이런 방식으로 모든 dimension에 대한 gradient를 업데이트 시킬 수 있으며, 이 계산 방법을 **Numeric** gradient라고 한다.

numeric gradient 계산 방법은 간단하고 쉬운 방법이지만, 
1. dimension의 개수만큼 계산해야하므로 **느리고**(ex. W matrix의 크기가 매우 큰 경우), 
2. gradient 값이 정확한 값이 아닌 **근삿값**이라는 단점이 있다.(h가 진짜로 0에 무한하게 가깝게 갈 순 없기 때문에)

### Analytic Gradient
![](https://i.imgur.com/WBoLOTm.jpeg)
gradient를 구하는 두번째 계산 방법은 **Analytic** gradient(해석적 gradient)이다.

Analytic gradient 계산법은 **정확한 수식**을 이용하여 직접 미분을 하는 계산법으로, 우리는 벡터 미적분학의 도움을 받아 Loss function에 대한 weight matrix의 gradient를 계산할 수 있다

![](https://i.imgur.com/bkaByuf.jpeg)
그럼 어떻게 analytic gradient를 구하는 걸까?

-> 이것은 6강의 backpropagation에서 상세히 다룬다.

우리는 analytic gradient의 방법으로 매우 큰 형태의 weight matrix의 계산도 다룰 수 있게 된다.

### Gradient Check
![](https://i.imgur.com/tO4EIh2.jpeg)
즉,  gradient 계산법에는 2가지가 존재한다.
1. numeric gradient : 근삿값을 계산하고, 계산이 느리지만, 작성이 쉽다.
2. Analytic gradient : 정확한 값을 계산하고, 빠르지만, 실수가 발생하기 쉽다.

실제로 gradient를 계산할 때는 항상 Analytic gradient를 사용하고, numeric gradient를 활용해서 검증의 과정을 거친다고 한다. -> 이를 **gradient check**라고 부른다.
![](https://i.imgur.com/4apSyW5.jpeg)
Pytorch에서는 `gradcheck`라는 함수를 제공하기에, 이를 사용해서 gradient check를 쉽게 할 수 있다.

![](https://i.imgur.com/tptOKV9.jpeg)
`gradgradcheck`라는 함수도 있는데, 이차 미분을 체크할 때 사용한다.
> 3, 4차 미분을 체크하는 함수는 없다고 한다.

## Gradient Descent
![](https://i.imgur.com/gkgUSG5.jpeg)
`dw` : $\frac{dLoss}{dw} = L(W)$

gradient descent는 negative gradient방향으로 이동하면서 계속 W값을 update시켜서, 가장 낮은 loss를 보이는 weight matrix, $w^*$를 찾는 과정을 말한다.

gradient descent를 구현할 때 설정해야하는 하이퍼파라미터는 아래와 같다.
1. weight 초깃값을 설정하는 방법 : 
	- 초깃값이 어디냐에 따라 수렴하는 위치가 달라진다.
	- 보통은 weight 초기값으로 random value를 사용한다.
2. number of step(step의 수; 얼마나 반복할 것인지)
	- step의 수가 너무 크면 시간이 소요되고
	- 너무 작으면 충분히 내려가기 전에 알고리즘이 종료될 수 있다. 
	- 우리는 언제까지고 학습할 수 없기 때문에, step의 수를 잘 설정해야한다.
3. learning rate(학습률)
	- step의 크기를 조절함.(이 방향을 얼마나 신뢰할건지)
	- 너무 크게 설정 : 학습시 발산해버려서 loss값이 커질 수 있음
	- 너무 작게 설정 : global minimum까지 가는데 너무 오래 걸림.

![](https://i.imgur.com/FEOpMNG.jpeg)
gradient descent의 기하학적인 해석이다.

현재 $L(W)$는 2차원 convex함수이며, contour plot(등고선)으로 $L(W)$를 나타냈다.

- convex : 토기모양을 생각해라. 
  -> ![|30](https://i.imgur.com/H4CZPUP.png)

negative gradient 방향으로 이동하면서 w를 업데이트 시키면 global minimum(=loss의 최솟값; 즉, optimal(최적점))으로 갈 수 있는 것을 볼 수 있다.

### (Full) Batch Gradient Descent
![](https://i.imgur.com/Ky80bc6.jpeg)
지금까지 사용한 gradient 방식은 모든 training data를 계산에 사용하는 **Batch Gradient Descent** 또는 **Full** Batch Gradient Descent이다.

training data의 사이즈가 매우 클 경우, 파라미터 W를 한번 업데이트 하기 위해서 모든 training data를 활용하는 것은 계산의 연산 비용이 많이 들기 때문에 이러한 방법을 고수하는 것은 좋지 않은 방법이다.

### Stochastic Gradient Descent(SGD)
![](https://i.imgur.com/bi21QN7.jpeg)
Full batch gradient descent는 $W$를 업데이트 하는 데에 시간이 너무 오래 걸리기 때문에, 모든 training set을 사용하는 것이 아니라, 전체 training set에서 <u>일부 샘플을 랜덤으로 추출</u>해서 실행하는 **Stochastic** Gradient Descent(SGD)방법을 주로 활용한다.

이때, **랜덤으로 추출한 샘플**을 **minibatch**(미니배치)라고 부르며,  일반적으로 32, 64, 128개의 사이즈를 이용한다고 한다.
> 다시말해, $W$ 업데이트도 minibatch를 이용해서 한다.

SGD를 이용하면 hyperparameter의 개수가 더 증가한다.
1. weight 초깃값을 설정하는 방법
2. number of step(step의 수; 얼마나 반복할 것인지)
3. learning rate(학습률)
4. Batch size(미니 배치의 사이즈)
	- batch size는 gpu memory가 가용하는 한 최대로 크게 설정하는 것이 가장 heuristic 접근 방법이다.
5. Data sampling(데이터를 샘플링(랜덤 추출)하는 방법)

![](https://i.imgur.com/SRwj4Tb.jpeg)
$${(x_i,y_i)}^N_{i=1} \sim (x, y) \sim p_{data}$$ 

$\sim$ : 유사하다.

$(x, y) \sim p_{data}$: 확률분포 p를 따르는 확률변수 x,y


Stochastic Gradient Descent의 해석은 **확률적** 경사 하강법이다. 이는 loss fucntion을 확률적 관점에서 볼 수 있다는 의미에서 붙여졌다.(랜덤으로 추출한 minibatch의 평균을 전체 dataset의 평균으로 근사할 수 있다!)

p라는 확률 분포에서 N개의 데이터를 샘플링 했을 때, 그 샘플링된 데이터에서 **loss**를 계산하여, <u>모집단 평균 loss를 sample 평균 loss로 근사하는 것이 가능</u>하다고 한다.(monte carlo estimation 이용)

![](https://i.imgur.com/FZBh3xZ.jpeg)


마찬가지로, p라는 확률 분포에서 N개의 데이터를 샘플링 했을 때, 그 샘플링된 데이터에서 **gradient**를 계산하여, <u>모집단 평균 gradient를  sample 평균 gradient로 근사하는 것이 가능</u>하다고 한다.(monte carlo estimation 이용)

> 참고 : monte carlo estimation

## Gradient Descent의 문제점
![](https://i.imgur.com/n5hvrVT.jpeg)
vanila(기본) version의 SGD는 일부 문제를 마주할 수도 있다. 

위의 그림은 taco shell 모양인 loss function의 contour plot을 그린 것이다. 그림을 살펴보면, 수직 방향에서 loss 변화는 매우 빠른 데에 비해서, 수평 방향에서의 loss 변화는 매우 느리게 진행되고 있음을 알 수 있다. 

SGD 방법을 이용할 때 이러한 경우의 loss function을 만난다면, 어떻게 진행될까?

> loss function이 매우 높은 condition number를 가지면 옆으로 넓게 퍼진 모양이다.

![](https://i.imgur.com/10E15PW.jpeg)
첫번째는 $L(W)$함수가 옆으로 넓게 퍼진 taco shell의 모양이라면, gradient descent 방법을 통해 step을 진행할 때 우리의 step이 진동(oscilate)한다는 것이다. (지그재그 모양으로 이동한다는 뜻으로, 비효율적이다.)

이러한 문제를 피하기 위해서 step size를 적게 설정하면 알고리즘의 수렴 속도는 매우 느려진다.
![](https://i.imgur.com/cHuCBeq.jpeg)
두 번째 문제는 **local minimum**값(극솟값)이나 **saddle point**(안장점)이 있는 경우이다. 

Local minimum과 saddle point에서의 gradient($\Delta L(W)$)값은 0이기 때문에, global minimum(optimal)이 아닌 위치에서 gradient값이 0이되면 이 곳에 갇혀서 더이상 $w$값이 업데이트되지 못한다.

![](https://i.imgur.com/3tq93go.jpeg)
![|200](https://i.imgur.com/Ecj8iu6.png)
세 번째 문제는 gradient 계산을 minibatch를 이용해서 계산하기 때문에 매우 nosie가 높다는 것이다. 

SGD의 방향이 gradient descent의 true direction과 정확하게 일치하지 않아서 $w^*$를 찾아가는 데에 상당한 step이 필요할 수 있다.
![](https://i.imgur.com/OLTS3hd.jpeg)
이러한 문제점들로 인해서 vanila 형태의 SGD는 잘 사용하지 않는다고 한다.

## SGD의 변형 알고리즘
### Momentum
![](https://i.imgur.com/AG6XRhm.jpeg)
Neural Net을 training 시킬 때는 SGD+Momentum을 사용한다.

$v_t$ : 속도

$\rho$ : 마찰계수(rho); 과거 속도가 마찰에 의해 조금씩 날아간다.

모멘텀(Momentum)은 물리에서의 관성에서 온 아이디어로, $v_t$(현재속도)에 gradient가 누적되고, 이를 통해 $w$를 업데이트한다. 정확한 gradient 방향이 아니라 계산된 velocity의 방향으로 이동하게 된다.

하이퍼파라미터로 도입된 $\rho$를 과거 속도에 대해 감소를 일으키는 역할을 한다.(=decay rate)
![](https://i.imgur.com/5kyLFcL.jpeg)
어떤 책이나 논문에서는 SGD+Momentum에서 다른 형태의 식을 이용하는 것을 볼 수 있다. 두 수식 중 어떤 수식을 선택하더라도 동일한 $W$ matrix를 얻을 수 있으며, 단지 구현의 차이일 뿐이다.

**즉, 두 수식은 같다.**
![](https://i.imgur.com/dGh9s8G.jpeg)
![](https://i.imgur.com/mrsFiCc.png)
모멘텀 방법을 도입함으로써, 앞의 문제들을 해결할 수 있다.

1. vanila SGD와는 다르게 local minimum에 빠져도 이전에 내려오던 velocity가 있기 때문에 지나갈 수 있다. 
2. saddle point에서도 누적된 값이 있어서 가던 방향으로 계속 나아갈 수 있다.
3. 속도가 누적되기 때문에 SGD보다 더 빠르다.
4. vanila SGD보다 gradient 계산의 noise가 줄어든다.


### Nesterov Momentum
![](https://i.imgur.com/qHubUng.jpeg)
모멘텀의 업데이트 방식은 현재지점까지 그동안 누적되어 온 velocity(vector)와 gradient(vector)의 가중합으로 actual step(실제 스텝)을 이루어 $w$를 업데이트하는 방식이었다.
![](https://i.imgur.com/b8UDEbF.jpeg)
Momentum의 또다른 버전으로 **Nesterov Momentum**이 존재한다.

이 방법은 velocity가 update된 지점에서 미래의 gradient를 예측하고, 이를 가중합(linear combination)하여 actual step을 이룬다.
![](https://i.imgur.com/14svUz4.jpeg)
Nesterov momentum 방법은

1. velocity($v_t$)로 방향 이동을 먼저하고($x_t+\rho v_t$)
2. 그 지점에서 gradient를 구하는 것이다.

대부분의 optimization 문제는 현재 지점에 대한 gradient만 계산하기 때문에 velocity vector을 이용하여 간 지점에 대한 gradient를 구하기는 쉽지 않다. 

![](https://i.imgur.com/xIjigeG.jpeg)
그래서 대부분의 api에서는 현재 지점에서의 gradient를 다루는 방식으로 변형하여 사용한다고 한다.

($\tilde x_{t+1}$의 첫번째 공식이 `w`를 업데이트 하는 곳에 사용된 것을 확인할 수 있다.)
![](https://i.imgur.com/BMvywrP.png)

SGD+Momentum과 Nestrov Momentum은  초반에는 가속화하여 매우 빠르게 진행되다가 수렴하는 부근에서는 step이 매우 천천히 진행되는 것을 확인할 수 있다. 또한, optimal에서 overshoot이 발생하는 것을 볼 수 있다.
- overshoot : 경사가 가파른 경우 빠른 속도로 내려오다가 최소지점을 만나면 gradient는 작아지지만 속도가 아직 남아있어 global minimum을 지나치는 현상

### AdaGrad
![](https://i.imgur.com/l8SHwX2.jpeg)
AdaGrad의 방법을 살펴보면 아래와 같다.
1. gradient를 계산한 후에 gradient 값의 제곱한 값을 누적시킨다. 
2. W를 업데이트하는 과정에서 누적된 gradient의 제곱값에 루트를 씌워서 `learning rate * gradient`값을 나눠준다.

즉 과거 전체의 gradient 제곱값을 추적하는 방식이다.

이 방식은 Per-parameter learning rates 또는 adaptive learning rates 방식이라고도 불린다.
![](https://i.imgur.com/mt1aDlm.jpeg)
다시 taco shell을 떠올려보자. 

각 dimension에 대해서 

- gradient가 클 경우(수직방향; "steep" ; y축 방향) : gradient 제곱값은 더욱 커지고, learning rate를 큰 값으로 나누기 때문에 step의 size가 작아지게 된다. -> 원래 가려던 것보다 덜 간다.(**damped(감소된)**)
- gradient가 작은 경우(수평방향; "flat" ; x축 방향) : gradient의 제곱값이 작아지고, learning rate를 1보다 작은 값으로 나누기에 step size가 커지게 된다. -> 원래 가려던 것보다 더 간다.(**accelerated**)

**즉, gradient가 클 경우, step이 작아지고 / gradient가 작은 경우, step이 커진다.**따라서, 이를 통해 진동하는(지그재그로 이동하는) 현상을 완화할 수 있다.

하지만 AdaGrad는 step이 많이 진행되면 진행될수록 결국엔 누적합인 grad_squared가 너무 커져서 gradient descent가 멈춰버릴 가능성이 있다.

-> 이를 해결하기 위한 것이 RMSProp이다.

### RMSProp
![](https://i.imgur.com/ro2z4kh.jpeg)
RMSProp은 Leak AdaGrad라고도 한다.

AdaGrad가 과거 정보를 계속 누적하는 반면에, RMSProp은 현재 gradient를 보장하면서도 grad_squared에 decay rate term을 추가해서 과거 정보를 조금씩 축소시켜서 받아들여서 step이 0이 되지 않게 한다.
![](https://i.imgur.com/905QZbG.png)
SGD + Momentum은 깊이 감소하는 방향으로 많이 갔다가(overshoot) 돌아오는 방식이다.

RMSProp은 loss surface가 taco shell과 같은 찌그러진 모양이어도 많이 치우치지 않고 안정적인 것을 확인할 수 있다.

### Adam
![](https://i.imgur.com/LgYlxgj.jpeg)
Adam은 딥러닝에서 가장 많이 사용하는 optimization 방법 중 하나로, RMSProp과 Momentum이 합쳐진 개념이라고 생각하면 된다.
![](https://i.imgur.com/XN5CrF1.jpeg)
Adam에서는 두가지 변수를 도입한다. 하나는 SGD+Momentum에서 본 velocity(momentum1)이고, 다른 하나는 RMSProp에서 본 Leaky squared_grad(momentum2)이다.

Momentum의 아이디어를 차용한 부분은 빨간색 박스 부분이다.
![](https://i.imgur.com/cM2ljun.jpeg)
RMSProp의 아이디어를 차용한 부분은 파란색 박스 부분이다.
![](https://i.imgur.com/0S9yoz2.jpeg)
Adam에는 약간의 문제가 있다.
Q. Beta2가 0.999일때 t=0에서 어떻게 작동할까?

즉, optimization 초기단계(step=0일때)에서 beta2의 값이 0.999라고 가정하면 moment2가 0에 매우 가까운 값을 얻게 될 것이다.

따라서 $w$를 업데이트 할때, 0에 가까운 수로 나누므로, step이 갑자기 커지게 된다.

![](https://i.imgur.com/cajJgOl.jpeg)
이 문제를 해결하기 위해서 Bias correction을 이용한다고 한다. 이를 통해 moment1과 moment2가 초기에 0으로 편향되어 있는 것을 해결할 수 있다고 한다.
![](https://i.imgur.com/stnUIwJ.jpeg)

Adam은 매우 많은 딥러닝 시스템에서 잘 작동한다.

일반적으로 beta1=0.9, beta2=0.999, learning rate=1e-3, 1e-4, 1e-5로 시작하는 것이 많은 model에서 좋은 starting point라고 한다.
![](https://i.imgur.com/l1x3iut.jpeg)
위와 같은 논문들에서는 모두 Adam을 사용하였지만 각기 다른 learning rate를 적용하였다고 한다.

Adam은 어떤 task든 매우 robust하게 작동하고, 튜닝할 하이퍼파라미터도 많지 않아서 Neural Net을 빌딩할 때 우선적으로 Adam을 적용시키는 것이 좋다.

![](https://i.imgur.com/3hKIKDB.jpeg)
지금까지 배운 Optimization 방법들을 해보면 위와 같이 나타낼 수 있다.
1. 속도를 추적하는가?
2. 제곱 term을 보관하고 있는가?
3. 과거를 잊고 있는가?
4. 초기 moment에 대한 편향을 해결할 수 있는가?



![](https://i.imgur.com/6pVbmlW.jpeg)
지금까지 배운 모든 알고리즘은 First-Order Optimization의 방법이다. (1차 근사 최적화)

![](https://i.imgur.com/Wust3QR.jpeg)
모두 gradient의 정보(first derivative; 일차미분)만을 이용해서 gradient step을 만들기 때문이다.
![](https://i.imgur.com/qxLdlyK.jpeg)

물론 보다 higher-Order Optimization도 존재한다. Second-Order Optimization(2차근사 최적화)이 그 예시이다. 

2차 근사는 2차 다변수 함수의 optimal(global minimum)을 찾아서 이동하는 방식으로, gradient(일차 미분)뿐 아니라 hessian(이차 미분)도 이용한다.
![](https://i.imgur.com/mYAdW2v.jpeg)
이 방식의 장점으로는 flat한 지역(low curvature)이 있을 때, 한번에 많은 step을 이동할 수 있다는 점이다. 즉, loss surface에 대해서 유연하게 대처하는 것이 가능하다.
![](https://i.imgur.com/NDLa2hv.jpeg)
> 참고 : Second Order Optimization은 테일러 근사를 활용해서 위와 같이 기술할 수 있다고 한다.

![](https://i.imgur.com/k9M2mS5.jpeg)
그러나 Second-Order Optimization는 딥러닝 시스템에 사실 실용적이지 않다.
![](https://i.imgur.com/iu9HvGw.jpeg)
2차 근사를 계산할 때는 Hessian matrix가 활용되기 때문이다. 즉, $W$가 $n$개의 차원을 가질 때, $n^2$개의 value들을 가지고 있기 때문에, O($n^2$)의 비용이 들며, 역행렬도 계산해야하기 때문에, 총 O($n^3$)의 계산을 요구한다.

따라서 이를 RAM에 저장하려고 하면, 계산 비용이 많이 발생하고, 공간의 효율성도 매우 떨어지게 된다. 
> 참고 : L-BFGS
> ![](https://i.imgur.com/KEVoV4S.jpeg)
> ![](https://i.imgur.com/YjedK8a.jpeg)
> (1\*) L-BFGS는 Hessian matrix를 계산하거나 저장하기 위한 비용이 합리적이지 않은 경우 사용할 수 있는 방법이며, 밀도가 높은 $n×n$의 Hessian 행렬을 저장하는 대신 $n$차원의 벡터 몇 개만을 유지하여 Hessian 행렬을 추정(approximation)하는 방식이다.
>
> 이 방식은 Full batch에서는 잘 작동하지만, 소량의 샘플을 뽑아서 gradient/hessian matrix를 계산하는 mini-batch 환경에서는 잘 작동하지않는다고 한다.
>
> ref : (1*) <https://wikidocs.net/22155>

## In Practice
![](https://i.imgur.com/AOrjPdj.jpeg)
즉 , 요약을 하자면 다음과 같이 나타낼 수 있다.
- $w^*$를 찾는 최적화 방법으로 Adam을 주로 default case로 활용한다.
- SGD + Momentum이 좋은 성능을 낼 순 있겠지만, 많은 튜닝이 필요하다.
- 만약 full batch를 통한 업데이트를 할 수 있다면, L-BFGS를 활용해보는 것도 좋다.


![](https://i.imgur.com/sDLnzxn.jpeg)
4강까지의 강좌에서는 다음과 같은 내용을 다루었다.

1. image classification problem을 해결하기 위한 Linear Model
2. Loss Function을 통해 가중치 선택에 대한 우리의 preference를 표현하는 방법
3. Loss function을 minimize하고 model을 훈련시키는 SGD방법


![](https://i.imgur.com/lWqDkvL.jpeg)
다음 장에서는 Neural Network에 대해서 알아본다.

## Ref
- <https://ploradoaa.tistory.com/76>
- <https://sites.google.com/view/statml-smwu-2020s>

