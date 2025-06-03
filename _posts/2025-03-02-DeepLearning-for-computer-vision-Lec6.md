---
title: "[EECS 498-007] Lecture6: Backpropagation"

categories: ["EECS 498-007 Deep Learning for Computer Vision"]
tags:
  - ["Deep Learning", "Computer Vision"]

date: 2025-04-06
last_modified_at: 2025-04-06
---

![](https://i.imgur.com/ZPzhU4k.jpeg)
6장에서는 Neural Network를 학습시킬 때 gradient를 계산하는 방법인 Backpropagation(역전파)에 대해서 알아본다.
![](https://i.imgur.com/syUDphU.jpeg)
지난 강의에서는 Neural Network와 관련하여 아래와 같은 것을 배웠다.

- Neural Network의 특징
- Space Warping과 Activation function을 추가하여 non-linear한  표현을 할 수 있는 것
- Universal Approximation이 보장
- Non-convex function으로 Loss surface가 존재함에도, 실전에서 어떻게든 사용이 가능하다는 것

![](https://i.imgur.com/GeQTcOC.jpeg)
문제는 w에 대한 loss의 미분(편미분), 즉 gradient를 어떻게 계산할지에 대한 것이다.
![](https://i.imgur.com/6J9pjv7.jpeg)
손으로 직접 gradient를 계산하려고 하면 매우 많은 수식과 종이가 필요하다. 
loss 식을 바꾸는 경우, 또는 Neural network의 아키텍쳐를 변경할 때마다 수식을 새롭게 변경하고, gradient를 계산해야한다.

따라서 다양한 architecture와 loss function에 확장을 가능하게 하기 위해서 gradient를 구할 때 모듈화된 접근이 필요하다.
> 물론 이제는 gradient를 직접 구할 일은 없다.


## Computational graph
![](https://i.imgur.com/p17PaD3.jpeg)
따라서 아무리 복잡한 모델이 와도 컴퓨터가 계산할 수 있는 모듈을 만들고자 한다. 이 때 모듈 계산을 위해 사용할 수 있는 것이 **computational graph**이다.

- computational graph : 계산그래프; 모델의 연산과정을 표현한 그래프

위의 사진은 선형 변환과 multi-class SVM Loss(hinge Loss)를 computational graph로 나타낸 것이다.

1. $f = Wx$는 matrix-vector multiplication이다. 즉, matrix-vector multiplication 함수에, $W$와 $x$라는 input이 들어가는 형태로 생각할 수 있고, 이때 계산된 아웃풋은 $s$(score)이다.
2. 이후, $s$를 hinge loss 함수에 input으로 넣고, 계산한다.
3. 초록색 R(W)는 Regularization term을 말한다.
4. $L = L_i + R(W)$



***WHY computational graphs?***

Computational graph는 복잡한 모델로 갈수록 **간단한 연산을 그래프 형태로 나타냄으로써, 전체 미분을 계산하는 데에 큰 도움**을 준다.

![](https://i.imgur.com/5Vz61n6.jpeg)
AlexNet과 같은 복잡한 구조의 Neural Network에서도 computational graph를 만들고 모델이 수행하는 모든 연산의 결과를 추적할 수 있다.
![](https://i.imgur.com/fLMWiCb.jpeg)
Neural Turing Machine의 Computational graph는 위와 같이 생겼는데, 이것의 gradient를 직접 계산하는 일은 매우 힘들다. 위의 사진은 한번의 iteration때의 사진을 나타낸 것이며
![](https://i.imgur.com/Nxoa9HU.jpeg)
iteration을 반복할 경우, 매우 복잡한 형태를 가지게 된다. 따라서 모듈화된 gradient를 구할 수 있는 방법인 computational graph가 매우 필요함을 알 수 있다.
![](https://i.imgur.com/ZhlXmqr.jpeg)
deep neural network에서의 backpropagation을 이해하기 위해서, 먼저 간단한 model을 통해서 어떻게 gradient를 구하는지 알아보자.

$x, y, z$라는 3개의 scalar 변수가 있고, $(x+y)z$를 output으로 산출하여 loss를 계산하는 경우를 가정해보자.
- $x = -2, y = 5, z = -4$라고 가정한다.

![](https://i.imgur.com/Mx7bKPf.jpeg)
gradient를 계산하기 위한 1번째 step은 forward pass(순전파)이다.
forward pass : output을 계산한다.

forward pass에서는 output 값을 구하기 위해서 왼쪽에서 오른쪽으로, input 값으로부터 node에 적혀진 연산을 수행하는 것을 말한다.
1. $x$와 $y$를 먼저 더하고, 그 결과를 $q$라고 한다.
	- 즉, `+` 노드에 x와 y를 input으로 넣은 형태이다.
2. 그리고, $q$와 $z$를 곱하여 나온 결과가 $f$다.
	- 즉, `*` 노드에 q와 z를 input으로 넣은 형태이다.

![](https://i.imgur.com/hSGf0Se.jpeg)
gradient를 계산하기 위한 2번째 step은 backward pass(역전파)이다.
backward pass : 미분 값을 계산한다.

backward pass는 output $f$를 각각의 input value에 대해서 미분값($\frac {\partial f}{\partial x}, \frac {\partial f}{\partial y}, \frac {\partial f}{\partial z}$)을 구하기 위해서 오른쪽에서부터 계산을 진행한다.

> 즉, forward pass와는 반대방향으로 진행되기 때문에 Backward Pass라고 부르고, Backpropagation이라는 이름이 붙은 이유이다.

![](https://i.imgur.com/qEI2QyZ.jpeg)
먼저, 초깃값은 $\frac {\partial f}{\partial f}$로 1의 값을 가진다.

backpropagation을 할 때, computational graph의 표현법으로 value값은 위에, gradient 값은 아래에 적는다.

![](https://i.imgur.com/jX7UX65.jpeg)
두번째로 $\frac {\partial f}{\partial z}$를 구하기 위해서는 $f=qz$라는 **국소적 연산**을 살펴보아야한다.

$\frac {\partial f}{\partial z} = q = x+y = 3$이므로,  $\frac {\partial f}{\partial z}$의 값은 3으로 적을 수 있다.
![](https://i.imgur.com/aRBAR0o.jpeg)
동일하게, $\frac {\partial f}{\partial q}$를 구할때도 $f=qz$라는 **국소적 연산**을 살펴보아야한다.

$\frac {\partial f}{\partial q} = z = -4$이므로,  $\frac {\partial f}{\partial q}$의 값은 -4로 적을 수 있다.
![](https://i.imgur.com/MnEKA9e.jpeg)
하지만, input y는 f와 직접적으로 연결되어 있지는 않기 때문에, $\frac {\partial f}{\partial y}$를 계산하기 위해서는 **Chain Rule**(연쇄법칙)을 적용해야한다. 

![](https://i.imgur.com/srnvCBZ.jpeg)
<font color="#c00000">Chain rule은 **합성함수의 미분은 그 합성함수를 구성하는 함수들의 미분의 곱**으로 나타낼 수 있다는 성질이다.</font>

위와 같이 Chain Rule을 적용하여  $\frac {\partial f}{\partial y}$의 값을 구할 수 있다.

![](https://i.imgur.com/JMeTn97.jpeg)
Chain Rule에 의해 계산되는 식에 이름을 붙일 수 있다.
- **Downstream gradient** : 노드의 input 자리에 있는 변수에 대한 gradient; 하류의 gradient
- **Local gradient** : 현재 노드 상의 input에 대한 (현재 노드의) output의 미분 값 
- **Upstream gradient** : 앞쪽 노드에 대한 그래디언트; 상류에서 내려온 gradient; 

수식으로 표현하면 아래와 같다.
**<font color="#c00000">Downstream = Local * Upstream</font>**

이처럼 Chain Rule을 활용하여 상류에서부터 Local Gradient와 Upstream gradient를 곱하면서 Downstream Gradient를 구하고, 이를 하류로 내려보내면서 반복하는 것이 Backpropagation의 과정이다. -> 즉, **<font color="#c00000">국소적 미분들을 Chain Rule을 통해 곱함으로써 모든 미분을 구할 수 있게 되는 것</font>**이다.  
![](https://i.imgur.com/1TU4sRs.jpeg)
$\frac {\partial f}{\partial x}$에 대해서도 Chain Rule을 통해 같은 방식으로 계산된다.

### 국소적 연산
#### forward pass
![](https://i.imgur.com/pKELc5V.jpeg)
Computational graph의 노드 1개를 자세하게 살펴보자. 

이 노드에서는 다른 노드의 부분이나 이전 layer, 다음 layer와 같은 어떠한 정보도 필요하지 않다. 단지 **국소적 연산을 진행할 뿐**이며, 전체적인 계산도, 단순한 국소적 연산이 그래프를 순회하면서 계산함으로써 진행된다.

`f`라는 연산 노드에 input $x$, $y$를 받아서 $f(x,y)=z$를 local output으로 출력한다. 

#### backward pass
![](https://i.imgur.com/kr2U5Tt.jpeg)

1) 맨 끝 노드에서부터 gradient값이 하류로 전달되고, 현재 노드에 들어오게 된다.

![](https://i.imgur.com/R975ZLQ.jpeg)

2) $z = f(x,y)$를 이용하여 현재 노드의 input에 대한 local output의 미분값인 Local gradients를 계산한다.
![](https://i.imgur.com/FXbW258.jpeg)
3) Chain rule을 활용하여 Local gradients와 상류에서 내려온 Upstream gradient값을 곱해서, Downstream gradients를 얻는다.
![](https://i.imgur.com/ZkrtBbn.jpeg)
4) 계산한 Downstream gradients는 하류로 전파되며 하류의 노드에서 upstream gradient로 작동하게 된다.

이때 Downstream gradients가 다른 노드에서는 어떻게 쓰이는지는 전혀 상관하지 않은 채로 국소적인 범위에서만 계산하여도, 단순히 전파만 시킨다면  Network의 모든 gradient를 계산할 수 있다.

> 이처럼 Computational graph를 이용하면 gradient를 구하는 과정을 모듈화시킬 수 있다.

### Another Example
![](https://i.imgur.com/qWAoalO.jpeg)
Computational graph에 대한 다른 예시를 확인해보자.
![](https://i.imgur.com/XYpzWsZ.jpeg)
forward pass에서는 $w_0, x_0$와  $w_1, x_1$을 각각 내적하고, bias term인 $w_2$를 더할 것이다. 이후에는 $f(x,w)$의 식과 같이, 가중합을 한 것을 sigmoid function에 input으로 넣어주고, output을 출력한다.

결과값은 모두 scalar 값이다. 
![](https://i.imgur.com/wt21Oz1.jpeg)
Backward pass에서는 최종 output(score)으로부터 계산된 국소적 미분값이 최초의 Upstream gradient가 되어 하류의 노드들로 전파된다.

맨 처음(Base case)의 미분 값은 자기 자신에 대한 미분이기 때문에 항상 gradient가 $1$이다.
![](https://i.imgur.com/2DSo1BM.jpeg)
***Local gradient***

$$\frac{\partial 1}{\partial x} = \frac{1}{-x^2} =-\frac{1}{(1.37^2)}= -0.53 $$ 

***Upstream gradient***

$$1$$

***Downstream gradient***

Downstream = local x upstream = (-0.53) x 1  = -0.53

![](https://i.imgur.com/TehbARw.jpeg)
***Local gradient***

$$1$$

***Upstream gradient***

$$-0.53$$

***Downstream gradient***

Downstream = local x upstream = 1 x (-0.53) = -0.53

![](https://i.imgur.com/ECHtNEb.jpeg)
***Local gradient***

$$e^{-1}$$



***Upstream gradient***

$$-0.53$$



***Downstream gradient***

Downstream = local x upstream = e^-1 x (-0.53) = -0.20
![](https://i.imgur.com/MBM10iX.jpeg)
***Local gradient***

$$-1$$



***Upstream gradient***

$$-0.20$$



***Downstream gradient***

Downstream = local x upstream = -1 x (-0.20) = 0.20
![](https://i.imgur.com/5HkPS5h.jpeg)
아래에서도 동일한 방식으로 계속 Backward propagation 하면된다.
![](https://i.imgur.com/70z6cxb.jpeg)

![](https://i.imgur.com/F9wr9xm.jpeg)

![](https://i.imgur.com/zE7nDDr.jpeg)

![](https://i.imgur.com/J0wqm9b.jpeg)
흥미로운 점은 Computational graph를 여러 형태로 나타낼 수 있다는 것이다(not unique). 강의의 예시처럼, 4개의 연산 노드를 1개의 sigmoid 연산 노드로 합칠 수 있다.

즉, 함수 여러개를 sigmoid 함수 노드 하나로 사용할 수 있게 되는 것이다. 
![](https://i.imgur.com/G7SNOX2.jpeg)
압축적으로 graph를 그리면 local(국소적) gradient를 계산하기 어려울 수 있지만, 시그모이드 함수를 미분하면 단순히 **(1 - 시그모이드 함수의 출력값) * (시그모이드 함수의 출력값)**의 형태를 띈다. 

-> 따라서 local gradient 계산시에 시그모이드 함수의 출력값($\sigma(x)$)만 있어도 계산할 수 있다.

이처럼 파란색 부분에 대해서 local gradient를 매우 쉽게 계산할 수 있기 때문에,  계산을 쉽게 할 수 있는 부분에 대해서는 backward pass를 효율적으로 하기 위해 graph를 구축할 수 있다.

![](https://i.imgur.com/5tttomB.jpeg)
그래서 파란색 chunk의 좌우 input, output을 통해서 Backward pass를 수행할 수 있고, 중간 노드의 계산 과정은 모두 생략해도 된다.

단, local gradient를 계산할 때 $x$가 아닌 시그모이드 함수의 출력값($\sigma(x)$)을 대입해야함을 주의하자.
## Backpropagation의 Pattern
Computational graph에서 수행하는 각 노드의 backward propagation 연산에는 어떠한 pattern이 존재한다.
### ***add gate***  : gradient distributor
![](https://i.imgur.com/xYpfL3r.jpeg)

$$z = x + y$$

local gradient는 아래와 같다.



$$\frac{\partial z}{\partial x} = 1$$



$$\frac{\partial z}{\partial y} = 1$$

즉, 각 변수에 대해서 local gradient의 값은 둘 다 1이다.

따라서,  <font color="#c00000">upstream gradient를 하류로 그대로 전파하는 역할</font>(local * upstream)을 하며, 노드가 갈라질 때는, upstream gradient를 각 노드로 분배한다.

### ***copy gate*** : gradient adder
![](https://i.imgur.com/w0ypb8W.jpeg)

$f(x)  = (x, x)$로, 1차원을 2차원으로 보내는 함수이다. 따라서, Upstream gradient가 2차원인 경우이다.

local gradient는 아래와 같다.



$$\frac{\partial f}{\partial x} = (1, 1)$$

우리가 알려고 하는 것은 x에 대한 loss의 미분값이므로 식을 정리하면 아래와 같다.



$$\frac{\partial L}{\partial x} = \frac{\partial f}{\partial x} \cdot \frac{\partial L}{\partial f} = (1, 1) \cdot \frac{\partial L}{\partial f} = (\frac{\partial L}{\partial f})_1 + (\frac{\partial L}{\partial f})_2$$

결과적으로, upstream gradient를 더해주는 효과를 갖는다.

> 이러한 연산을 수행하는 이유는 regularization term을 추가해야할 때, weight에 대한 동일한 정보를 흘려야하기 때문에 수행한다고 한다. 또한, 동일한 weight로 서로 다른 task를 진행할 때도 weight의 정보를 복사하기 위해서 이러한 branch를 사용한다고 한다. 서로 다른 목적으로 사용되었기 때문에, backward pass동안 다른 gradient값이 전파되고 있는 것을 확인할 수 있다. (4와 2)
> 
> add gate와 copy gate는 정확하게 반대의 역할을 한다.(add gate의 forward = copy gate의 backward)

### ***mul gate*** : swap multiplier
![](https://i.imgur.com/5EhTF3h.jpeg)

$$z = x \cdot y$$



local gradient는 아래와 같다.



$$\frac{\partial z}{\partial x} = y$$



$$\frac{\partial z}{\partial y} = x$$



결과적으로, <font color="#c00000">forward pass때의 각각의 input값을 서로 뒤바꾸고(swap), upstream gradient 값에 이를 곱하여(multiplier) 전파시키는 역할</font>을 한다.

따라서, mul gate에서 **local gradient는 각 input값을 서로 바꾸어주기만 하면 된다**.
### ***max gate*** : gradient router
![](https://i.imgur.com/nC7XTDu.jpeg)

$$f(x) = max(x, y) = \begin{cases} x & \text{if x > y} \\ y & \text{if x > y} \end{cases}$$



local gradient는 아래와 같다.



$$\frac{\partial f}{\partial x} = \begin{cases} 1 & \text{if x > y} \\ 0 & \text{if x < y} \end{cases}$$



$$\frac{\partial f}{\partial y} = \begin{cases} 0 & \text{if x > y} \\ 1 & \text{if x < y} \end{cases}$$



따라서, <font color="#c00000">노드의 input인 x와 y중 max값인 쪽으로만 upstream gradient를 전파시키고(gradient router)</font> 그 외에는 모두 0(zero gradient)을 전파시킨다.

> 모델에 max gate가 많아지면 zero gradient를 자주 흘리게 되고, gradient flow에 문제가 발생하기 때문에 max gate를 사용하는 것은 자주 권장되지는 않는다고 한다.

## Backprop Implementation
Pytorch에서 Backpropagation을 구현하는 방법을 살펴보자.
### Flat gradient code
![](https://i.imgur.com/gh0ruwL.jpeg)

Backpropatation의 implementation으로는 크게 두가지가 있는데 하나는 "Flat"한 버전이라고 불리는 코드이다.

우선 forward pass로 output을 계산한다.
![](https://i.imgur.com/iW2dyuj.jpeg)
backward pass에서는 base case(gradient=1)부터 시작하여, forward pass의 반대방향으로 gradient를 계산하는 코드를 이어서 적으면 된다.
![](https://i.imgur.com/4vI6hoQ.jpeg)

![](https://i.imgur.com/G5UOFtW.jpeg)

![](https://i.imgur.com/zlAPlcC.jpeg)

![](https://i.imgur.com/IPMFj2F.jpeg)

![](https://i.imgur.com/BsDIYCr.jpeg)
이처럼 배운 rule 대로 forward pass를 적고, backwardpass를 적으면 된다.
![](https://i.imgur.com/OxqXWTd.jpeg)
2번째 과제가 flat version의 backpropagation 코드를 적는 것이다.
![](https://i.imgur.com/RgN4En5.jpeg)
### modular API 
![](https://i.imgur.com/Q3KM9yV.jpeg)
현장에서 많이 사용되는 implementation 방법은 modular API 방식으로, class 하위에 forward pass와 backward pass를 정의하는 형태이다. 연산 하나하나에 대해 모둘화 시키는 경우, 식이 조금 바뀌었을 때 코드를 수정하기가 훨씬 수월해진다.

> 위의 예시는 실제로는 동작하지 않는 pseudo code이다.

![](https://i.imgur.com/AQxFvCX.jpeg)
이미 pytorch에서는 연산에 대해 구현해둔 모듈들이 존재한다. `torch.autograd.Function`을 상속함으로써 computational graph의 연산에 대한 노드 객체들을 만들 수 있다. 위의 예시는 mul gate에 대해 forward와 backward 과정을 구현해둔 코드이다.
![](https://i.imgur.com/nN4ZK1j.jpeg)
즉, Pytorch는 작은 함수들에 대해 forward pass와 backward pass가 기술된 Autograd Engine이다.

![](https://i.imgur.com/25shNbU.jpeg)
Pytorch에서 사용하는 sigmoid layer의 코드를 살펴보자.

![](https://i.imgur.com/9ig1xXh.jpeg)
forward pass 부분을 살펴볼 수 있으며, 이것에 대한 구체적인 코드는 어디에선가 정의되어 있을 것이다.

![](https://i.imgur.com/TSPyRrD.jpeg)
backward pass를 계산하는 부분에서 우리가 봐온 부분을 그대로 사용하고 있음을 확인할 수 있다.

## Backprop with Vector
![](https://i.imgur.com/YLgS0G7.jpeg)
지금까지 미분을 구조적으로 어떻게 구현하는지 살펴보았다. 이제까지는 모두 scalar에 대한 미분만을 다루었지만, 실제로 우리는 vector를 다루어야하는 경우가 많다. 이제 vector 혹은 matrix 형태가 들어오면 어떻게 처리하는지를 알아보자. 
![](https://i.imgur.com/kL6AYRO.jpeg)
기존에 배운 미분은 input과 output이 모두 scalar인 경우였으며, 미분(gradient)도 scalar로 정의가 되어있었다. 
> gradient는 x가 변화할때, y가 얼마만큼 변하는지를 의미한다.

![](https://i.imgur.com/e01BfNT.jpeg)
가장 많이 보는 형태로는 output은 scalar, input은 vector인 경우이다. 
이 경우의 미분 값 **gradient**로 표현되며, gradient는 input과 dimension의 수가 같다. 

gradient의 n번째 element : n번째 input이 조금 변화했을 때, 전체 output이 얼마나 변하는지에 대한 값 = $\frac{\partial y}{\partial x_n}$

![](https://i.imgur.com/YpnDvBK.jpeg)
input과 output이 모두 vector인 경우에, 미분(derivative)은 Jacobian matrix로 표현된다. jacobian의 (n,m) component는 n번째 input이 변화하는 경우, m번째 output의 변화정도를 의미한다.

$$\frac{\partial y_m}{\partial x_n} = (\frac{\partial y}{\partial x})_{n,m}$$


> 단, 어떤 경우에도 우리가 최종적으로 구해야하는 'Loss'는 scalar 값을 가지는 것을 기억하자.

![](https://i.imgur.com/WuAkrZQ.jpeg)
$x, y, z$ : vector

- x와 $D_x$ 
- y와 $D_y$
- z와 $D_z$

위의 gradient는 input값과 dimension이 같다.
![](https://i.imgur.com/VPAWeYj.jpeg)
gradient는 input값과 dimension이 같기 때문에, upstream gradient는 똑같이 $D_z$차원, downstream gradient는 각각 $D_x$, $D_y$차원을 가지게 된다.
![](https://i.imgur.com/S7WqvFO.jpeg)
Local gradient를 구해보자. 다변수 input, 다변수 output을 가지기 때문에 jacobian matrix의 형태로 나타난다.
![](https://i.imgur.com/m3Wdlw2.jpeg)
따라서, downstream을 구하는 과정은 matrix-vector multiply로 나타나게 된다.

-> downstream = local(jacobian 행렬) x upstream(벡터)
![](https://i.imgur.com/QLg41Sj.jpeg)
하지만, jacobian matrix가 매우 커질 수 있기 때문에,  그대로 메모리에 사용하는 것이 아니라 연산의 패턴을 파악하고, 메모리를 절약할 수 있는 방향으로 구현하여야한다. 

위와 같은 연산을 진행하는 경우를 생각해보자.

input과 output의 dimension은 모두 4차원이며, elementwise하게 max를 취하는 경우이다. 이 경우 negative value들은 모두 0으로 바뀔 것이다.

![](https://i.imgur.com/QyMP19U.jpeg)
Backprop을 진행한다고 가정했을 때, output과 같은 4-dimension의 vector가 하류로 전파된다.


![](https://i.imgur.com/yZdH9VO.jpeg)
$\frac{\partial L}{\partial x}$를 구하기 위해서, local gradient인 jacobian matrix를 구해야한다. 
이 경우에,  jacobian matrix의 비대각 element들은 전부 0의 값을 가지게 된다.

그 이유는 뭘까?

위에서 설명했듯, jacobian의 input은 n, output은 m일 때, jacobian matrix의 각 element는 아래와 같다.



$$\frac{\partial y_m}{\partial x_n} = (\frac{\partial y}{\partial x})_{n,m}$$



따라서,  input의 1번째 element에 대한 output의 1번째 element의 미분값은 jacobain matrix의 (1,1) element에 나타날 것이다. 

즉, 이는 <font color="#c00000">각 output의 gradient가 대응하는 input에만 영향을 준다</font>는 것을 나타낸다. 이럴 때는 jacobian matrix의 diagonal matrix이 된다.

![](https://i.imgur.com/XY0wKm3.jpeg)
또한, (1,1)과 (3,3) element는 모두 1의 값을, (2,2)와  (4,4) element는 0의 값을 가지는 것을 알 수 있는데, 이는 $max(0,x)$의 미분식에 따라 input값 $x > 0$인 경우에만 1의 미분 값을 가지기 때문에, 위와 같은 대각성분들을 가진다.  

위의 matrix-vector multiplication은 downstream을 구하기 위한 식을 표현한 것과 같다.
![](https://i.imgur.com/kxdrwcY.jpeg)
결과적으로 **downstream gradient는 jacobian matrix의 대각성분이 1인 행에 대해서는 upstream gradient를 가지고 오고, 0인 행에 대해서는 0의 값을 가지고 온다.** (upstream gradient를 가지고 오지 않는다.)
![](https://i.imgur.com/4B5fIjW.jpeg)
하지만 이 경우에 jacobian은 매우 sparse한 형태이다. 비대각 요소들이 전부 0이므로, 코드를 짤 때 메모리에 모두 업로드 하는 것은 매우 비효율적임을 알 수 있다.
![](https://i.imgur.com/P6jFieR.jpeg)
따라서, 이 경우에는 명시적으로 jacobian matrix form을 써주는 것보다는 implicit multiplication이라고 불리는 것을 이용한다.

implicit multiplication : input data(x)가 양수인지, 음수인지에 따라서 upstream gradient를 그대로 유지할지, 0으로 truncate(절삭)할지 정하는 것

Pytorch code로는 아래와 같이 작성할 수 있다고 한다.
```python
dx = torch.zeros_like(x)    # x는 input vector이다.
dx[x > 0] = dy[x > 0]    # x가 non-zero의 값을 가지는 경우를 true로 하여, dy의 값을 대입한다.
```

## Backprop with Matrices(or Tensors)
![](https://i.imgur.com/sBzz4ol.jpeg)
input과 output이 모두 vector가 아닌, matrix(또는 tensor)인 경우에도 같은 개념으로 구현할 수 있다고 한다.

![](https://i.imgur.com/PpWZdkx.jpeg)
> Loss의 값은 항상 scalar로 유지되며, $\frac{\partial L}{\partial x}$는 항상 $x$와 같은 shape을 가지는 것을 주의해라.  

![](https://i.imgur.com/7k0I7gV.jpeg)
z와 동일한 형상을 가진 미분 값이 upstream gradient로 내려오는 상황이다.
![](https://i.imgur.com/kiG70ul.jpeg)
이때는 **matrix를  모두 벡터로 펼쳐서 생각해보면 된다.** 
이 경우, input과 output은 아래와 같이 변환된다.
- input x : $D_x \times M_x$ 크기의 vector
- input y : $D_y \times M_y$ 크기의 vector
- output z :  $D_z \times M_z$ 크기의 vector

즉, vector의 backprop과정을 수행하게 되며, 이 경우, Local gradient는 여전히 jacobian matrix로 나타날 것이다. 
![](https://i.imgur.com/Mip9R8l.jpeg)
따라서 downstream gradient는 여전히 matrix-vector간 곱셈이 된다.
하지만 matrix가 backprop 되는 경우에도, 여전히 jacobian matrix의 메모리 문제를 가지고 있으며, local gradient가 어떠한 pattern으로 계산되는지 알아야한다.
![](https://i.imgur.com/fw86xwM.jpeg)
forward pass로 행렬 간 곱셈을 계산한 것이다.
![](https://i.imgur.com/AK9bWAb.jpeg)
backprop의 목표가 x와 형상이 같은 미분값 $\frac{\partial L}{\partial x}$을 구하는 것이라고 하자.

우선, output인 y와 형상이 같은 미분 값  $\frac{\partial L}{\partial y}$이 upstream gradient로 전달될 것이다. 
![](https://i.imgur.com/HwY14Kh.jpeg)

- $N$ : mini batch size
- $D$ : input data의 dimension
- $M$ : Neural Network의 hidden layer의 수

메모리를 줄이기 위해 Jacobian을 명시적으로 사용하는 것보다는, implicit multiplication을 수행해야할 것이다. 

![](https://i.imgur.com/NpqT9yj.jpeg)
우선, $\frac{\partial L}{\partial x}$의 (1,1) element를 먼저 찾아보자.

downstream을 구하는 공식을 적용하면 위와 같이 나타날 것이다. (chain rule 이용)
![](https://i.imgur.com/jOfLdS6.jpeg)
이 때, $\frac{\partial y_{1,1}}{\partial x_{1,1}}$부터 차례대로 값을 구해보자. 
![](https://i.imgur.com/DvvPSRO.jpeg)
x와 w의 matrix-matrix multiplication을 수행하면, $y_{1,1}$은 위와 같은 식으로 구성된다.(x의 1행과 w의 1열)
![](https://i.imgur.com/NKCzTsW.jpeg)
하지만 $x_{1,1}$에 대해 편미분을 수행하는 경우, 결국 $y_{1,1}$과 관련있는 부분은 $x_{1,1} \cdot w_{1,1}$이며, 이는 $w_{1,1}$의 값과 같다.
![](https://i.imgur.com/nqbOMi7.jpeg)
같은 방식으로 계산해보는 경우,  local gradient부분의 2번째 element는 아래와 같이 나타날 것이다.
![](https://i.imgur.com/xVou2Lq.jpeg)

![](https://i.imgur.com/qt3qphV.jpeg)

![](https://i.imgur.com/Z7FclVL.jpeg)
이를 통해, 나머지 부분들도 1행의 나머지 element들과 같음을 쉽게 확인할 수 있다.

![](https://i.imgur.com/de3LJtR.jpeg)
2번째 행의 계산부터는, $x_{1,1}$의 값이 존재하지 않으므로, 남은 모든 element가 0으로 채워지게 될 것이다.
![](https://i.imgur.com/N6mkfTx.jpeg)

![](https://i.imgur.com/v7xJGr6.jpeg)

![](https://i.imgur.com/4ST0YHp.jpeg)
최종적인 결과가 위와 같음을 확인할 수 있다.

<font color="#c00000">즉, 행에서 $x_{1,1}$과 곱해지는 부분이 자리를 채우게 되는 것이다.</font>

![](https://i.imgur.com/jOuL8OH.jpeg)
$\frac{\partial L}{\partial x}$의 (1,1) element를 구하기 위해 대입해보면, downstream을 구하는 공식에 $w_{1,:}$ (1행 전체)의 행이 그대로 사용되는 것을 알 수 있다. 이것에 upstream gradient를 곱하기만 하면 된다.
![](https://i.imgur.com/MDBoc7w.jpeg)
또 하나의 예제를 들어보자. $\frac{\partial L}{\partial x}$의 (2,3) element를 구하기 위한 예제이며, 마찬가지로 $x_{2,3}$과 곱해지는 w의 부분이 local gradient slice가 될 것이다.
![](https://i.imgur.com/d4if96l.jpeg)
해당 예제의 결과는 위와 같다.
![](https://i.imgur.com/mJX4Ycd.jpeg)
즉 일반화해보자.

$\frac{\partial L}{\partial x}$의 값을 구할때, local gradient와 upstream gradient를 곱해서 구할 수 있다.

local gradient를 구할 때, $x_{i,j}$와 관련있는 미분 값을 구하는 경우에는 $w_{j,:}$를 가져오면 된다는 것을 알 수 있다. 

> 이 방식을 암기하지 말고 차근차근 보면서 이해하는 것이 더 좋은 것 같다.


![](https://i.imgur.com/7guLmF3.jpeg)
이 일반화한 식을 재조합해보면 위와같이 나타난다.
이 방법을 쉽게 기억하는 방법은 "어떻게든 형상이 잘 나오는"방법으로 matrix를 조합하는 것이다.

예를 들어, $\frac{\partial L}{\partial x}$의 경우에, 우리는 NxD의 형상을 만들어야하므로, upstream에 어떤 형상을 곱해야 NxD의 형상이 나오는지 생각해보면, $w^T$를 곱하는 것이 바람직하다는 것이다. 
![](https://i.imgur.com/jgaken0.jpeg)
그 반대로, $\frac{\partial L}{\partial w}$를 구할 때에는 $x^T$를 곱하면 원하는 형상이 나올 것이다.

> 앞의 예시를 차근차근 구해보면 왜 굳이 transpose를 시키는지 알 수 있다.
> transpose 시킴으로써,  내적을 구하던 방식에서 항상 matrix-vector간의 multiplication을 하는 방식으로 변환할 수 있기 때문이다.


## Advanced topics
즉, backpropagation는 미분을 편하게 하기 위한 것이며, 어떻게 더 쉽게 할 수 있을지를 알아보자.
### input, output이 모두 vector인 경우
![](https://i.imgur.com/bJG6frv.jpeg)
Loss는 scalar 값이며, $x_0$~$x_3$이 vector인 상황을 생각해보자. 목표가 $x_0$에 대한 loss의 미분값을 구한다고 생각하면, 위와 같이 chain rule로 나타낼 수 있다.
![](https://i.imgur.com/0qBfFGy.jpeg)
forward가 vector간 곱셈이었으므로, backprop시에는 jacobian matrix의 형태로 미분값들이 나타날 것이다. 이때, 행렬 곱셈은 결합법칙이 성립하기 때문에, 어떤 순서로 결합해주어도 상관이 없다.
![](https://i.imgur.com/GoPtq5n.jpeg)
계산을 할 때도 오른쪽에서 왼쪽으로 계산하면, matrix-vector간 multiplication이 계속 진행되는 것을 알 수 있고, 결과는 $D_0 \times 1$의 vector로 나타나는 것을 알 수 있다. 즉, 4개의 곱을 matrix-vector간 multiplication으로 연산할 수 있게 되는 것이다.
![](https://i.imgur.com/lgKYOlp.jpeg)
반대방향으로 연산을 한다면, 행렬간 곱셈이 되기 때문에 비효율적이다.

### input이 scalar인 경우
![](https://i.imgur.com/CW34C87.jpeg)
다만, 이번에는 input이 scalar인 경우이다.
![](https://i.imgur.com/Y2wM4rY.jpeg)
input이 scalar이고, output이 vector(다변수)이고, scalar에 대한 output의 변화량($\frac{\partial x_3}{\partial a}$)을 보고 싶은 경우라면 앞에서부터 순서대로 곱해주는 것이 좋다.

>아래의 hessian matrix부분은 필요할 때 직접 찾아보자.

![](https://i.imgur.com/R4RCPkE.jpeg)
backprop을 이용해서 Hessian matrix도 계산해낼 수 있다고 한다.
![](https://i.imgur.com/EaSLkhz.jpeg)

![](https://i.imgur.com/hRcGfG7.jpeg)

![](https://i.imgur.com/yf1sf7Y.jpeg)

![](https://i.imgur.com/rfEdvJL.jpeg)

![](https://i.imgur.com/fP35ADU.jpeg)

![](https://i.imgur.com/3ojrbCC.jpeg)

![](https://i.imgur.com/hhBg826.jpeg)
즉 미분을 계산할 때, 
1. 함숫값 계산을 먼저 한 후(forward propagation)
2. 역순으로 해당 노드에 대한 미분을 계산한다(Backpropagation)



![](https://i.imgur.com/VfJuVvI.jpeg)
코드로 구현하는 경우에도, 많은 경우에 forward와 backward가 module로써 구현되어 사용된다.
![](https://i.imgur.com/N1LGgRK.jpeg)
하지만, 우리는 이제까지 input image를 flatten함으로써 image의 structure를 무시하는 방법을 택했었다. 이를 해결하기 위해 CNN을 사용한다.
![](https://i.imgur.com/a2Itghw.jpeg)
다음시간에서는 Convolutional Neural Network에 대해 배운다.

## Ref
- <https://ploradoaa.tistory.com/76>
- <https://sites.google.com/view/statml-smwu-2020s>
