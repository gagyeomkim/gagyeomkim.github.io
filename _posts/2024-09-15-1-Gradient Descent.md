---
layout: single
title: "Gradient Descent"
categories: ['Machine Learning']
tag: Machine Learning Specialization
typora-root-url: ../
---



> Coursera의 Machine Learning Specialization강의를 정리한 내용입니다.

## Gradient Descent(경사 하강법)

Gradient Descent(경사하강법)은 Cost Function을 최소화하는데 사용할 수 있는 알고리즘이다.

이 알고리즘은 Linear Regression 뿐 아니라 대부분의 머신러닝의 모든 Cost Function을 최소화하는데 실제로 사용되는 알고리즘이다.

<br>

이전 게시물에서, 우리는 Cost Function J를 최소화하는 것을 목표로 삼았었다.

<br>

$$ Want\,min\,J(w,b)$$

<br>

>  물론, parameter가 두개 이상인 cost function에도 gradient descent가 사용될 수 있다.
>
> 예를들어 $w_1,w_2$에서 $w_n,b$까지의 cost function J를 사용한다고 하자. 우리의 목표는 Cost Function J를 최소화하는 것이며, Gradient descent를 통해 이것을 수행할 수 있다.
>
> $$ Want\,min\,J(w_1,w_2,...,w_n,b) $$

<br>

Gradient Descent 알고리즘은 다음과 같은 방법으로 진행된다.

1. Start with some $w,b (set \ w=0, b=0)$ 선형회귀에서는 초기값이 얼마인지는 그다지 중요하지 않으므로 $w$,$b$ 두 값 모두 0으로 설정하는 것이 일반적이다.
2. Keep changing $w,b$ to reduce $J(w,b)$
3. Until we settle at or near a minimum

**활모양이나 해먹모양이 아닌 일부 함수 J의 경우 최솟값이 두 개 이상일 수 있다**는 점을 주의하자.

![image-20240915003917149](https://1drv.ms/i/s!AvDtmE0jTiDWgiFnhKHDFCCCF0dT?embed=1&width=1133&height=632)

위의 사진은 신경망 모델을 훈련시킬 때 얻을 수 있는 일종의 Cost Function이다.

높이가 낮을수록 Cost Function의 값이 작아지며, $w$와 $b$의 값이 달라지면, 이 표면에는 다른 cost function 지점이 나오게 된다.

<br>

최대한 효율적으로 아래로 내려가기 위해서는 **가장 가파른 경사로 내려가는 것**이 좋은 방법일 것이다.

-> 이것이 Gradient descent라고 불리는 이유이다.

<br>

또한, 첫번째와 두번째 최솟값을 **local minima**라고 부른다.

-> 왜냐하면 첫번째 길로 하강시 두번째 minima를 찾을 수 없고, 그 반대도 똑같이 적용되기 때문에 해당 최솟값은 **지역적 특성**을 가진다.

<br>

이처럼 초기값에서 시작을 해서 주위를 둘러보며 minima로 향하는 가장 빠른 길을 내려가며 minima를 찾아내지만, 만약 초기값이 다르다면 다른 결과(different local minima)를 도출할 수도 있다.

------

### - Gradient Descent algorithm

![image-20240915004011213](https://1drv.ms/i/s!AvDtmE0jTiDWgiJuGP7PqlzmXFOF?embed=1&width=1080&height=438)

Gradient descent algorithm을 수학적으로 정의하면 위와 같다.

각 단계에서 파라미터 w,b는 새롭게 업데이트 된다.

<br>

$=$: 대입(할당) 연산자로 쓰였다.

$\alpha$ : Learning Rate

시작점에서 minima를 찾아나갈 때, 얼마나 큰 step으로 찾아나갈 것인지를 결정하는 **양의 상수**이다. 이 값이 클수록 한 step에 움직이는 정도가 커진다. **일반적으로 0과 1사이의 작은 양수이며 0.01정도를 채택**한다.

$\frac{\partial}{\partial w}J(w,b)$ : Derivative Term(미분 계수, 도함수의 값)

step을 밟을 **방향을 결정**한다.

------

<br>

gradient descent 알고리즘의 경우, **수렴(convergence)할 때까지 위의 업데이트 단계를 반복**해야한다.

수렴은 파라미터 w와 b가 더이상 많이 변하지 않는 local minima에 도달한다는 뜻이다.

또한, w와 b를 **동시에(simultaneously)** 업데이트 해야한다.

<br>

아래 사진은 코드상에 작성시 올바른 예와 올바르지 않은 예를 보여준다.

`tmp_w`와 `tmp_b`는 기존 w의 값이 변하지 않도록 새롭게 선언한 변수이다.

<br>

잘못된 예시에서는 `tmp_b`를 계산할 때, 새롭게 업데이트 되어버린 `w`가 `J`에 사용되고 있음을 알 수 있다.

따라서 왼쪽의 예시처럼 **동시에 update**를 해주어야한다.

![image-20240915004107233](https://1drv.ms/i/s!AvDtmE0jTiDWgiVSDFR3sU-3xUoL?embed=1&width=1135&height=303)

------

### - Gradient Descent Intuition(경사하강법 직관)

gradient descent가 어떤 역할을 하는지, 왜 의미가 있는지 알아보자

![image-20240915004122957](https://1drv.ms/i/s!AvDtmE0jTiDWgib_u6QSbiO1lzDz?embed=1&width=1131&height=249)

먼저, 오른쪽처럼 파라미터 1개만 사용하는 예시로 단순화해서 알아보자

단순화시 $w$는 오른쪽과 같은 표현식을 얻게 된다.

![image-20240915004133778](https://1drv.ms/i/s!AvDtmE0jTiDWgicX5659UO1N6y0F?embed=1&width=1131&height=363)

$\frac{\partial}{\partial w}J(w,b)$는 $w$에 대해 $J(w)$에 편미분을 취한 편도함수이다.

**한 지점에서** **도함수를 생각해보는 방법은 접선을 그려보는 것**이며,

**접선의 기울기**는 **함수 J의 해당 지점에서의 미분 계수(=도함수의 값)**와 같다.

learning rate는 항상 양수(positive number)의 값을 가지므로 미분 계수가 양수이면, w는 기존의 값보다 더 작은 값으로 update된다.

![image-20240915004603298](https://1drv.ms/i/s!AvDtmE0jTiDWgjN4mbY75S1kUBme?embed=1&width=1099&height=301)

반면, **미분 계수가 음수(negative number)**라면 w는 기존의 값보다 더 큰 값으로 update된다.

**(- 음수 = 양수이므로)**

<br>

이처럼 gradient descent는 w를 변경해가면서 cost의 최솟값에 가까워질 수 있다.

------

### - Learning Rate

Learning Rate에 대한 선택은 gradient descent 구현의 효율성에 큰 영향을 미친다.

![image-20240915004200162](https://1drv.ms/i/s!AvDtmE0jTiDWgigHWcZuQYxl02Wp?embed=1&width=1138&height=589)

Learning Rate가 너무 작다면 수렴(Convergence)하는데에 매우 느리고, 너무 크다면 최솟값에 이르지 못해 수렴하지 못하거나 발산(diverge)하는 문제가 발생할 수 있다. 따라서 적절한 Learning rate를 선택하는 것은 매우 중요하다.

![image-20240915004211413](https://1drv.ms/i/s!AvDtmE0jTiDWginZf9TKTvjFShRp?embed=1&width=1133&height=637)

파라미터를 통해 local minimum에 도달했다면, 미분계수 $\frac{\partial}{\partial b}J(w,b)$의 값이 0이 나오기 때문에, **파라미터는 더 이상 update 되지 않으며, 값을 유지하게 된다.**

![image-20240915004229727](https://1drv.ms/i/s!AvDtmE0jTiDWgirRuHL4eIsbJhVs?embed=1&width=1133&height=551)

또한, learning rate가 특정한 값으로 고정되었어도, gradient descent 알고리즘은 local minimum에 도달할 수 있다.

1. 분홍색 지점을 초기 w의 값으로 세팅했다고 하자. 해당 지점에서의 미분계수(기울기)는 꽤 크기 때문에 gradient descent시 비교적 큰 step을 거치게 된다.
2. update된 두번째 단계에서는 기울기가 첫번째 지점만큼 가파르지 않으므로, 보다 작은 step을 거친다.
3. 이처럼 미분계수는 점점 작아지게 되며, gradient descent를 실행할 때에는 local minimum에 도달할 때까지 점점 더 작은 step을 밟게 된다.

<br>

즉, 최솟값 근처까지 아래와 같은 단계가 **자동으로** 수행된다.

- 미분 계수(기울기)의 값은 점점 작아지며
- Update step은 점점 더 작아진다.

<br>

대부분의 경우에 local minimum에 가까워질 수록 미분계수의 값이 0에 가까워지면서 점진적으로 업데이트 되기 때문에 Learning rate를 수동으로 계속 변경하지 않아도 된다.

------

## Gradient Descent for Linear Regression(선형회귀에 경사하강법 적용)

이제 gradient descent를 linear regression에 적용해보자

![image-20240915004250743](https://1drv.ms/i/s!AvDtmE0jTiDWgiudXV0-nYt0Zg56?embed=1&width=1132&height=595)

Gradient Descent algorithm에서 w와 b에 대한 미분계수를 오른쪽과 같은 식으로 바꿔쓸 수 있다.

해당 식은 편미분을 사용해서 도출된다.

![image-20240915004304599](https://1drv.ms/i/s!AvDtmE0jTiDWgix-axLOWZRitIge?embed=1&width=1138&height=563)

$\frac{\partial}{\partial w}J(w,b)$의 경우, J(w,b)는 아래의 식과 같음을 확인했고, 이를 해당 식에 대입한다.

<img src="https://1drv.ms/i/s!AvDtmE0jTiDWgi39eNvJzfd14nDi?embed=1&width=778&height=167" style="zoom:50%;" />

이후 w에 대한 편미분을 진행하면, 겉미분으로 인해 곱해진 2에해서 분모의 2는 사라지게 된다. 이후 속미분시 w외의 변수도 전부 상수 취급되어 사라지게 되며, w에 곱해져있던 $x^{(i)}$가 식에 곱해진다.

<br>

$\frac{\partial}{\partial b}J(w,b)$의 경우에는 b에 대한 편미분을 진행하며, b외의 변수는 모두 상수 취급하여 속미분시 없어지게 된다. b에는 곱해진게 아무것도 없었기에 2만이 곱해지게 된다.(마찬가지로 분모의 2와 약분된다.)

![image-20240915004347486](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-1-Gradient%20Descent/image-20240915004347486.png)

이후 정리된 해당 식을 통해서 수렴할 때까지 w와 b를 반복적으로, 동시에 update 해주어야한다.

------

gradient descent의 문제점 중 하나는 global minimum대신 local minimum으로 이어질 수 있다는 것이었다.

**global minimum**은 cost function J의 **모든 점 중 가장 최소의 값을 갖는 점**을 의미한다.

![image-20240915004408253](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-1-Gradient%20Descent/image-20240915004408253.png)

하지만, linear regression의 제곱 오차 비용함수는 local minimum을 여러 개 가지지 않으며, covex function(볼록함수)이기 때문에 **오직 단 하나의 global minimum만을 가진다.**

convex function에 gradient descent를 구현할 때 유용한 특성 중 하나는 **learning rate를 적절하게 선택하기만 하면 항상 global minimum으로 수렴**한다는 것이다.

![image-20240915004426911](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-1-Gradient%20Descent/image-20240915004426911.png)

즉, 적절하게 규칙을 따라서 알고리즘을 진행하면 결국 최솟값에 도달하게 된다.

![image-20240915004438813](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-1-Gradient%20Descent/image-20240915004438813.png)

------

### - Batch gradient descent

우리가 배운 Gradient descent 알고리즘의 다른 이름은 Batch Gradient descent이다.

여기서 **Batch**는 **all the training set**을 의미한다.

즉, 매 단계마다 training set의 일부분이 아니라 **모든 training example을 살펴본다**는 의미이며, 실제로 **계산시에도 모든 training set에 대한 error를 구하기 때문에** 이러한 이름이 붙여졌다.

> 물론, 모든 data set을 사용하지 않고 더 작은 하위 집합(subsets)을 살펴보는 gradient descent도 존재하지만, linear regression에서는 batch gradient descent를 사용한다.

![image-20240915004455062](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-1-Gradient%20Descent/image-20240915004455062.png)