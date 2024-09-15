---
layout: single
title: "Multiple Linear Regression"
categories: ['Machine Learning']
tag: Machine Learning Specialization
typora-root-url: ../
---

> Coursera의 Machine Learning Specialization강의를 정리한 내용입니다.

## Multiple features

우리가 앞서 배운 **_Univariate Linear Regression_**은 single feature x(집의 넓이)가 있었고, 그 x로 y(집의 가격)를 예측했었다.

그러나, 침실수, 층수, 주택 연수와 같은 feature가 늘어난다면 가격을 예측하는데 필요한 더 많은 정보를 얻을 수 있을 것이다.

이처럼 **input feature가 여러 개 있는 유형의 linear regression model**을 **Multiple Linear Regression(다중 선형회귀)**라고 부른다.

![image-20240915010520908](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915010520908.png)

위의 사진은 Multiple features의 예를 든 사진이다.

각 feature를 $X_1, X_2, X_3, X_4$라는 변수를 사용하여 표시하겠다.

> 표기법에 대한 정리는 위를 참고하자. `i`는 **training example**의 index이고, `j`는 **feature**의 index를 의미한다.

즉,  $x_j^{(i)}$는 `i`번째 training example의 `j`번째 feature를 의미한다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915010630665.png" style="zoom:50%;" />

위의 사진은 행 벡터를 나타낸다.

단일 숫자가 아니라 숫자목록을 나타내는 **벡터**임을 시각적으로 보여주기 위해서 위에 화살표를 그려서 표시할 수도 있다.

<br>

---

<br>

Multiple feature의 경우 모델이 어떻게 정의될지에 대해서도 살펴보자. 이전의 single feature일 때는 아래와 같이 model을 정의했다.

<br>

$$Previously:  f_{w,b}(x) = wx+b$$

<br>

하지만, Multiple feature을 이용할 때는 아래와 같이 다르게 정의한다. 

![image-20240915010701077](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915010701077.png)

<img src="/images/2024-09-15-2-Multiple Linear Regression/image-20240915010726613.png" alt="image-20240915010726613" style="zoom:50%;" />

이처럼 각각의 feature들이 개별적인 w에 곱해지는 Hypothesis를 가지게 된다.

이 표현식을 표기법을 이용하여 좀 더 살펴보자

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915010742024.png" style="zoom:67%;" />

우선,  $w_1,w_2,w_3$부터 $w_n$까지의 숫자목록을 의미하는 행 벡터를 정의해보자.

또한, b는 단일 숫자이므로 벡터 w와 b는 모델의 parameter가 된다.

X또한 행벡터로 작성해보자.

![image-20240915010756404](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915010756404.png)

이렇게 하면 model을 아래와 같이 **vector의 dot product(내적)와 숫자 b를 더한 것**으로 간결하게 작성할수 있다.

목록으로 구성된 두 vector의 dot product는 서로 대응하는(index가 같은) 숫자 쌍을 확인하며 계산된다.  

![image-20240915010818638](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915010818638.png)

이처럼 dot product 표기법을 사용하면 더 적은 수의 문자를 사용하여 더 간결하게 모델을 작성할 수 있다.

이처럼 입력 feature가 여러개 있는 linear regression model의 이름은 Multiple Linear Regression(다중 선형회귀)이며, Multiple feature를 갖는 linear regression을 나타낸다.

> **_multivariate regression_**은 다른 용어를 가리키므로 사용하지 않는다.

---

## Vectorization(벡터화)

![image-20240915010834446](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915010834446.png)

`np.dot`은 벡터 w와 x사이의 수학적 dot product를 구현한 다음, b를 더할 수 있다.

즉, `dot` 함수는 두 벡터 사이의 dot product 연산을 vectorization하여 구현한 것이다.

$n$(number of features)이 늘어날 경우, 벡터화를 사용하지 않는 연산보다 훨씬 더 빠르게 실행된다.

![image-20240915010932982](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915010932982.png)

예를들어, for 문을 사용하는 경우 계산을 순차적으로 한단계씩 계산하지만, 벡터화가 적용된 코드는 w와 x의 각 쌍을 동시에 **병렬적**으로 곱할 수 있다.

---

Multiple linear regression에서는 이것이 어떻게 도움이 될까? 

![image-20240915010950118](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915010950118.png)

단순화를 위해 parameter b를 무시하고 계산을 진행해보자.

(learning rate는 0.1로 잡았다)

`d`는 16개의 도함수 값(미분계수)를 계산하여 담을 배열이다.

vectorization에서는 병렬처리 하드웨어를 사용하여 한번에 parameter를 update하고, 효율적인 수행을 진행한다.

---

## Gradient Descent for Multiple Regression

우선 벡터화를 사용하면 Vector 표기법을 사용하여 더 간결하게 작성할 수 있다.

![image-20240915011032641](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011032641.png)

-   Parameters : w를 별도의 여러 개의 파라미터로 생각하지 않고 $\vec {w}$처럼 벡터로 모은다. $\vec {w}$의 길이는 n이된다.
-   model : dot product(내적)을 이용하여 model을 update 했다.
-   Cost Function: 여러개의 w 파라미터로 생각하는 대신 $\vec{w}$ 로 대체했다.
-   Gradient descent : 마찬가지로 J의 파라미터 부분을 벡터로 대체했다.

![image-20240915011102277](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011102277.png)

Gradient descent의 경우에는 1개의 feature만을 가질 때와 n개의 feature을 가질 때 표현식이 달라진다.

먼저, w를 update할 때는 w에 대한 J의 미분 계수로, b를 update할 때는 b에 대한 미분계수로 나타낼 수 있었다.

<br>

error에 관련된 항은 여전히 $f$에서 target value y를 뺀 값을 취하지만, $\vec{w}$와 $\vec{x}$와 같이 vector로 표현하고 오른쪽에 곱해지는 표현식에 $x_1^{(i)}$처럼 아래 첨자가 생겨서 특정 example( $x_1^{(i)}$은 반복시로 가정했을 때 j=1)일 때만 해당되는 update 구문이 만들어졌다.

중요한 점은 n개의 feature가 있을 때도  **w와 j를 동시에 update 해야한다**는 것이다.

---

![image-20240915011133429](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011133429.png)

gradient descent를 사용하는 것 외에도 linear regression에서만 사용되는 w와 b를 찾는 **Normal equation**이라는 방법도 존재한다.

(대부분의 알고리즘에서는 gradient descent를 사용하는 방법이 더 나은 방법이다.자세한 내용은 설명을 읽어보자.)

---

## Feature scailing

feature의 값의 범위가 얼마나 큰지와 해당 feature에 곱해지는 파라미터의 크기 사이의 관계에 대해서 살펴보자.

![image-20240915011149063](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011149063.png)

feature $x_1$은 집의 크기, $x_2$는 침실의 개수라고 두고 price를 예측해보자.

$x_1$ 의 범위는 300~2000, $x_2$ 의 범위는 0~5라고 할 때, 각 feature에 곱해지는 적절한 $w_1$과 $w_2$의 값은 어떻게 선택할수 있을까? 

위의 사진을 보면 **좋은 model은 feature의 범위가 클수록 작은 파라미터 값을 사용하는 것**을 알 수 있다.

<br>

scatter plot(산점도)와 contour plot(등고선도)으로 다시한번 살펴보자

![image-20240915011209796](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011209796.png)

> scatter plot에서 가로축의 배율이 세로축에 비해 훨씬 크고, 값의 범위 또한 넓다는 것을 알 수 있다.  
> 또한, contour plot에서는 가로축에 비해 세로축의 범위가 훨씬 넓기에 찌그러진 타원의 모양을 가진다.

<br>

$w_1$(size)은 범위가 큰 $x_1$을 곱하기에 아주 조금만 변경하여도 price에 큰 영향을 미치는 반면,

$w_2$은 범위가 작은 $x_2$를 곱해서 조금만 변경하면 cost에 거의 영향을 미치지 않는다.(Contour plot에서는 타원의 중심에 다가갈 수록 cost가 최소가 된다.)

![image-20240915011230088](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011230088.png)

즉, **training set을 그대로 사용하면 위와 같이 global minimum에 도달할 때까지 오랜 시간이 걸릴 수 있다.**

<br>

이러한 상황에서 유용한 방법은 **feature scale을 조정하는 것**이다.

즉 training set을 변형하여 feature scale을 0에서 1까지의 범위로 조정한다면,$x_1$과 $x_2$가 비슷한 범위의 값을 취하게 되어 global minimum에 더 빠르게 도달 할수 있게 된다.  

![image-20240915011300985](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011300985.png)

---

### \- Feature scaling

![image-20240915011314540](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011314540.png)

Feature scaling의 기본적인 방법은 **범위의 최댓값으로 해당 feature을 나누는 것**이다. 

또한, $\frac{x^i-min}{max-min}$을 사용하여 각 feature의 최소와 최대 범위를 rescaling 할 수도 있다.

<br>

두 방법 모두 feature를 -1과 1의 범위로 정규화하는데,

전자는 간단하고 강의의 example에 잘 맞는 feature에 사용되고, 후자는 모든 feature에 사용된다.

### \- Mean normalization

![image-20240915011340302](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011340302-1726330422499-1.png)

최댓값으로 나누는 것 외에도, **Mean normalization(평균 정규화)**이라는 작업을 수행할수 있다.

**Mean normalization은 feature 모두가 0을 중심으로 배치되도록 scaling을 하는 과정**이다.

<br>

training set에 있는 $x_1$의 평균을 600이라고 할때, 이것은 아래와 같이 나타낸다.

$$\mu_1=600$$

> $\mu$는 그리스 알파벳으로 [mu,뮤\]라고 발음한다.

Mean normalization으로 feature scaling을 진행하는 방법은 해당 feature의 값에서  $\mu_j$(여기서 j는 feature의 index)의 값을 뺀 뒤, 범위의 (최댓값 - 최솟값)으로 나눠주는 것이다.

### \- Z-score normalization

![image-20240915011359608](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011359608.png)

$ \sigma $(표준편차)를 이용한 Z-score normalization(Z-점수 정규화)방법도 존재한다.

먼저, 평균 $ \mu $를 계산하고, feature에서 $ \mu_j $를 뺀 다음, $\sigma$로 나눠주는 것으로 feature scaling을 진행할 수 있다.

![image-20240915011424301](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011424301.png)

Feature scailing을 진행할 때는 **\-1 ~ 1까지 정도의 범위에서 지정하는 것**이 좋다.

하지만 -3~ 3의 범위 안에 $x_j$가 존재하거나, -0.3 ~ 0.3의 범위안에  $x_j$가 존재하는 등 허용가능한 범위 내에서 다양한 것은 전혀 문제가 되지 않는다. **위의 예시는 Feature의 범위가 너무 크거나 작은 경우 rescaling이 필요함을 보여준다.**

---

## Checking Gradient Descent for Convergence(경사하강법의 수렴 확인)

gradient descent를 실행할 때 수렴 여부에 대해 어떻게 판단할 수 있을까?

주요 단계중 하나는 **learning rate $\alpha$값을 선택하는 것**이다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011450295.png" alt="image-20240915011450295" style="zoom:67%;" />

gradient descent의 역할은 cost function J를 최소화하는 파라미터 w와 b를 찾는 것임을 명심하자.

적절하게 구현한다면, 매 iteration(반복)마다 cost function J의 값은 감소해야한다.

![image-20240915011511238](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011511238.png)

training set을 기반으로 계산되는 J를 각 iteration마다 plotting함으로써 gradient descent가 잘 작동하는지 확인할 수 있다.

**iteration에 따른 cost의 그래프를 Learning Curve라고 한다.**

매 iteration마다 cost function J가 감소하고 있다면, 제대로 작동하는 것이고

한번의 반복 후에 J가 증가한다면, learning rate가 너무 크거나 코드에 버그가 존재할 수 있음을 나타낸다.

<br>

해당 곡선을 보면 400번째 iteration에서는 비용 J가 평준화(수평)되고 있으며, 많이 감소되지 않는 것을 확인할 수 있는데, cost가 크게 감소하지 않을 때, gradient descent가 어느정도 convergence(수렴)한다고 확인할 수 있다.

<br>

각 애플리케이션마다 수렴하는 반복횟수를 미리 파악하는 것은 매우 어려운 일이기에, Learning Curve를 사용할 수 있다.

---

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011530215.png" alt="image-20240915011530215" style="zoom:67%;" />

또한, Automatic convergence test(자동 수렴 테스트)를 사용할 수도 있다.

$\epsilon$ 이라는 아주 작은 수를 임계점으로 두고, cost function J가 $\epsilon$  보다 작다면, convergence하는 것으로 판단하는 것이다. 

<br>

올바른 임계값 $\epsilon$  을 선택하는 것은 매우 어렵기에, Automatic convergence test에 의존하기 보다는 **Learning curve를 통한 직관을 얻는 것이 더 좋다.**

---

## Choosing the Learning Rate(Learning Rate 선택하기)

적절한 Learning rate를 선택하면, 학습 알고리즘이 훨씬 더 잘 실행된다.

**때로는 cost가 올라가고, 때로는 cost가 내려가는 그래프가 나온다면, learing rate가 너무 커서 overshooting이 발생하거나, code에 문제가 생겼음으로 판단할 수 있다.**

또한, **꾸준하게 cost의 값이 증가할 때**도, 같은 경우를 의심해 볼 수 있다.

![image-20240915011558826](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011558826.png)

코드에 문제가 있는지 확인하기 위해서는, 충분히 작은 learning rate $\alpha$를 설정하는 것이다.

잘 구현되었다면 cost function J는 매 iteration마다 항상 감소해야한다.

그러나, 수행시도가 너무 많아지기 때문에 실제 training에서는 너무 작은 learning rate를 사용하진 않는다.

learning rate를 낮추는 것은 debugging을 위해서 자주 사용한다.

![image-20240915011618559](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011618559.png)

 따라서, 수행시에는 다양한 범위의 learning rate를 적용시켜 보는 것이 좋다.

예를들어, 0.001에서 시작하여 1까지 10배씩 곱해보면서, cost를 빠르게 감소시키면서도 일관되게 감소시키는 것으로 보이는 learning rate값을 찾을 수 있다.

해당 과정에서는 Learning rate의 값을 **3씩 곱하거나 나누며 조절하는 것**을 추천한다.

**1\. 가능한 가장 큰 learning rate를 선택하거나, 2. 가장 큰 값보다 약간 작은 값을 선택함**으로써 model에 맞는 좋은 learning rate값을 찾을 수 있다.

---

##  Feature Engineering

 이번에는 알고리즘에 적합한 feature를 간단하게 선택하거나 설계하는 방법을 살펴보겠다.

![image-20240915011640468](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011640468.png)

주택에 두가지 feature가 있다고 가정해보자

-   $x_1$: frontage(너비)
-   $x_2$: depth(깊이)

두가지 feature가 있다면 , 아래와 같은 model을 만들 수 있다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011658395.png" alt="image-20240915011658395" style="zoom: 50%;" />

그러나, 이보다 model에서 feature 선택에 대한 효과적인 방법이 존재한다.

area(면적)은 frontage x depth의 값과 같으므로, area(땅 넓이)를 기준으로 price를 예측하는 것이 너비나 깊이를 개별적으로 선택해서 예측하는 것보다 더 효과적임을 알 수 있다. area를 $x_3$이라는 새로운 feature로 정의함으로써 더욱 정확한 예측을 이끌어 낼 수 있다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011741551.png" alt="image-20240915011741551" style="zoom: 67%;" />

즉, Feature engineering은 learning algorithm이 더 쉽게 정확한 예측을 할 수 있도록 문제에 대한 지식이나 직관을 사용해서 **원래 feature를 변환하거나 결합하여 새로운 feature를 설계하는 과정**이다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011757051.png" alt="image-20240915011757051" style="zoom:67%;" />

---

## Polynomial regression(다항식 회귀)

Feature engineering을 통해서 직선 뿐 아니라 곡선, 비선형함수를 데이터에 맞출 수 있다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011819023.png" alt="image-20240915011819023" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011834057.png" alt="image-20240915011834057" style="zoom:67%;" />

위와 같은 주택 data set이 있다고 가정해보자. 집의 크기에 따른 가격 그래프인데, 직선만으로는 이 데이터 집합을 잘 표현할 수 없다. 

데이터에 곡선을 fitting하거나 2차함수를 데이터에 맞춤으로써 어느정도 해결할 수 있다.

\-> feature를 제곱함으로써 2차함수의 형태를 만들어준다면, 적절하게 fitting될 가능성도 있다.

그러나, 2차함수로 맞출 시 **x가 커질수록 결국은 함수가 감소하기 때문에** 집의 크기가 커지면 가격이 올라가는 것과 같은 예제에서는 2차함수의 모형은 별로 의미가 없을수도 있다.   

<br>

이러한 경우에서는 $x^2$뿐 아니라 $x^3$이 있는 3차함수를 선택할 수도 있다. 3차함수는 크기가 결국 다시 커지기 때문에 데이터에 더 잘맞는 곡선을 fitting할 수 있다.

![image-20240915011854821](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011854821.png)

**단,** 해당 예시처럼 원래 feature의 제곱처럼 **feature를 거듭제곱하여 새로운 feature를 만들게 되면, feature scaling이 매우 중요해지게된다.** 원래 feature의 범위에 제곱을 취하기 때문에 새롭게 설계된 feature는 **원래 feature과 매우 다른 값의 범위**를 취하게 된다. 

따라서 feature scaling을 적용하여 비교가능한 범위 내로 바꾸어주어야한다.

![image-20240915011912807](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-2-Multiple%20Linear%20Regression/image-20240915011912807.png)

또 다른 방법으로 x의 제곱근을 사용하는 방법도 있다.

$\sqrt{x}$의 그래프는 덜 가파르지만 x의 증가에 따라 꾸준하게 증가하고 절대내려오지 않기 때문에 해당 data set에서도 잘 작동할 수 있다.

---

Course2에서 다양한 feature를 선택하고 포함시키는 방법을 배우기에 이 강의에서는 **어떤 feature를 사용할지 우리가 선택할 수 있다**는 점을 알아두는 것이 중요하다.