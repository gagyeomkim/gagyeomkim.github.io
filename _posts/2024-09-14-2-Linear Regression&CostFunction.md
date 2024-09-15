---
layout: single
title: "Linear Regression & Cost Function"
categories: ['Machine Learning']
tag: Machine Learning Specialization
typora-root-url: ../
---

> Coursera의 Machine Learning Specialization강의를 정리한 내용입니다.

## Linear Regression Model(선형 회귀 모델)

![image-20240914185954408](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914185931200.png){: .align-center}

위 그래프는 주택 size에 따른 주택 price의 그래프를 나타낸 것이다.

한 사람이 1250평방피트 크기의 집을 판매할 때, 어느정도의 가격을 받을 수 있는지의 예에 대해서 살펴보자

우리는 해당 dataset에 적합한 직선을 그릴 수 있으며, 판매시 가격이 약 22만 달러라는 것을 예측할 수 있다.

data set에는 모든 주택에 대한 price(정답)가 나와있다.

바로 이것이 지난시간에 배운 Supervised Learning(지도학습)이라고 불리는 것의 한 예이다.

먼저 **정답(right answer)이 있는 데이터를 제공하여 모델을 학습**시키기 때문이다.

Output으로 무한히 가능한 많은 숫자를 예측하기에, 위와 같은 모델을 Regression model이라고한다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914190028062.png" style="zoom:67%;" />{: .align-center}

또한, 위와 같이 도표로 시각화하는 것외에도 표를 통해서도 시각화할 수 있다.

***가로축***과 **`세로축`은 *size***와 **`price`**에 해당하는 두개의 열과 대응된다.

또한, 왼쪽 그림에는 X표시가 47개 있으며, 각 X는 **표의 한 행(즉, 전체 행은 47개)에 해당**한다.

------

### - Notation(표기법)

데이터를 설명하기 위한 몇가지 표기법에 대해서 살펴보자.

**Training set : 모델 학습에 사용되는 dataset**(고객의 집은 아직 판매되지 않았으므로 data set에 포함되지않는다.)

<br>

출력값을 예측하기 위해서는 먼저 Training set을 통해서 학습하도록 모델을 훈련시켜야한다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914190056176.png" style="zoom:50%;" />{: .align-center}

- $x$= "**input"** variable(입력변수) or "**feature"**(특징)
- $y$ = "**output"** variable(출력 변수) or "**target"** variable(목표 변수)
- $m$ = number of training examples(training example의 총 개수)
- $(x,y)$= single training example(training example 1개를 나타내기 위해 사용)
- $(x^{(i)},y^{(i)})$ = $i^{th}$ training example( $1^{st}, 2^{nd}, 3^{rd}$) 여기서 $(i)$는 지수(exponent)가 아니라, 행을 나타내는 인덱스(index)일 뿐이며 테이블의 i행을 참조한다.

------

이제 Training set을 이용하여 Supervised Learning의 작동방식을 살펴보자.

<img src="/images/2024-09-14-2-Linear Regression&CostFunction/image-20240915153930618.png" alt="image-20240915153930618" style="zoom:50%;"/>{: .align-center}

먼저, 입력(features)과 출력(targets)를 모두 포함하는 Training set을 학습 알고리즘에 공급한다.

그러면 Supervised learning 알고리즘은 함수를 생성해 내는데, 이 함수는 소문자 $f$로 작성한다.

여기서 $f$는 함수를 나타낸다. 이 함수는 Hypothesis(가설함수)이라고도 불리며, 많은 테스트 이후 생성된 f함수는 model이라고 한다.

> Q. Is a model the same as a hypothesis?(model과 Hypothesis의 차이)
>
> <br>
>
> **A hypothesis is just an idea that explains something.** It must go through a number of experiments designed to prove or disprove it.  
> **Model**: A hypothesis becomes a model **after some testing has been done and it appears to be a valid observation.**

<br>

**Hypothesis를 사용한 작업은 새로운 입력값 $x$를 가져와서 $\hat{y}$ (예측값)을 출력하는 것이다.**

머신러닝에서 $\hat{y}$ 은 y에 대한 추정값 또는 예측값으로, 실제 참값일 수도, 아닐 수도 있다.

>  기호문자가 $y$일 경우에는 훈련 세트의 실제 참 값인 target variable을 나타낸다.

<br>

그래서 함수 f(=Hypothesis)는 어떻게 표시할까?

우리는 f를 아래와 같이 표현할 것이다.

<br>

$$f_{w,b}(x) = wx+b$$

<br>

w와 b는 숫자이고, w,b에 대해 선택한 값이 input인 feature $x$를 기반으로 $\hat{y}$ 을 결정한다.

<br>

물론, 아래 첨자로 w와 b를 명시적으로 포함시키지 않고 아래와 같이 f(x)만 쓰기도 한다.

<br>

$$f(x) = wx+b$$

<br>

Linear function(선형함수)은 비교적 단순하고 다루기 쉬우므로, 기초적 model을 구축할 땐 Linear function을 자주 선택한다.

해당 model은 **변수가 하나(one variable)인 linear regression이다**.

-> 변수가 1개라는 말은 single feature x(단일 입력 특징)을 가진다는 것이다.

입력 변수가 하나인 선형 모델은 ***Univariate linear regression***으로 표기하기도 한다.

***uni***는 라틴어로 **1**을, ***variate***는 **변수**를 의미한다.

------

### - Python에서의 일반적인 표기법

파이썬에서는 아래와 같은 표기법을 사용하는 것을 추천한다.

| **General Notation** | **Description**                                              | **Python**   |
| -------------------- | ------------------------------------------------------------ | ------------ |
| $a$                  | scalar, non bold, 소문자                                     |              |
| $\mathbf{a}$         | vector, bold, 소문자                                         |              |
| **Regression**       |                                                              |              |
| $\mathbf{x}$         | Training Example **feature** values (in this lab - Size (1000 sqft)) | `x_train`    |
| $\mathbf{y}$         | Training Example **targets** (in this lab Price (1000s of dollars)) | `y_train`    |
| $x^{(i)}$, $y^{(i)}$ | $i_{th}$ Training Example                                    | `x_i`, `y_i` |
| m                    | Number of training examples                                  | `m`          |
| $w$                  | parameter: weight                                            | `w`          |
| $b$                  | parameter: bias                                              | `b`          |
| $f_{w,b}(x^{(i)})$   | The result of the model evaluation at $x^{(i)}$   parameterized by $w,b$: $f_{w,b}(x^{(i)}) = wx^{(i)}+b$ | `f_wb`       |

<br>

`.shape` : 각 dimension에 대한 정보가 있는 파이썬 튜플을 리턴한다.

`.shape[0]`: 배열의 길이(training example의 개수)

`plt.scatter(x_train, y_train, marker='x', c='r')` : `marker`와 `c`는 X로 data를 보여주기 위한 시각화 옵션.

------

## Cost Function(비용함수)

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914190056176.png" style="zoom: 50%;" />{: .align-center}

위 사진은 이전에 보았던 features(입력) x와 targets(출력) y를 포함하는 훈련세트이다.

해당 훈련세트에 사용할 model은 아래와 같다.

<br>

$$f_{w,b}(x) = wx+b$$

<br>

이때, $w$와 $b$를 모델의 **parameter**(매개변수)라고 한다.

기계학습에서 모델의 **parameter**는 **모델을 개선하기 위해 훈련중에 조정할 수 있는 변수**를 말하며,

파라미터 $w$와 $b$는 **계수**(**coefficients**) 또는 **가중치**(**weights**)라고 부르기도 한다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914190434698.png" style="zoom: 50%;" />{: .align-center}

함수 f에서 얻은 직선은 위와 같이 파라미터 $w$와 $b$의 값에 따라 다른 형태를 가지게 된다.

즉, linear regression의 경우 함수 f에서 얻은 직선이 데이터에 잘 맞도록 아래처럼 매개변수 $w$와 $b$를 조정할 필요가 있다.

<br>

- $w$: 선의 기울기(slope)
- $b$: y절편(y-intercept)

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914190500435.png" style="zoom:50%;" />{: .align-center}

>  $y^{(i)}$은 target을 뜻한다.

<br>

주어진 입력값 $x^{(i)}$에 대해서 함수 f는 y에 대한 예측값인 $\hat{y}$를 만들게 된다.

<br>

$$\hat{y} = f_{w,b}(x^{(i)})$$

<br>

또한, $f_{w,b}(x^{(i)})$는 아래와 같이 정의된다.

<br>

$$f_{w,b}(x^{(i)}) = wx^{(i)}+b$$

$$ \hat{y} = f_{w,b}(x^{(i)}) = wx^{(i)}+b$$

<br>

우리가 찾아야하는 것은 예측값($\hat y^{(i)}$)이 실제 목표($y^{(i)}$)에 가깝도록 $w$와 $b$의 값을 찾는 방법이다.

-> 이 방법이 cost function을 이용하는 것이다.

<br>

cost function은 $\hat{y}$에서 $y$ (target) 을 뺀 값을 이용하는데, 이 차이를 오차(error)라고 하며 error는 예측값이 목표값과 얼마나 멀리 떨어져 있는지를 측정한다.

<br>

$$ error = \hat{y}^{(i)} - y^{(i)}$$

<br>

우리는 **오차가 작을 때,** 선택한 파라미터가 데이터와 잘 일치하는 것으로 판단할 수 있다.

단, 오차는 음수로도 표현될 수 있기 때문에 **오차의 제곱이 최소**가 되도록 하여 찾는 것이 합리적이다.

<br>

$$ minimize\, \left( \hat{y}^{(i)} - y^{(i)} \right)^2 $$

<br>

또한, 실제 Training set은 1부터 m까지 존재하기 때문에 1부터 m까지의 example의 error를 모두 합해서 최소가 되는 cost를 구해야한다. 공식으로 나타내면 아래와 같다.(계산시 편의를 위해 2로 나눠준다)

> m은 Training example의 개수이다.

<br>

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

<br>

f의 출력값과 $\hat y^{(i)}$은 동일하므로 이를 아래처럼 다시 작성할 수 있다.

<br>

$$ J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left(f_{w,b}(x^{(i)}) - y^{(i)} \right)^2 $$

<br>

위에서 구한 표현식 J가 cost function이며, 오차항의 제곱과 평균을 취하기 때문에 **Mean squared error** (평균제곱오차) function이라고도 부른다.

------

### - Cost Function Intuition(직관)

비용함수 $J$를 최대한 작게 만드는 파라미터 $w$,$b$를 찾는 것이 선형회귀의 목표이다.

수학에서는 아래와 같이 쓴다.

<br>

$$  minimize\,J(w,b) $$

<br>

![image-20240914190725108](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914190725108.png){: .align-center}

cost function J를 잘 시각화 하기위해 오른쪽 사진과 같이 단순화된 버전을 이용해보자.

파라미터 w만 쓰는 model을 만들기 위해서는 파라미터 b를 0으로 설정하는 것으로(또는 제거하는 것으로) 생각할 수 있으며, b가 0으로 설정되면, **파라미터 w 하나만이 J에 영향을 미치게 되므로** 목표는 아래와 같이 바뀐다.

<br>

$$  minimize\,J(w) $$

<br>

이제 몇가지 예제를 확인해서 cost function이 무슨일을 하고, 왜 사용해야하는지 확인해보자

![image-20240914190758319](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914190758319.png){: .align-center}

파라미터 **w가 정해진경우, y의 추정값은 입력값 x에 따라서만 변화**하게 된다.

오른쪽은 w에 따른 J의 값을 나타낸 사진이다.

이 예시에서 w가 1이라고 가정하면, cost function J는 0이 되며, 이는 오른쪽 그래프에 하나의 점으로 나타낼 수 있다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914190827711.png" style="zoom:50%;" />{: .align-center}

w가 0.5일때를 계산해보면 cost function이 0.58 정도의 값을 가지며, 오른쪽과 같이 하나의 점으로 나타낼 수 있다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914190845835.png" style="zoom: 50%;" />{: .align-center}

이처럼 w의 각 값에 대해서 대응되는 $J(w)$가 생성되며, 이 점들을 이용해서 오른쪽과 같은 그래프를 그릴 수 있다.

위에서 살펴봤듯이, 우리의 목표는 $J(w)$를 최소화하는 것이며, 제곱 오차를 최소화하는 w를 선택하면 좋은 model을 얻어낼 수 있다.

------

### - Visualizing(시각화)

위의 예시에서는 단순화를 위해 b를 0으로 설정했었기에, 아래와 같은 그래프의 cost가 나왔었다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914190915270.png" style="zoom:50%;" />{: .align-center}

이제 파라미터가 w,b로 두개가 되었고, 이전보다 Cost function J의 그래프는 더욱 복잡해 질 것이다. 마찬가지로 포물선의 형태를 띄고 있을 테지만 아래와 같이 3차원 그래프로 나타낼 수 있을 것이다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914190947419.png" style="zoom: 50%;" />{: .align-center}

높이에 따라 cost function이 나타나므로, **그래프 표면의 어떤 한 점이 특정한 값(cost)**를 가지게 된다.

![image-20240914191010702](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914191010702.png){: .align-center}

이런 cost function을 더 편리하게 도표화하기 위해서, 3차원 표면도를 사용하는 대신 등고선 그래프(Contour plot)를 이용하는 방법이 있다.

![image-20240914191025980](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914191025980.png){: .align-center}

Contour plot을 만드는 방법은 3d 표면을 수평으로 자르는 것이며, 자른 조각들은 Contour plot의 타원 중 하나로 표시된다.

위 오른쪽 그래프가 Contour plot이며, 여기서 **Cost Function $J(w,b)$의 최솟값은 타원의 중심 좌표**이다.

또한, **같은 선 위에 있는 점들은 w와 b의 값이 다르더라도 동일한 cost값을 가진다.**

<br>

시각화 예제를 조금 더 살펴보자

![image-20240914191042728](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914191042728.png){: .align-center}

$b=800, w= -0.15$로 잡으면, 오른쪽 그래프에 하나의 점으로 표시할 수 있다. dataset을 살펴보면, training data에 있는 실제 **target**값과는 상당히 다르므로, Contour plot에서 해당지점의 cost가 최솟값(타원 중심)과는 거리가 먼 것을 확인할 수 있다.

>  데이터에 잘 맞지 않는 것은 실제로 cost의 최솟값에서 더 멀리 떨어져있다는 것을 나타낸다.

<br>

즉, 위의 Hypothesis가 좋은 Hypothesis가 아니라는 것을 알 수 있다.

![image-20240914191113725](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-14-2-Linear%20Regression%26CostFunction/image-20240914191113725.png){: .align-center}

위 그래프는 cost의 최솟값 지점과 인접하므로 적합한 Hypothesis라고 예상할 수 있고, Training set과 비슷한 형태인 것을 확인할 수 있다.

<br>

### - 주의해야할점!

**Training example들이 모두 선위에 있지는 않으므로, cost의 최솟값은 0이 될 수 없다.**

---

## Reference:

* [https://junstar92.tistory.com/14](https://junstar92.tistory.com/14){: .align-center}

