---
layout: single
title: "Logistic regression 1"
categories: ['Machine Learning']
tag: Machine Learning Specialization
typora-root-url: ../
---



> Coursera의 Machine Learning Specialization강의를 정리한 내용입니다.

## **Motivations**

이번글과 다음글에서는 target value y가 몇가지 가능한 값 중 하나만 가질 수 있는 classification(분류) 문제에 대해서 알아볼 것이다.

![image-20240915015015116](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015015116.png)

위의 사진은 classification에 대한 몇가지 예이다.

email이 스팸인지 아닌지, 온라인 금융거래가 사기인지 아닌지, 종양이 악성인지 아닌지 등을 구분하는 예시들이다.

<br>

이때, 예측하려는 variable은 가능한 두 값 중 하나(yes 또는 no)일 것이며,

**가능한 output이 두 개 뿐**인 이러한 유형의 **classification 문제**를 **binary classification(이진 분류)**이라고 한다.

> 'binary'라는단어는 가능한 class 또는 category가 두개 뿐인것을 가리키는 말이다.  
> classification 문제에서는 class와 category를 비교적 같은 의미로 사용한다.

---

우리는 종종 yes나 no, false나 true, 0과 1로 거짓이나 참을 표시한다.

(학습 알고리즘 유형에 가장 잘 맞기에 보통 숫자를 사용하여 답 y를 많이 표현한다.)

![image-20240915015028771](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015028771.png)

우리는 **false인 값**을 **negative class**, **true인 값**을 **positive class**라고 지칭할 수 있다.

스팸 메일과 관련된 예시를 들어보자면 아래와 같다.

> Is this email spam?  
> negative class: No(스팸이 아님)  
> positive class : Yes(스팸임)

단, 이 용어가 좋음과 나쁨의 개념을 나타내는 것은 아니고,

absence(부재) 또는 false의 개념 / presence(존재) 또는 True의 개념과 대응되는 용어를 사용한 것일 뿐이다.

질문에 대한 **참의 답**을 **positive class**, **거짓의 답**을 **negative class**로 표시한 것이라고 생각하면 된다.

물론, 어느 쪽을 True라고 할지, 어느쪽을 False라고 할지는 상황에 따라 달라질 수 있다.

---

### Classification 알고리즘은 어떻게 구축할까?

![image-20240915015051347](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015051347.png)

위의 사진은 종양이 악성인지, 양성인지 분류하기 위한 training set의 example이다.

week1에서 classification에 대해서 알아볼 땐 아래의 모양처럼 하나의 직선에 plotting했지만, 

이번엔 세로축을 이용해서도 plotting했다.

해당 example들을 classification하기 위해 linear regression을 시도해보자.

![image-20240915015107646](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015107646.png)

기존에 배운 Hypothesis를 통해 $f_{w,b}(x)=wx+b$라는 식을 세울 수 있다.

Classification 문제를 해결하기 위해 linear regression을 이용해서 시도해 볼 수 있는 것은 **threshold**(임계값)을 설정하는 것이다.

예를 들어, 임계값을 $0.5$로 설정한다고 하면, model의 output이 0.5미만이면 $0$으로 악성이 아니고 , $0.5$이상이면 $1$로 악성과 같다고 판단할수 있다. 즉, 임계점이 있는 곳에 수직선(구분선)을 그리면 **왼쪽에 있는 모든 example의 예측값이 $0$이되고, 오른쪽에 있는 모든 example의 예측값은 $1$**이된다. 

분류가 잘 되는 것 같아 보이지만, 오른쪽에 하나의 example을 추가해보자.

![image-20240915015133182](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015133182.png)

하나의 training example이 추가됨에 따라 Hypothesis는 오른쪽으로 이동하게 되고, 수직으로 그은 구분선 또한 오른쪽으로 이동한다. (해당 구분선은 **decision boundary**라고 불린다.)

해당 예제에서는 threshold를 0.5로 잡은 경우에 training example이 추가됨에 따라 분류가 제대로 되지 않음을 확인할 수 있으며, linear regression으로는 classification problem을 해결하는데 실패하는 경우가 많음을 알 수 있다.

<br>

이를 해결하기 위해, classification에는 **logistic regression(로지스틱 회귀)**를 이용한다.

>  logistic regression에 regression이라는 단어가 붙어있지만 Classification 문제에 사용됨에 주의해라.

---

## Logistic regression

위에서 확인했듯이, linear regression은 Classification 문제에 적합한 알고리즘이 아니다.

![image-20240915015146740](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015146740.png)

세로축은 $0$또는 $1$만 사용하는 binary classification의 예이다. 

logistic regression에서는 data set에 위와 같은 s자형 곡선을 맞춘다.

해당 예제의 경우 보라색 선과 같은 x값을 가진 환자가 들어오면 알고리즘은 0.7을 출력하는데, 이는 악성에 더 가까울 가능성이 높다는 것을 의미한다.

**단, threshold를 거쳐 실제로 결정되는 output label y는 0.7이 아니라 0 또는 1이어야한다. 여기서 출력된 값은 $\hat y$이다.**

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015200861.png" alt="image-20240915015200861" style="zoom:67%;" />

logistic regression algorithm에 적용하기 위해 **Sigmoid** **function**(시그모이드 함수)라는 중요한 함수가 존재한다.

sigmoid function은 **logistic** **function**이라고도 한다.

Sigmoid Function의 특징은 아래와 같다.

-   가로축에는 음수와 양수값이 모두 표시되고, 가로축에는 $z$라는 label이 붙어있다.  
    (여기서는 -3 ~ 3까지의 범위만 보여준다. 원래는 실수 전체다)
-   0과 1사이의 output값을 가진다.

보통 g(z)를 사용하여 이 함수를 나타내는데, 식은 아래와 같다.

 $$g(z)=\frac{1}{1+e^{-z}}$$

여기서 e는 약 2.7의 값을 갖는 수학 상수이다.

$z$의 값이 매우 클수록 분모가 1에 가깝게 되어 output이 1에 수렴하며,

$z$의 값이 매우 작을수록 분모가 0에 가깝게 되어($\frac{1}{\infty}$) output이 $0$으로 수렴하게 된다.

$z$값이 $0$일때는 $0.5$의 값을 가진다.

---

이제 이 함수를 이용해 Logisitc regression 알고리즘을 구축해보자.

![image-20240915015230445](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015230445.png)

linear regression function과 같은 직선함수는 w와 x의 dot product와 b의 합으로 정의할 수 있었다.

이 값을 z라고 부르는 변수에 저장해보자.(이 값은 sigmoid function z와 동일한 z가 된다.)

<br>

이후, 이 z를 sigmoid 함수에 대입하면, $g(z)$는 해당 공식으로 계산된 값을 출력하게 된다.

$$g(z)=\frac{1}{1+e^{-(\vec{w} \cdot \vec{x}+b)}}$$

또한, logistic regression의 hypothesis로 g(z)를 사용하므로 $ f(x) = g(z)$ 라고도 나타낼 수 있다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015251370.png" alt="image-20240915015251370" style="zoom:50%;" />

이 model이 logistic regression model이다.

![image-20240915015330282](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015330282.png)

logistic regression의 Hypothesis의 출력값은 **특정 input x에 대해** **y(class)가 1이 될 확률**을 의미한다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015341643.png" alt="image-20240915015341643" style="zoom: 50%;" />

예를 들어 악성 종양인지를 구분하는 예제에서 output이 0.7이라면, 악성으로 판명될 확률(y=1일 확률)이 70%라고 생각할 수 있다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015400691.png" alt="image-20240915015400691" style="zoom:50%;" />

y의 output은 0또는 1이되어야만 하기에, y=0일 확률과 y=1일 확률의 합은 1이 된다. 따라서, y=0일 확률은 0.3임도 알 수 있다.

 logistic regression의 표기법은 아래와 같이 사용한다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015415897.png" alt="image-20240915015415897" style="zoom:50%;" />

세미콜론 `;`을 통해 input feature x에서 y가 1이 될 확률에 영향을 미치는 파라미터를 구분할 수 있다.  

---

## **Decision boundary**

![image-20240915015442709](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015442709.png)

Logisitc regression의 Hypothesis의 출력값이 y가 1이될 확률을 나타내므로, 0.7 또는 0.3과 같은 값을 가지게 될 것이다.

이 함수를 예측(predict) 알고리즘에 사용할 때, y의 값은 1이 될까 0이 될까?

<br>

우리가 사용할 수 있는 방법은 **threshold**(임계값)을 설정하는 것이다.

일반적인 선택은 threshold를 **0.5**로 선택해서 f(x)(즉, 출력값)가 0.5보다 크거나 같으면 $\hat y=1$이 되도록 하고, 그렇지 않다면 $\hat y=0$이 되도록 할 수 있다.

<br>

그렇다, f(x)가 $0.5$ 이상일 때는 언제일까?

![image-20240915015453447](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015453447.png)

Logistic regression의 hypothesis는 Sigmoid function을 사용하므로 $f(x) = g(z)$로 나타낼 수 있다.

따라서 $g(z) \geq 0.5$로 표현할 수 있으며, 이는 **z값이 $0$ 이상일 때**와 같다.

z는 위와 같이 $\vec{w} \cdot \vec{x}+b$로 나타낼 수 있으므로, $\vec{w} \cdot \vec{x}+b\geq0$일 때 model은 1을 예측한다.

<br>

두개의 feature가 있는 classification 문제의 예를들어보자.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015525245.png" alt="image-20240915015525245" style="zoom: 67%;" />

X가 positive class, O가 negative class를나타내는 training set이다.

즉, X는 y=1에, O은 y=0에 해당한다. 

![image-20240915015709177](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015709177.png)

위에서 배웠듯이, logistic regression에서는 $g(z)$를 사용해서 예측을 수행한다.

feature가 2개이므로 위와 같이 표현식을 잡을 수 있고,

$w_1$은 1, $w_2$는 1, $b$는 -3이라고 가정해보자.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015545780.png" alt="image-20240915015545780" style="zoom:67%;" />

**$\vec{w} \cdot \vec{x}+b=0$이 되는 경우** y=1인지 y=0인지에 대해 중립적인 위치(즉, 확률이 0.5인경우 ; z=0인 경우 sigmoid function의 output은 0.5)를 가지게 되며, 이 경우들을 모아서 나타낸 선을 _**Decision boundary**_라고 부른다.

$x_1+x_2=3$을 만족하는 경우를 나타낸 보라색 선이 Decision boundary이다.

<br>

이 선의 왼쪽에 있으면 logisitc regression은 0을 예측하고, 오른쪽에 있으면 1을 예측하게된다.

**(즉, z가 0보다 크거나 작은 경우를 고려해서 확률로 출력된 output의 $\hat y$(0또는 1)을 결정한다.)**

> 물론 파라미터를 다른값으로 선택한다면 decision boundary는 다른 선이 될 것이다.

### \- Non-linear decision boundaries

![image-20240915015641127](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015641127.png)

logisitc regression에서도 polynomials(다항식)을 사용할 수 있다.

decision boundary를 결정짓는 원리(z=0인 지점)는 같으며, 더욱 복잡한 decision boundary를 만들 수 있다.

![image-20240915015627372](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-3-Logistic%20regression%201/image-20240915015627372.png)

또한, feature의 dimension(차수)을 더 늘리면 훨씬 더 복잡한 decision boundary를 만들 수 있다. 

하지만, 이러한 고차 다항식을 하나도 포함하지 않고 **차수가 1인 feature만 사용할 경우 decision boundary는 항상 선형이고 직선이 된다.**

---

## **요약**

-   linear regression은 classification 문제엔 적합하지 않아 logisitc regression을 사용한다.
-   logistic regression은 sigmoid 함수를 사용하며, 출력값으로는 y=1이될 확률을 가진다.
-   classification에서는 decision boundary를 통해서 threshold 이상이면 이 확률(출력값)을 1로 취급하고, 임계값 미만이면 0으로 취급한다.