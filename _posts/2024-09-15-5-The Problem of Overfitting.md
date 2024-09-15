---
layout: single
title: "The Problem of Overfitting"
categories: ['Machine Learning']
tag: Machine Learning Specialization
typora-root-url: ../
---

> Coursera의 Machine Learning Specialization강의를 정리한 내용입니다

<br>

우리가 이제까지 배운 linear regression과 logistic regression 알고리즘은 많은 머신러닝 문제들에 적용시킬 수 있다.

하지만 Overfitting(과적합)이라는 문제에 빠져 알고리즘의 성능이 나빠질 수 있다.

이번 게시글에선 Overfitting에 대해 알아보고 underfit이라는 정반대의 문제에 대한 것과, Overfitting의 해결방법인 Regularization(정규화)에 대해서 알아보자.

## Overfitting

Overfitting이란 무엇일까? linear regression에서 사용했던 주택 가격 예측 예제로 돌아가보자.

 input feature x는 주택 size이고 y는 주택 price의 예측값이다.

![image-20240915180244240](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180244240.png){: .align-center}

---

### - underfitting(언더피팅, 과소적합)

data set들을 fitting하는 한가지 방법은 왼쪽처럼 직선으로 fitting하는 것이다. 

하지만 선형적인 fitting은 규모가 커질수록 training data에 잘 맞지 않는다. **training set에 잘 fitting 되지 않는 경우**를 나타내기 위해 **underfitting(과소적합)**이라는 용어를 사용하며, 또다른 용어로는 **high bias(편향이 높다)**라고 나타낸다. (두 용어는 거의 같은 의미로 사용된다.) 

> 보통 "모델이 training data를 underfitting하고 있다"는 말로 쓰이며, 이는 알고리즘이 data를 제대로 맞추지 못한다는 것을 나타낸다.

<br>

이때 bias란 어떤 의미로 쓰일까? 알고리즘이 training data를 잘 fitting하지 못하게 되면, **알고리즘이 포착할 수 없는 pattern이 존재**한다고 판단할 수 있으며, 이러한 형태는 **알고리즘이 매우 강한 선입견, 즉 매우 강한 bias**를 가지고 있다는 것과 같다. 즉 이러한 bias때문에 데이터에 잘 맞지 않는 직선을 이루게 된다.

### - Generalization(일반화)

중간의 그래프를 살펴보자. 2차함수 모양의 feature를 가져감으로써 dataset에 적합한 곡선을 그리고 있으며, 새로운 집을 구매한다고 할때, 가격을 잘 맞게 예측할 수 있을 것 같다.

이때, **학습 알고리즘이 training set에 없는 example에서도 잘 작동하기를 바라는 생각**을 **generalization**이라고 한다.

엄밀히 말하면, 우리는 학습 알고리즘이 generalization되기를 원한다.

> 중간의 그래프는 just right라고 표현되었는데,  
> 이는 기술적으로 정의된 공식 용어가 아니며 그냥 적절하게 잘 맞는 경우를 표현하기 위해서 강의에서 이렇게 적었다.

### - overfitting(오버피팅, 과적합)

오른쪽 그래프는 모든 training data를 완벽하게 통과하기 때문에 cost가 0이 된다. 하지만, 이것은 매우 구불구불한 모양의 곡선 형태를 띄게 되며, 분홍색 지점과 같이 size에 따른 price의 가격이 새로운 example에 잘 들어맞지 않는 경우가 발생할 수 있다.(집의 크기가 큰데, 가격이 더 싼 현상이 일어났다)

<br>

기술적인 용어로는 해당 model에 overfitting 문제가 있다고 말할 수 있으며, **학습 알고리즘이 training set에 너무 잘 맞는 경우**를 **overfitting**이라고 한다. 또 다른 용어로는 **high variance**(고분산)이라고도 표현한다.(보통 두 용어는 거의 같은 의미로 사용된다.)

<br>

high variance는 무슨 뜻일까?

training set에 너무 잘 맞기 때문에, training set에 없는 새로운 example이 나타나면 알고리즘이 fitting하는 함수도 완전히 달라지게 된다. 즉, **약간만 다른 dataset에 fitting한 경우에도 완전히 다른 예측이나 가변적인 예측이 나올 수 있기 때문에 알고리즘의 variance(분산)이 크다**고 말하는 것이다.

<br>

최종적으로 **machine learning의 목표는 underfitting도, overfitting도 없는 model을 찾는 것**이라고 할 수 있다.

![image-20240915180306695](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180306695.png){: .align-center}

Classification에도 overfitting이 적용되기도 한다.

underfitting(=high bias)의 경우와 generalization의 경우, overfitting(high variance)된 경우를 보여준다.

해당 사진에서도 고차 polynomial(다항식) feature들을 모두 사용하게 되면 overfitting이 발생할 수 있음을 확인할 수 있다.

---

## Addressing Overfitting(Overfitting 해결법)

Overfitting은 어떻게 해결할 수 있을까?

한가지 방법은 더 많은 training example들을 수집하는 것이다.

<br>

training set이 클수록 학습 알고리즘은 덜 흔들거리는 함수에 적합하도록 학습할 것이다.

많은 고차 다항식을 가진 function도 training example이 충분한 경우에는 괜찮기도 하다.

하지만, 더 많은 데이터를 확보하는 것이 항상 가능한 것은 아니다.

<br>

overfitting은 많은 feature를 사용하고 데이터가 불충분한 상황에 발생할 수 있다.

overfitting 문제를 해결하기 위한 두번째 방법은 사용할 수 있는 feature의 수를 줄이는 것이다.

즉, **너무 많은 polynomial feature를 사용하지 않고 유용한 feature만 고르는 것**이다.

![image-20240915180325929](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180325929.png){: .align-center}

관련성이 높은 feature들만 선택한다면 model이 더이상 overfitting하지 않을 수 있으며,

**사용하기에 가장 적합한 feature set을 선택하는 것**을 **feature selection**이라고 부른다.

feature selection의 한가지 단점은 일부 feature만 사용함으로써 유용한 feature가 버려질 수도 있다는 것이다.

> Course2에서는 예측작업에 사용할 가장 적합한 training set을 자동으로 선택하는 알고리즘에 대해서도 설명한다.

<br>

Overfitting을 줄이기 위한 3번째 방법이 바로 Regularization(정규화)이다.

![image-20240915180348034](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180348034.png){: .align-center}

위에서 알아본 feature selection은 일부 feature를 제거함으로써(또는 feature의 값을 0으로 설정함으로써) overfitting을 해결했다. Regularization은 일부 feature를 제거하는 극단적인 조치를 취하지 않고도 일부 feature의 영향을 좀 더 완만하게 줄일 수 있다.

즉, **Regularization은 파라미터를 정확히 0으로 설정하도록 요구하지 않고도 학습 알고리즘이 파라미터의 값을 줄이도록** 하는 방법이다.

<br>

이를 통해 고차 polynomial을 사용하더라도 작은 파라미터 값을 사용하여서, feature가 지나치게 큰 영향을 주어 overfitting이 일어나는 것을 방지한다.

Regularization을 통해 결국 training set에 더 잘 맞는 곡선이 만들어진다.

<br>

**일반적으로 파라미터 $w(w_1...w_n)$까지의 크기만을 줄이고 파라미터 b는 regularization하지 않는다.**

파라미터 b는 regularization하든 안하든 거의 차이가 없기 때문이다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180401176.png" alt="image-20240915180401176" style="zoom: 50%;" />

즉 overfitting을 해결하는 세가지 방법은 아래와 같다.

<br>

1.  더 많은 data를 수집한다.
2.  feature selection을 통해 feature를 선택한다.
3.  Regularization을 이용하여 파라미터의 크기를 줄인다.

---

## Cost Function with Regularization

고차 polynomial을 fitting하면 아래와 같이 overfitting된 곡선이 생길 확률이 높다는 것을 위에서 알아봤다.

![image-20240915180418557](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180418557.png){: .align-center}

만약 파라미터 $w_3$와 $w_4$를 매우 작게 만들 방법이 있다고 가정해보자.

비용함수를 수정해서 위와 같이 바꿔보겠다.

수정된 비용함수를 사용하면 실제로 $w_3$와 $w_4$의 크기가 클 경우 model을 최소화하는데 불이익이 될 수 있다.

cost function을 최소화하려는 경우 유일한 방법은 $w_3$와 $w_4$를 모두 작게 만드는 것이며, 최소화하면 두 파라미터가 0에 가까워지게 되는데, 이것은 hypothesis에도 영향을 미친다.

즉, **파라미터를 최소화시킴으로써 특정 feature가 과도한 영향을 미치는 것을 막으면서도 feature의 기여도가 완전히 사라지진 않게 만들 수 있다.**

---

![image-20240915180434951](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180434951.png){: .align-center}

Regularization의 기본 개념은 **파라미터의 값이 더 작은 것**은 **더 단순한 model을 만드는 것**과 비슷하다는 것이다.

또한, feature 수가 적어 overfitting가능성이 낮은 model을 만드는 것이라고도 할 수 있다.

위의 예제에서는 w3와 w4만 Regularization했지만 일반적으로 **feature가 많을 때 어떤 feature에 regularization을 적용해야할지 모르는 경우**가 많기에 **모든 feature에 penalty(regularization)을 적용한다.**

즉, **모든 $w_j$ 파라미터에 penalty를 적용**하면 일반적으로 overfitting이 덜 발생하고 더 잘 fitting할 수 있게 된다.

<br>

예를 들어 100개의 feature를 사용하는 model을 만들어보자. 

어떤 파라미터가 중요한 파라미터가 될지 모르기 때문에 모든 파라미터에 약간의 panalty를 주고 모든 파라미터 값을 축소해보겠다.

![image-20240915180453155](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180453155.png){: .align-center}

Regularization이 적용된 cost function의 식은 위와 같다.

learning rate로 alpha값을 선택했던 것처럼 이제는 **lambda**에 대한 숫자도 선택해야한다.

lambda의 값이 커지면, 알고리즘은 cost function을 최소화하기 위해서 파라미터 $w_j$를 줄여야하며, 이를 이용한 것이 Regularization이다.

<br>

또한, 위와 같이 만들어진 새로운 cost function을 최소화하기 위해서는 mean squared error 부분과 regularization term(정규화 항)부분을 모두 최소화해야한다.

> 아래는 regularization시 사용되는 중요한 관례이다.  
> 1\. lambda에 적합한 값을 선택하는 과정을 더 쉽게 만들기 위해 첫째항처럼 둘째항도 **$\frac{1}{2}$로 나누어 같은방식으로 scaling** 한다.  
> 2\. 파라미터 b가 매우 크다고 해서 **b를 regularization하지는 않는다**. 실제로 b에 regularization을 적용하든 안하든 차이가 거의 없기 때문이다.

![image-20240915180505842](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180505842.png){: .align-center}

수정된 cost function에 대해서 조금 더 살펴보자면, mean squared error를 최소화함으로서 training set에 알고리즘이 잘 맞출 수 있게 되며, Regularization term을 최소화함으로써 파라미터 $w_j$가 작게 유지되어 overfitting이 줄어들 수 있다.

즉, Regularization 적용시 cost function은 위와 같이 새롭게 정의되고 lambda는 data fitting과 파라미터를 작게 만드는 것의 균형을 조절하는 역할을 한다.

---

학습시 lambda 값이 달라지면 어떤 작업을 수행하게 될까?

linear regression을 이용한 주택 가격 예측 예제를 사용해보자.

![image-20240915180523260](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180523260.png){: .align-center}

lambda에 0을 사용하면 regularization term에 0을 곱하기 때문에 regularization term을 전혀 사용하지 않게 된다.

따라서 lambda가 0으로 설정되면 overfitting이 발생하게된다.

<br>

그렇다면 lambda가 매우 큰 수(ex. $10^10$)이라면 어떨까?  

lambda가 매우매우 거대하다면 regularization term의 비중이 너무 커지게 되고, 이 값을 최소화하는 유일한 방법은 파라미터 w를 0에 거의 근접하게 하는 것이다. 이로인해 hypothesis는 b와 같아지므로 learning algorithm은 수평 직선으로 fitting되고 underfitting이 발생하게 된다.

따라서 우리가 원하는 것은 mean squared error와 regularization term이 적절하게 맞춰지도록 그 사이에 있는 lambda값을 구하는 것이다. 

> 나중에 lambda에 적절한 값을 선택할 수 있는 방법을 배운다.

---

## Regularized Linear Regression

아래 사진은 일반적인 mean squared error에 regularization term이생긴 cost function을 보여준다. 

lambda는 regularization parameter이고, regularization이 적용된 cost function을 최소화하는 파라미터 w,b를 찾아야한다.

![image-20240915180535909](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180535909.png){: .align-center}

regularize된 linear regression update 식은 기존과 똑같으며, cost J가 다르게 정의되기에 관련된 derivative(도함수) term이 다르게 정의된다. $w_j$의 derivative term에서는 새롭게 regularization term을 미분한 값이 더해지지만, $b$에는 regularization적용하지 않으므로 derivative term은 그대로 유지된다. 

중요한 점은 여전히 파라미터를 동시에 update해주어야한다는 것이다.

---

### - 그렇다면 파라미터 w는 어떻게 점점 감소하게 되는걸까?

해당 derivative term을 update 식에 대입하면 아래와 같은 식으로 바뀐다. 

![image-20240915180548592](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180548592.png){: .align-center}

$w_j$에 대한 위의 식을 풀면 아래와 같이 나타낼 수 있다.

![image-20240915180604183](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180604183.png){: .align-center}

즉 $w_j$에는 새로운 항이 곱해지게 되며, 해당 값에서 일반적인 update식을 뺀 값이 된다.

$\alpha$는 learning rate로 아주 작은 양수였고, $\lambda$는 보통 작은 숫자(ex. 1 또는 10)이다.

$m$은 training set의 크기(example 개수)이며 50이라고 가정해보자.

<br>

즉 해당 식은 0.0002정도의 작은 양수를 가지게 되며, 1에서 $\frac{\alpha\*\lambda}{m}$를 뺀 값은 1보다 약간 작은 숫자가 된다.

이것은 **gradient descent를 반복할 때 마다 $w_j$에 0.9998을 곱한다는 것이며, 이로인 매 iteration마다 w의 값이 줄어들게 된다.** 

---

### - derivative term 계산법

![image-20240915180619522](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180619522.png){: .align-center}

**편미분**을 이용해 계산한다.

mean squared error 항의 계산은 기존과 같다.

가장 중요한 점은 **regularization term을 미분할 경우 시그마가 사라진다**는 점이다.

<br>

왜 이런 현상이 발생할까?

<br>

포인트는 "**편미분**"을 수행한다는 점이다. 

만약 j=1인 feature에 대해서 편미분을 한다고 해보자.

해당식은 아래와 같이 표현된다.

$$\frac{\partial}{\partial w_1} J(\vec{w}, b)$$

또한, $\sum_{j=1}^{n}w_j^{2}$항은 아래와 같이 표현할 수 있다.

$$w_1^{2}+ w_2^{2}+ w_3^{2}... w_n^{2}$$

이렇게 되면 해당 식을 미분시 $w_1$외의 항은 전부다 상수 취급하여 0이되어 사라진다.

즉, sigma를 통해 더해도 아래와 같이 j=1인 feature에 대한 파라미터만 남는다.

$$2*w_1+ 0 + 0.. 0$$

마찬가지로 같은 원리가 j=2, j=3..j=n인 feature에도 적용되므로 sigma는 더이상 필요가 없어져서 붙지 않게 된다.

따라서 j번째 파라미터에 대한 미분식만 곱해주면 된다.

---

## Regularization to Reduce overfitting

![image-20240915180640297](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180640297.png){: .align-center}

logisitc regression에서 z가 고차 polynomial이면 decision boundary가 지나치게 복잡해지는 overfitting이 일어날수 있었다.

regularization을 사용하려면 위의 식과 같이 regularization term을 추가해주면 된다. 해당 식은 linear regression에서와 똑같다.

regularization term을 사용하면 많은 파라미터가 포함된 고차 polynomial을 fitting하더라도 적합한 decision boundary가 생긴다.

logistic regression에서도 gradient descent를 사용해보자.

![image-20240915180657669](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-5-The%20Problem%20of%20Overfitting/image-20240915180657669.png){: .align-center}

linear regression과 마찬가지로 $w_j$에 대한 derivative term이 추가적인 항을 얻게되며,

logisitic regression에서도 b를 regularization하진 않는다.

update 과정은 linear regression과 매우 비슷하며, $f$(hypothesis)의 정의가 sigmoid function(또는 logistic function)이라는 차이점만이 존재한다.
