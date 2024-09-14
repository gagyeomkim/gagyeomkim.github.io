---
layout: single
title: "Logistic Regression 2 (Cost Function, Gradient Descent)"
categories: ['Machine Learning']
tag: Machine Learning Specialization
typora-root-url: ../
---

> Coursera의 Machine Learning Specialization강의를 정리한 내용입니다
 
이번글에서는 Logistic Regression의 Cost Function, Gradient Descent에 대해서 알아볼 것이다.

## Cost function for logistic regression

![image-20240915024304341](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915024304341.png)

> $m$ : training example의 개수. i를 통해 indexing함  
> $n$ : feature의 개수. j를 통해 indexing함

logisitc regression은 binary classification 작업이므로 target label y는 0 또는 1 두개의 값만을 취한다.

logisitc regression model은 위와 같은 방정식으로 정의될 수 있는데, training set을 사용할 때 적합한 파라미터 w와 b를 어떻게 고를까?

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915024554891.png" alt="image-20240915024554891" style="zoom: 67%;" />

linear regression에서 사용했던 Squared error cost function을 살펴보자.

$\\frac {1}{2}$를 summation안에 넣음으로써 식을 약간 변형하였다.

이때의 hypothesis는 $\\vec{w} \\cdot \\vec{x} +b$이다.

linear regression에서는 Squared error cost function을 이용해서 plotting할 때 convex(볼록) 모양의 함수가 나왔음을 기억할 것이다.

따라서 gradient descent를 통해 global minimum에 접근할 수 있었다.

---

logistic regression에서도 해당 cost function을 사용하면 어떨까?

logistic regression에서는 hypothesis가 sigmoid함수였다.

![image-20240915024622730](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915024622730.png)

logistic regression에 Mean Squared error cost function을 이용하여 plotting할 경우에는 위와 같이 non-convex(볼록하지 않은)한 형태의 그래프가 plotting되며, local minimum이 여러개 생기게 되어서 gradient descent를 이용할 때 global minimum에 convergence하도록 보장할 수 없다.

이를 통해 우리는 logistic regression에 사용할 수 있는 새로운 cost function이 필요함을 알 수 있다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915024635226.png" alt="image-20240915024635226" style="zoom:67%;" />

그전에 우리는 cost function J의 정의를 조금 변경해볼 것이다.

사각형안의 내용을 _**Loss**_라고 표현할 것이며, 대문자 $L$을 이용해서 나타낼 것이다.

즉, Loss는 **제곱 오차의 1/2배**한 식을 나타내며, 이를 식으로 나타내면 아래와 같다.

 $$L( f\_{w,b}(x^{(i)} ,y^{(i)}) = \\frac{1}{2m}(f\_{w,b}(x^{(i)}) - y^{(i)})^2 $$ 

> loss와 cost를 헷갈리지 말자!  
> loss function은 하나의 training example에서 계산한 값이며,  
> cost function은 모든 training example의 loss를 합산해 평균낸 값이다.

logistic regression에 사용할 loss function의 definition을 적어보자.

![image-20240915024648002](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915024648002.png)

loss function은 위와 같은 식으로 정의된다.

target label y가 1일 때와 0일 때로 나눠서 생각해보자.

> 아래의 식은 hypothesis에 대한 식이다. 기억이 나지 않는다면 전 게시글을 한번 더 살펴보자  
> $ f\_{\\mathbf{w},b}(\\mathbf{x^{(i)}}) = g(z^{(i)}) $  
> $ z^{(i)} = \\mathbf{w} \\cdot \\mathbf{x}^{(i)}+ b $  
> $ g(z^{(i)}) = \\frac{1}{1+e^{-z^{(i)}}} $

### \- $y^{(i)}$=1일 때 

![image-20240915024717893](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915024717893.png)

$y^{(i)}$=1일 때의 loss function은 $-log(f(x))$함수로 표현된다.

logistic regression의 출력값($g(z)$)은 항상 0과 1사이이므로 $f$(=$g(z)$)는 항상 0과 1사이여야한다.

\-log 함수에서 해당하는 범위를 가져와서 확대해보자.(x축은 Hypothesis의 예측값이다.)

실제 target label y가 1의 값을 가지므로,

만약 알고리즘의 예측값($f(x)$)이 1에 가까운 값을 가진다면(ex.0.9) loss는 0에 근접할 것이며,

예측값이 0에 가까운 값이라면(ex.0.1) loss는 $\\infty$만큼 커질 것이다.

(loss는 예측값과 실제값의 차이를 포함하고 있기 때문에)

**즉, 오른쪽 어구와 같이 Loss는 f(x)의 예측값이 true label y에 가까울 때 가장 작다.**

알고리즘은 loss가 작을때 incentive를 제공하거나 알고리즘을 강화해서 더 정확한 예측을 수행할 수 있게 한다.

### \- $y^{(i)}$=0일 때 

$y^{(i)}$=0일 때의 loss function은 $-log(1-f(x))$함수로 표현된다.

logistic regression의 출력값($g(z)$)은 항상 0과 1사이이므로 $f$(=$g(z)$)는 항상 0과 1사이여야한다.

해당 범위를 가져와서 확대해보자.

![image-20240915024733647](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915024733647.png)

이번엔 실제 target value가 0의 값을 가지고 있으므로

f(x)의 예측값이 0에 가까우면(ex.0.1) loss는 0에 근접할 것이고,

f(x)의 예측값이 1에 가깝다면(ex.0.999) loss는 $\\infty$에 근접할 것이다.

즉, f(x)의 예측값이 실제값인 target $y^{(i)}$와 멀수록 더 높은 loss를가지게 될 것이다.

### \- Cost

![image-20240915024747295](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915024747295.png)

즉, Cost를 정의해보면 위와 같은 사진으로 한번에 나타낼 수 있다.

이렇게 loss function을 사용하면 전체 cost function을 convex(볼록)하게 만들 수 있으므로, gradient descent를 이용해서 global minimum에 도달할 수 있다.(왜 cost function이 볼록하게 되는지는 따로 강의의 범위를 벗어나기에 증명은 나중에 시도해보자) 

Cost function은 **전체 training example의 loss를 평균낸 것**이며,cost function을 최소화할 수 있는 파라미터 w,b 값을 찾아야한다.

---

## **Simplified** Cost Function for Logisitic Regression 

정의한 cost function을 좀 더 단순한 방정식으로 다시 한번 정의해보자.

![image-20240915024803649](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915024803649.png)

binary classification문제에서 target value y는 0이거나, 1의 값만을 가질 수 있으므로, loss function을 위와 같이 하나의 방정식으로 적을 수 있다.

![image-20240915024814964](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915024814964.png)

y=1인 경우, 뒤의 식이 0이되어 사라지고, 앞의 식만 남게 되는데, 이는 위에서 정의한 y=1일때 Loss function의 식과 일치한다.

![image-20240915024828088](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915024828088.png)

y=1인 경우, 앞의 식이 0이되어 사라지고, 뒤의 식만 남게 되는데, 이는 위에서 정의한 y=0일때 Loss function의 식과 일치한다.

이처럼 loss function을 한줄로 적게 되면, cost function도 좀 더 단순화할 수 있다.

![image-20240915025314752](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915025314752.png)

우선 맨 위의 식과 같이 loss function을 한줄의 방식으로 표현한다.

단순화된 loss function의 식을 cost function에 가져와 연결하고, -부호를 앞으로 빼면  최종적으로 아래와 같은 표현식을 얻을 수 있으며, 이것이 logistic regression을 training 시킬 때 가장 많이 사용하는 cost function이다. 

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915025219498.png" alt="image-20240915025219498" style="zoom:67%;" />

해당 cost function은 convex한 속성을 가지고 있다.

> Q. cost function이 수없이 많을 텐데 왜 이 cost function을 사용할까?  
> A. 다양한 model에 대한 매개변수를 효율적으로 찾는 방법인 maximum likelihood라는 통계적 원리를 사용하여 해당 cost function을 도출해냈다고 한다. 지금은 이것에 대해 자세히 알 필요는 없다.

실제로 파라미터를 변형시켜서 확인했을 때, 더 나쁜 decision boundary를 갖는 model에서는 더 큰 cost 값이 나옴도 확인할 수 있다.

(target과 예측값의 차이가 커지는 경우가 발생하기 때문)

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915025348126.png" alt="image-20240915025348126" style="zoom:67%;" />

---

## Gradient descent

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915025405341.png" alt="image-20240915025405341" style="zoom: 50%;" />

해당 강의에서는 파라미터 w와 b를 잘 선택하는 방법에 대해 살펴본다.

파라미터를 선택한 후 x라는 새로운 input을 주면 model이 예측을 하고, label y가 1일 확률을 추정할 수 있다.

![image-20240915025426099](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915025426099.png)

일반적으로 cost를 최소화하는데 사용하는 방법은 gradient descent이다.

logistic regression에서 **파라미터를 update 하는 방법은 linear regression에서의 update방법과 똑같다.**

J의 도함수를 계산해서 기존 식에다가 대입하는 방법또한 같다.

중요한 점은, 파라미터를 동시에 update해야한다는 것이다.

즉, 파라미터 w와 b에 대한 update 방정식은 linear regression에서 생각해냈던 것과 똑같다.

![image-20240915025441974](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-4-Logistic Regression2(Cost Function, Gradient Descent).md/image-20240915025441974.png)

하지만 이 방정식들이 똑같아도 **linear regression이 아닌 이유로는 f(x)(Hypothesis)에 대한 정의가 바뀌었기 때문이다.**

linear regression에서는 $f\_{\\mathbf{w},b}(x^{(i)})=\\vec{w} \\cdot \\vec{x} + b$라는 hypothesis를 가지고 있었지만,

logistic regression에서는 hypothesis로 sigmoid함수가 사용된다.

즉 update에서 작성된 알고리즘이 같아보여도 **f(x)에 대한 정의가 동일하지 않기 때문에 매우 다른 알고리즘**이다.

또한, logistic regression에서도 linear regression에서 사용한 방법들을 똑같이 사용할 수 있다.

-   gradient descent를 모니터링한 learning curve(iteration에 따른 cost 그래프)를 통해서 cost가 수렴하는지 판단했었는데,  
    logistic regression에서도 똑같이 사용된다.
-   vectorization을 이용할 수 있다.
-   서로 다른 feature을 scaling하여 비슷한 범위를 가지도록하는 Feature scaling을 통해 gradient descent의 속도를 높일 수 있다.

---

### \- scikit-learn을 사용한 logistic regression

```python
import numpy as np

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
```

1\. \`fit\` function : model을 training data에 fitting할 수 있다.

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)
```

2\. \`predict\` function : model에 의해 만들어진 prediction을 볼 수 있다.

```python
y_pred = lr_model.predict(X)

print("Prediction on training set:", y_pred)
```

3\. \`score\` function : model의 정확성을 계산할 수 있다.

```python
print("Accuracy on training set:", lr_model.score(X, y))
```

