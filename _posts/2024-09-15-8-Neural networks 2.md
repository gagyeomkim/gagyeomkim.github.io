---
layout: single
title: "Neural Network 2( Tensorflow&Python )"
categories: ['Machine Learning']
tag: Machine Learning Specialization
typora-root-url: ../

---

## TensorFlow로 neural network구현

### Inference in Code

TensorFlow는 딥러닝 알고리즘을 구현하기 위한 프레임워크 중 하나이다.

해당 강의에서는 inference(추론) code를 구현하는 방법에 대해서 살펴본다.

Learning algorithm이 로스팅 과정에서 얻어지는 원두의 품질을 최적화하는 예를 살펴보겠다.

![image-20240915191222305](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191222305.png){: .align-center}

커피의 로스팅과정에서 생각해볼 수 있는 파라미터는 커피 원두로 만드는 Temperature(Celsius)와 원두를 로스팅하는 Duration(minutes; 시간)이다.

X는 positive class(y=1),O는 negative class(y=0)에 해당한다.

결국 적당한 온도와 적당한 시간을 가지고 로스팅했을 경우 postive class를 가지게 되며, 작은 삼각형 안의 점들이 좋은 커피에 해당하는 것을 확인할 수 있다.

input feature로는 $\vec{x}$ (ex. [섭씨 200도, 17분])가 주어지며 , 이를 이용해 neural network에서 inference를 수행해야한다.

Tensorflow 내에서는 아래와 같이 구성한다.

![image-20240915191230405](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191230405.png){: .align-center}

먼저, x를 두개의 숫자로 구성된 array로 설정한다.(input값은 섭씨 200도, 17분이다.)

이후 `Dense`를 이용하여 `unit=3`과 `activation='sigmoid'`로 설정한다.

이는 해당 layer에 activation function으로 sigmoid function을 사용하는 hidden unit이 3개 있다는 뜻이다.

Dense는 **layer**의 또다른 이름으로, 지금까지 배운 layer와 동일한 layer 유형이다. (Dense뿐 아니라 다른 유형의 layer또한 존재한다.)

이후 x값에 `layer_1` 함수를 적용하여 `a1`을 계산한다. 

`a1`에는 unit이 3개이기 떄문에 3개의 숫자로 구성된 목록이 된다.([0.2,0.7,0.3])

![image-20240915191243354](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191243354.png){: .align-center}

이후 `layer_2`또한 같은 방식으로 만들어서 `a1`까지의 activation value에 `layer_2`함수를 적용하면 `a2`를 계산할 수 있다.

![image-20240915191254343](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191254343.png){: .align-center}

마지막으로 threshold를 설정하는 코드를 통해 $\hat{y}$를 1또는 0으로 구분할 수 있으며, 이것이 Tensorflow를 사용하여 inference를 수행하는 방법이다.

---

digit classification 문제에 적용시켜보자.

![image-20240915191326917](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191326917.png){: .align-center}

x는 pixel intensity의 목록이며, x는 numpy의 array인 `np.array`를 이용하여 해당 값들을 담는다.

이후 layer1을 `layer_1`으로 정의하고 forward propagation을 수행하기 위해 unit과 activation을 설정한 후 input feature에 적용시켜서 `a1`이라는 activation value로 출력해낸다.

![image-20240915191339938](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191339938.png){: .align-center}

나머지 layer들도 똑같은 방법으로 구성하며, input값으로 이전 layer의 activation value를 전달함으로써 forward propagation을 수행해나간다. 이후 threshold를 설정해서 `yhat`을 분류하여 binary prediction을 도출해낼 수 있다.

---

###  Data in TensorFlow

Tensorflow는 numpy가 만들어진 이후 만들어졌기 때문에 Numpy와 Tensorflow에서 data가 표현되는 방식에는 약간의 불일치가 존재한다. 따라서 올바른 코드를 구현하고 neural network에서 실행되도록 하려면 이러한 규칙을 숙지하는 것이 좋다. 

먼저 Tensorflow가 data를 나타내는 방법에 대해 살펴보자.

![image-20240915191351320](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191351320.png){: .align-center}

커피 예제에 위와 같은 dataset이 있다고 가정해보겠다.

`x=np.array([[200.0, 17.0]])`과 같이 사용할 것인데, 왜 대괄호가 2번 사용될까?(`[[`)

matrix의 예를 들어 Numpy가 vector와 행렬을 저장하는 방법을 살펴보겠다.

![image-20240915191402753](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191402753.png){: .align-center}

첫번째 matrix는 2개의 row, 3개의 column을 가진 matrix이며, 우리는 이것을 2x3 matrix(2 by 3 matrix)라고 부른다.

행렬은 $m$(row 개수) x $n$(column 개수)의 dimension을 가지고 있다고 지칭한다.

즉 2x3 matrix를 저장하는 코드에서는 아래와 같이 표현한다.

```
x=np.array([[1,2,3],
			[4,5,6]])
```

대괄호를 보면 [1,2,3]이 matrix의 첫번째 row이고, [4,5,6]이 matrix의 두번째 row라는 것을 알 수 있다.

즉, 첫번째(`[`[) 대괄호는 첫번째 행과 두번째 행을 묶는 역할을 하고,

두번째([`[`) 대괄호는 각 행을 구분하기 위해서 사용한다.

이렇게 대괄호를 2개 이용하여 2-D array로 matrix를 표현한다.

두번째 matrix는 4x2 matrix인 것을 확인 할 수 있으며, 마찬가지로 2개의 대괄호를 이용하여 matrix를 나타내며, 이를 이용해 숫자로 구성된 2-D array를 나타낼 수 있다.

![image-20240915191416184](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191416184.png){: .align-center}

이전 예시에서, 우리는 대괄호 2개를 통해 matrix를 표시했으며, 1x2 matrix와 2x1 matrix또한 같은 원리로 만든다.

용어를 좀 더 설명하자면, 단일 row로 구성된 vector를 **row vector,** 단일 column vector로 구성된 vector를 **column vector**라고 한다.

즉, 1x2 matrix는 row vector, 2x1 matrix는 column vector이다.

이중 대괄호를 사용하는 것과 단일 대괄호를 사용하는 것의 차이는 **단일 대괄호를 사용하면 1-D vector가 만들어진다**는 것이다.

즉, 행이나 열이 없고, value들 목록이 들어있는 1차원 array(배열)일 뿐이다.

Course1에서 linear regression과 logistic regression을 다룰때에는 1-d vector를 사용했었지만 TensorFlow는 매우 큰 dataset도 효율적으로 처리하기 위해서 데이터를 1-D array대신 matrix를 사용하여 내부 계산의 효율성을 높인다.

다시 dataset의 example로 돌아와서 살펴보면, 1x2 matrix로 example을 표현하고 있음을 확인할 수 있다.

![image-20240915191429031](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191429031.png){: .align-center}

---

neural network에서의 propagation을 수행하기 위한 코드를 다시 살펴보자.

![image-20240915191441668](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191441668.png){: .align-center}

`a1`은 3개의 unit을 가지고 있기 때문에 실제로 1x3의 matrix가 된다.

a1을 출력하면 아래와 같은 내용이 출력된다.

```
tf.Tensor([[0.2 0.7 0.3]], shape=(1,3), dtype=float32)
#[[0.2 0.7 0.3]] : 0.2,0.7,0.3의 element를 가지는 matrix이다.
#shape=(1,3) : 1x3 matrix이다.
#dtype=float32 : 부동소수점을 표현하기 위해 float32의 data type을 가진다.
```

_**Tensor**_란 **matrix에 대한 계산을 효율적으로 저장하고 수행하기 위해 만든 data type**이다.

즉, Tensor는 matrix를 표현하는 방법이라고 이해하면 된다.

---

해당 게시물의 첫 부분에서 numpy와 Tensorflow가 matrix를 표현하는 방식에 차이가 존재한다고 설명했었다.

Tensor인 `a1`을 numpy array로 다시 변환하고 싶다면 `a1.numpy()` 함수를 사용하면 된다.(data type을 변경해준다고 생각하자) 

즉, numpy로 data를 load하고 Tensorflow에 전달하면 Tensorflow의 내부 형식인 Tensor를 이용하여 작동하며, 다시 Numpy array로변환할 수 있다. (**추가적인 변환 작업**을 거쳐야한다.)

![image-20240915191456413](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191456413.png){: .align-center}

layer 2의 경우도 마찬가지로 진행된다.

---

### Building a neural network

![image-20240915191506656](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191506656.png){: .align-center}

forward prop을 수행하려면 위와 같이 code를 구축하면 된다.

layer1을 만들고 activation value를 출력하며, 이 값을 다시 layer2에 전달하는 형식이다.

![image-20240915191519362](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191519362.png){: .align-center}

하지만 수동으로 data를 전달하는 것보다 Tensorflow에 layer1과 layer2를 가져와서 서로 연결하여 neural network를 형성하면 더 좋을 것 같다.

`Sequential()`함수가 이러한 기능을 하는데, 이 함수는 layer를 순차적으로 연결하여 neural network를 만든다.

layer를 연결하여 model 객체에 담으면, 해당 model에 `.`을 이용해 함수를 사용할 수 있다.

-   `model.compile` : 파라미터를 사용하여 호출, 신경망을 훈련시키기 전 compile 작업을 수행함
-   `model.fit` : layer가 순차적으로 연결된 model을 가져와 x와 y데이터를 기반으로 훈련하도록 지시함 
-   `model.predict`:`x_new`라는 새로운 example이 있다고 가정할 경우, input feature `x_new`를 전달받아 forward prop을 수행하고,  최종적인 추론값을 출력함.(여기선 $\vec{a}^{[2]}$를 출력함)

![image-20240915191533293](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191533293.png){: .align-center}

실제 Tensorflow code를 보면 forward prop model일 경우 관례상 layer에 대한 변수를 명시적으로 할당하는 것이 아니라 위와 같이 Sequential 함수 안에 Dense를 이용해서 바로 layer 만든다.

---

숫자 분류 예제에서 이 작업을 수행해보면,

layer를 지정하고(물론 model 안에서 구현해줘도 된다)

`Sequential`, `compile`, `fit,` `predict` 함수를 이용해서 예측 model을 만들고 있다.

> 위에서 알아본 것과 크게 차이가 나지 않는다.

![image-20240915191546258](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191546258.png){: .align-center}

---

## Python으로 신경망 구현

### Forward prop in a single layer

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191556170.png" alt="image-20240915191556170" style="zoom:80%;" />{: .align-center}

여기서는 coffee roasting model을 계속 사용할 섯이다.

input feature x를 가져와서 ativation value a2를 얻는 방법에 대해서 살펴보겠다.

해당 python 구현에서는 1-D array를 사용하여 표시할 것이다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191612468.png" alt="image-20240915191612468" style="zoom:80%;" />{: .align-center}

추가적으로, 위와 같이 위 첨자는 바로 붙여서, 아래 첨자는`_`로 분리해서 나타낸다.

![image-20240915191626424](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191626424.png){: .align-center}

즉, 각 unit마다 activation value를 계산하는 과정은 위와 같다. 코드를 보고 이해할 수 있을 정도이면 된다.

모든 계산이 끝난 후 activation value를 `np.array`로 묶어서 vector로 취급하여 a1에 넘겨주었다.

![image-20240915191641144](https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191641144.png){: .align-center}

a2도 동일한 방식으로 계산하면 된다. w와 feature x의 dimension을 꼭 맞춰줘야한다는 것에 유의하자.

---

### General implementation of forward propagation

이전 비디오의 예시를 사용해서 forward propagation을 수행해보자

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191651633.png" alt="image-20240915191651633" style="zoom: 67%;" />{: .align-center}

$w_1, w_2, w_3$를 column vector로 나타내었고, 이를 $W$라는 matrix에 담았다. **column별로** w 파라미터가 몇번째 neuron의 것인지를 판단하려고 한다. 따라서 해당 matrix는 2 by 3의 형태를 가지게 되며, 

b는 단순히 scalar 값이기 때문에 1-d array에 담았으며, input인 $a^{[0]}$또한 1-d array로 나타냈다.

<img src="https://cdn.jsdelivr.net/gh/gagyeomkim/gagyeomkim.github.io@master/images/2024-09-15-8-Neural%20networks%202/image-20240915191708584.png" alt="image-20240915191708584" style="zoom: 67%;" />{: .align-center}

`dense`함수는 이전 layer의 activation을 `a_in`이라는 argument로 받아서 현재 layer의 activation을 출력하는 함수이다.

W는  2 by 3 matrix이므로, `W.shape[1]`은 3이다.

for 문을 이용해 unit의 개수만큼 반복을 돌리면서 각각의 activation을 출력해낸다.

이후 `sequential`함수를 정의해서, 각각의 layer마다 activation을 순차적으로 계산하는 forward propagation을 수행한다.

> 대문자 $W$는 matrix(행렬), 벡터와 스칼라는 소문자를 사용한다는 점에 유의하자
