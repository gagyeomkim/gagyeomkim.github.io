---
layout: single
title: "Supervised Learning & Unsupervised Learning"
categories: ['Machine Learning']
tag: Machine Learning Specialization
last_modified_at: 2024-09-14T12:00:00
typora-root-url: ../
---



> Coursera의 Machine Learning Specialization강의를 정리한 내용입니다.

## Supervised learning

Supervised learning(지도학습)은 x에서 y로 또는 **input(입력)과 output(출력) label의 데이터 셋이 주어져서** 이 정보를 통해 input과 output의 관계를 유추하는 것이다. 즉, **기존 정보를 토대로 출력값이 없는 새로운 input에 대해 output을 추측하는** 학습 방법이다.

<br>

다시 말해, 모델이 x와 y의 쌍으로부터 학습한 후에는 **이전에 본적이 없는 새로운 입력 x를 받아 적절한 출력값 y를 생성**하게된다.

<br>

Supervised Learning은 크게 **Regression(회귀)**과 **Classification(분류)**으로 나뉜다.

<br>

### Regression(회귀)

Regression이란 무한히 많은(모든 범위에서) 가능한 출력값을 예측하는 것이다.

![image-20240914184830646](https://1drv.ms/i/s!AvDtmE0jTiDWgkrOHwjVF67IAAJb?embed=1&width=1150&height=646){: .align-center}*regression의 예*

x축은 House size이며, y축은 Price를 의미한다. 우리에게는 X표시로 dataset이 주어졌다.

750 평방피트의 주택가격이 얼마인지 알고싶다고 가정해보면,

데이터에 가장 적합한 **직선**(150k 예상)이나 **곡선**(200k 예상) 또는 다른 것을 선택함으로써 주택의 예상가격을 예측할 수 있다.



### Classification(분류)

Classification이란 가능한 output의 **작은 집합만을 가지고** **출력 클래스(=출력 카테고리)를 예측하는 것**이다.

즉, **output이 될 수 있는 것이 정해져있다.**

>  분류과정에서 출력 클래스나 출력 카테고리는 같은 의미로 쓰인다.

<img src="https://1drv.ms/i/s!AvDtmE0jTiDWgkvQjPYP3_qT43MN?embed=1&width=1060&height=496" style="zoom:80%;" />{: .align-center}

의사가 유방암을 진단하기 위해 기계학습 시스템을 구축하고 있다고 가정하자.

위 그래프는 종양의 사이즈에 따른 악성(1)/양성(0) 여부이다.

가로축은 종양의 size를, 세로축은 두개의 값(0 또는 1)만을 나타내는 그래프로 데이터를 표시할 수 있다.

<br>

또한, 해당 예제에서는 가능한 범주(Category)가 2개 뿐이므로 dataset을 두번째처럼 하나의 직선상에 표시할 수도 있다.

양성 class를 O, 악성 class를 X 기호를 사용해서 표시하자.

<br>

Classification에서는 아래와 같이 출력 클래스가 2개 이상인 **Multiclass classification**(다중분류)도 가능하다.

![image-20240914185244449](https://1drv.ms/i/s!AvDtmE0jTiDWgkzPUYQ-02RHJKyx?embed=1&width=1250&height=257){: .align-center}

>  class는 0,1,2 같은 숫자일 수도 있다.
>
>  그러나 숫자를 해석할 때 classification는 0,1,2와 같이 가능한 **출력 범주의 작고 유한한 제한된 집합을 예측**하지만 regression은 가능한 모든 숫자를 예측한다는 점이 다르다.(0.5 또는 1.7...등등)
>
>  <br>
>
>  물론, 출력 클래스가 꼭 숫자일 필요는 없다.
>
>  ex. 사진이 고양이인지 개사진인지 예측하는 것, 양성/악성 종양을 구분하는 것도 Class가 될 수 있다.

<br>

<img src="https://1drv.ms/i/s!AvDtmE0jTiDWgk3ASSPvZOShh94l?embed=1&width=670&height=553" style="zoom:50%;" />{: .align-center}

또한, 둘 이상의 입력값을 사용하여 출력값을 예측할 수 있다.

위 그래프는 나이(Age) 조건을 추가해서 종양의 size와 age에 따른 악성/양성 그래프이다.

학습 알고리즘은 악성 종양과 양성 종양을 구분하는 boundary(경계)를 찾음으로써, 종양의 종류를 구분 할 수 있다.

따라서 학습 알고리즘은 해당 데이터를 통해 어떻게 boundary line을 맞출지 결정해야한다.

------

## Unsupervised Learning(비지도 학습)

Unsupervised Learning(비지도학습)은 라벨링되어있지 않은 데이터에서 흥미로운 무언가(구조, 패턴 등)를 찾는 것이다.

즉, **알고 있는 출력값(label y)없이 학습하는 머신러닝**을 의미한다. 오직 input data만으로 데이터에서 지식을 추출할 수 있어야한다.

다시말해, x에 따른 y의 출력을 예측하는 것이 아니라 주어진 **dataset의 구조나 패턴을 찾아내는 것**이다.

### Clustering(군집화)

Clustering이란 각 주어진 data들이 얼마나 유사한지에 따라 데이터를 군집으로 분류하는 것이다.

<img src="https://1drv.ms/i/s!AvDtmE0jTiDWgk6oqh386M4loVti?embed=1&width=670&height=525" style="zoom:50%;" />{: .align-center}

<img src="https://1drv.ms/i/s!AvDtmE0jTiDWgk-rcN9waYGPuVJR?embed=1&width=860&height=578" alt="image-20240914185318271" style="zoom:50%;" />{: .align-center}

위의 사진에서, Google news에서 클러스터링을 사용하는 것을 알 수 있다. 인터넷에 있는 수십만개의 뉴스 기사를 살펴보고 관련기사를 그룹화하고 있다.(다양한 기사에서 공통으로 언급된 단어가 발견되어 클러스터링 알고리즘이 해당 기사들을 찾음을 알 수 있다.)

<img src="https://1drv.ms/i/s!AvDtmE0jTiDWglDMAwENQijo9EP9?embed=1&width=960&height=539" style="zoom:50%;" />{: .align-center}

Clustering 알고리즘은 유전학 자료를 연구하는 데에도 사용된다.

위의 사진은 DNA microarray 데이터를 보여준다.

작은 열 하나하나가 한 사람의 DNA를 나타내며, 각 행은 특정한 유전자(눈색깔, 키..등등)를 나타낸다.

각 색은 개체마다 특정 유전자가 활성화 되어 있거나 활성화되지 않은 정도를 나타낸다.

클러스터링 알고리즘을 통해 개인을 **Type1, Type2, Type3와 같이 여러 범주로 그룹화**할 수 있다.

> 알고리즘에 특정한 특성을 가진 Type1의 사람이 있다고 미리 알려주는 것이 아니기에 비지도 학습의 일종이다.

<img src="https://1drv.ms/i/s!AvDtmE0jTiDWglF0q6CIv3Y5kwTl?embed=1&width=960&height=552" style="zoom:50%;" />{: .align-center}

서비스를 제공할 때, 시장 세분화를 통해서 사람들을 구별하는 것도 Clustering의 한가지 예이다.

------

### Anomaly detection & dimensionality reduction

또한, Clustering외에도 다른 유형의 비지도 학습이 존재한다.

**- Anomaly detection(이상탐지)**

이상 탐지는 비정상적인 이벤트를 탐지하는 데 사용된다. 금융 시스템 및 기타 여러 애플리케이션에서 사기 탐지에 매우 중요한 역할을 한다.

**- dimensionality reduction(차원 축소)**

dimensionality reduction은 큰 데이터 집합을 가져와서 정보를 최대한 적게 손실하면서 더 작은 데이터 집합으로 압축하는 것을 말한다.

------

## reference

* [https://junstar92.tistory.com/13](https://junstar92.tistory.com/13){: .align-center}