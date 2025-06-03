---
title: "[EECS 498-007] Lecture2: Image Classification"

categories: ["EECS 498-007 Deep Learning for Computer Vision"]
tags:
  - ["Deep Learning", "Computer Vision"]

date: 2025-03-02
last_modified_at: 2025-03-02
---

![](https://i.imgur.com/3ogD3Ke.jpeg)
Lec2에선 Image Classification에 대해서 알아본다.

![](https://i.imgur.com/rTN0iFz.jpeg)
Computer vision이라는 큰 분야에는 여러가지 문제들이 존재한다.

그 중에서도 Image Classification은 컴퓨터 비전의 핵심적인 task로, Input으로 Image가 주어지면, Output으로 해당 이미지의 label 하나를(고정된 카테고리 셋에서) 할당해주는 작업을 말한다.

![](https://i.imgur.com/3XG1lg3.png)
사람과 달리, 컴퓨터는 Image를 low-level 수준에서 보게 된다. 즉,  고양이 이미지를 보고 '고양이'라는 것을 바로 인식할 수 없고, 단순히 0~255 사이의 숫자로 표현된 픽셀 값으로 이루어진 그리드로 이미지를 인식하는 것이다. 

이처럼, 어떻게 컴퓨터가 이미지를 인식할 것인지에 대한 gap이 존재하는데, 이를 **Semantic Gap**이라고 한다.

![](https://i.imgur.com/wrXOlWU.png)
## Image Classification의 Challenges
Image classification은 다양한 challenge(난제)들을 가진다.
### Viewpoint Variation
![](https://i.imgur.com/o9uPLKa.png)

첫번째 Challenge는 Viewpoint Variation이다.

사람은 고양이 이미지를 다른 각도에서 보더라도, 여전히 '고양이'라고 인식할 수 있지만, 컴퓨터가 이미지를 볼 때는, 카메라 각도가 변경될때마다 매번 픽셀값이 바뀐다. 이 경우, 컴퓨터가 이미지를 '고양이'로 판단하기 어렵다.

이렇게 픽셀값들의 변화에도 불구하고, 고양이를 고양이로 인식할 수 있는 robust(<font color="#bfbfbf">변화가 거의 없는; 강한</font>)한 model을 만드는 것이 Image classification task의 목표가 된다

### Intraclass Variation
![](https://i.imgur.com/P9zyW40.png)

두번째 Challenge는 Intraclass Variation이다.

사진 속 고양이들은 다양한 모습을 가지고 있다.
이처럼, 같은 고양이(class)들끼리도 다른 픽셀값을 가지고, 같은 종이라도 차이(Variation)를 가지고 있다.

서로 다른 픽셀값을 가진 고양이들을 어떻게 모두 '고양이'로 인식할지도 문제이다.

### Fine-Grained Categories
![](https://i.imgur.com/ufVGBNG.png)

세번째 Challenge는 Fine-Grained Categories이다.

어떻게 시각적으로 비슷하게 보이는 고양이들을 서로 다른 종, 즉, **다른 하위범주**로 인식할지에 대한 것이다. 
세분화된 카테고리를 나누는 작업은 매우 어렵다고 한다.

### Background Clutter
![](https://i.imgur.com/Rkq1wfZ.png)

네 번째 Challenge는 Background Clutter이다.

바깥 배경과 고양이의 픽셀값에 큰 차이가 없을 경우에, 고양이를 탐지하는 것이 어려워진다.(배경 부분을 고양이로 인식하게 될 수 있다.)

### Illumination Changes
![](https://i.imgur.com/2rcCQYp.png)

다섯번째 Challenge는 Illumination Changes이다.

scene의 조도에 변화가 생겨 픽셀값이 달라지는 경우에도, 제대로 판단할 수 있어야한다.

### Deformation
![](https://i.imgur.com/xzmH99h.png)
여섯번째 Challenge는 Deformation이다.

때로는 고양이가 매우 다른 포즈를 취할 수도 있다.

객체가 다양한 형태를 가지는 경우에도, 올바르게 판단할 수 있어야한다.

### Occlusion
![](https://i.imgur.com/4yYwtTU.jpeg)
마지막 Challenge는 Occlusion이다.

고양이 전체가 보이지 않고, 일부가 가려져 있거나 일부만 나타나는 경우에는 어떻게 인식할지에 대한 문제이다.

> 픽셀 단위의 이미지를 보고 고양이라고 어떻게 인식시킬 것인지 그리
> 고 이러한 Challenge들에 대해서 robust한 분류기(classifier)를 어떻게 만들 것인지는 Computer vision의 근본적인 문제로 남아있다.

## Image Classification의 유용성
![](https://i.imgur.com/hsn0BCA.jpeg)
Image Classification은 의학 분야의 양성/악성 진단, 해양 연구, 은하의 분류 등 다양한 분야에서 활용된다.

![](https://i.imgur.com/yyi6XGv.jpeg)
또한, Image Classification은 다른 Computer Vision task의 **building block**이 된다.

예를 들어, Object detection은 이미지 내의 물체가 무엇인지 판단하는 것 뿐만 아니라 이미지 어딘가에 있을 사물의 위치까지 알아내길 원한다. Image Classification은 이런 Object detection과 같은 task에 sub part로써 작용한다. 

![](https://i.imgur.com/XV5dgEe.jpeg)
Object detection의 수행 방식을 보면, window 하나를 잡고, 이동시키면서 image의 sub-region을 classification하는 것이다. 
![](https://i.imgur.com/h24rXtf.jpeg)
이를 통해, 사람과 말을 감지할 수 있다.
![](https://i.imgur.com/2XoRNg4.jpeg)

![](https://i.imgur.com/ZGgUK7m.jpeg)

![](https://i.imgur.com/PyP5MEK.jpeg)

![](https://i.imgur.com/ELypq0K.jpeg)
Image Captioning task도 Computer vision task의 일종이다.

Input으로 들어온 image에 대해서 classification의 과정을 통해 'man'과 'horse'라는 label을 획득하고, Output으로는 언어끼리의 관계를 잘 이용하여 문장을 출력하는 작업이다.

이것 또한, Object detection과 마찬가지로 sequence of image classification이라고 볼 수 있다.
![](https://i.imgur.com/lqg1qo0.jpeg)
바둑 게임을 위한 AI에도 Image Classfication이 사용된다고 한다.

input : 현시점에서 돌의 위치가 포함된 image

output : 다음에 돌을 놓아야할 위치
![](https://i.imgur.com/C85WESP.jpeg)
이러한 Image Classification의 작업을 해주는 것을 Image Classifier라고 한다.

Image Classifier의 함수를 정의할 때, Input은 image, output은 label이 반환되도록 만들어야한다.

하지만, 이를 잘 표현할 수 있는 알고리즘을 표현할 명백한 방법은 보이지 않는다.
![](https://i.imgur.com/hBGllNb.jpeg)

생각해볼 수 있는 방법은 이미지에서 edge는 매우 중요한 역할을 한다는 점을 이용하는 것이다. 

그러나, "고양이의 귀나, 수염과 같은 edge의 종류에 따라서, 이러한 edge에 해당하면 '고양이'라고 판별한다"와 같은 hard-code algorithm을 통해 문제를 해결한다면,  다른 객체(ex. 은하분류)가 이미지로 주어졌을 때, 또 다른 algorithm을 생각해내야한다. 

![](https://i.imgur.com/HY5jC4c.jpeg)
따라서,  data-driven approach(데이터 기반 접근법)을 사용할 수 있다.

data-driven approach는 data가 바뀌더라도, 알고리즘을 수정할 필요 없이 프로그램이 이 데이터에 맞춰 자동으로 행동을 변화시킨다. 

머신러닝이 바로 data-driven approach를 사용한다. 머신러닝의 파이프라인은 아래와 같이 나타낼 수 있다.
1. 몇 만장의 이미지와 label을 수집(collect)한다.
2. 머신러닝을 사용해서 classifier를 train한다.
3. 새로운 test image가 input으로 들어왔을 때, model이 예측한 test image의 label을 반환한후, classifier를 evaluate(평가)한다.

이에 대해 중요한 두가지 API가 train과 predict이다.
- `train(images, labels)` : image와 label을 input으로 받아, training한 model을 return받는다.
- `predict(model, test_images)` : model을 사용하여 test images에 대한 label을 예측한 뒤, 출력한다.



>  Image Classification에는 대부분 이러한 패러다임이 적용된다.

## Well-known image datasets
> Dataset의 정보는 사진을 참고하자.

Image Classification을 수행하기 위해, 잘 알려진 데이터셋들에 대해서 알아보자.

![](https://i.imgur.com/KIjVyNZ.jpeg)
MNIST는 가장 유명한 데이터 셋이다.

grayscale 이미지로, channel이 1이다.

![](https://i.imgur.com/QIhyce6.jpeg)

MNIST 데이터셋은 컴퓨터비전에서 "초파리"라고도 불린다.  

생물학자들이 초파리를 가지고 가장 먼저 실험을 하듯이, CV 연구자들도 MNIST에 대해서 새로운 아이디어를 가장 먼저 실험해본다고 한다.

MNIST는 새로운 아이디어를 실현해보기 좋은 데이터셋이지만, MNIST에서 잘 된다고 다른 데이터 셋에서도 잘된다는 보장이 없기 때문에, MNIST에서 잘 되는 것은 proof of concept(개념검증; PoC)가 성공했다는 정도로만 판단하면 된다.
![](https://i.imgur.com/Nd35VfH.jpeg)
또다른 데이터셋은 CIFAR10이다. 과제에서도 CIFAR10 데이터를 이용할 것이다.

MNIST보다 label들이 분류하기 힘든편에 속하는 것들로 이루어져 있다.

![](https://i.imgur.com/FiEUCCt.jpeg)
CIFAR10에서 확장된 CIFAR100도 있다. 20개의 superclass(상위범주) 아래에 5개의 class가 추가로 존재하는 식이다.

![](https://i.imgur.com/LQHbB3N.jpeg)
ImageNet은 Image classification에서 알고리즘이 잘 작동하는지를 검증하기 위한 gold standard로 활용되고 있다.

검증받고자하는 알고리즘이 해당 데이터셋에서 잘 작동하는지를 확인해야하며, 만약 제대로 작동하지않는다면 알고리즘은 reject해야한다.

ImageNet은 잘 분류하기 어렵기 때문에, evaluate시에는 Top 5 Accuracy를 사용한다. 
Top 5 Accuracy : 특정 알고리즘이 label을 예측한 후, 그 중에서 true label이 있다면 잘 예측한 것으로 판단한다.
![](https://i.imgur.com/IfUZjN0.jpeg)
하지만, training에 너무 많은 시간이 걸리고, computing budget도 많이 필요하다는 단점을 가진다.
![](https://i.imgur.com/9OQUjYu.jpeg)
ImageNet이 Object에 집중한 반면, MIT Places는 scene(장면)에 집중한 dataset이다. 샘플 하나하나가 다양한 사이즈를 가지고 있고, 일반적으로 256x256사이즈로 resizing하여 training시킨다.
![](https://i.imgur.com/w7xt7jB.jpeg)
연구 트렌드는 더 큰 데이터, 더 큰 레이블 데이터 셋을 

최대한 robust하면서도 최소한의 계산시간으로 classification하는 것이다.
![](https://i.imgur.com/dAe3CnZ.jpeg)
앞서 소개한 수많은 데이터를 포함하고 있는 데이터셋들과 달리 각 category(=class)에 20개의 이미지만 가지고 있는 Omniglot이란 데이터셋도 있다. 오히려 모델이 적은 수의 데이터에 대해서도 robust하게 작동하는지 확인하기 위해 사용한다.

## k-Nearest Neighbors
처음으로 다뤄볼 Classifier는 Nearest Negibor이다.

파이썬 함수를 정의하는 형태로 Nearest Neighbor를 살펴보자. 

![](https://i.imgur.com/22VE20R.jpeg)
`train()` : input으로 들어온 모든 data와 label을 저장한다.

`predict()` : input으로 새로운 샘플(test image)이 들어왔을 때, 학습한 image set 중에서 input 이미지와 가장 비슷한(L1 distance 관점에서는 가장 가까운) 이미지가 어떤 label을 가지고 있는지를 확인하고, 가장 비슷한 이미지의 label로 input 이미지의 label로 예측하여 출력한다. 

![](https://i.imgur.com/15dWwg3.jpeg)
'비슷한' 이미지란 무엇일까? 수학에서 비슷하다는 말은 similarity(유사도)란 말로 나타낸다.

similarity를 나타내는 가장 대표적인 방법으로 L1 distance와 L2 distance가 있다.

1. L1 distance(Manhattan distance): 두 개의 이미지(픽셀 수 동일)
    의 각 픽셀 위치에 대응되는 요소끼리 element-wise하게 뺀 뒤(절댓값으로 계산), 그 차이를 모두 합함.
2. L2 distance(Euclidean distance): 두 개의 이미지(픽셀 수 동일)
    의 각 픽셀 위치에 대응되는 요소끼리 element-wise하게 뺀 값($a-b$)을 제곱하여 모두 합함. 이후 그 값에 루트를 씌움



![](https://i.imgur.com/JbtwLot.jpeg)
![](https://i.imgur.com/xudnxXi.jpeg)
코드 상으로 train 단계에서는 모든 training data를 저장하고 있다.
![](https://i.imgur.com/onARdga.jpeg)
이후, test set의 수만큼 반복하면서, 각 test image마다 모든 training set과의 distance를 구하고(여기서는 L1 distance), distance가 가장 낮은 것을 예측 label로 정하고 있음을 확인할 수 있다.

### Nearest Neighbors의 문제점
Nearest Neighbors에 대한 몇가지 질문을 생각해보자.
![](https://i.imgur.com/R206tNh.jpeg)
Q: N개의 샘플이 있을때, training 시간은 얼마나 걸릴까?

![](https://i.imgur.com/bEP6jTA.jpeg)
A: O(1). training 시간에는 문제가 없다.
![](https://i.imgur.com/jJy8j3F.jpeg)
Q: N개의 샘플이 있을때, 테스트 시간은 얼마나 걸릴까?
![](https://i.imgur.com/qjMeX0j.jpeg)
Nearest Neighbor의 Classifier는 새로운 테스트 이미지에 label을 할당하려고 할 때, N개의 training set과 모두 비교해야한다.
![](https://i.imgur.com/LOESMJe.jpeg)
우리는 모델을 적용한 application을 배포할 때, train은 길어도 괜찮을지 몰라도, 테스트 시간이 빠르기를 기대하지만 Nearest Neighbor Classifier는 그 점이 부족하다.

![](https://i.imgur.com/iMqKolq.jpeg)
이를 빠르게 계산하는 방안도 존재한다고 한다.


![](https://i.imgur.com/OswLpfd.jpeg)
위의 슬라이드는 CIFAR10 데이터 셋에 대해서 Nearest Neighbor를 적용한 사진이다.
![](https://i.imgur.com/piMz6nN.jpeg)
이 예시를 보면, 2번째와 같은 테스트 이미지에 대해서는 잘 판단했지만, 정답이 아닌 것이 6가지나 되는 것을 볼 수 있다.(4번째 테스트 이미지인 frog를 cat으로 판단했다.)

### Decision Boundary
![](https://i.imgur.com/OlLV3sq.jpeg)
Nearest Neighbor의 decision boundary에 대해서 알아보자.

**decision boundary** : 새로운 sample이 input 되었을 때, 어떤 class로 classification(분류)되는지에 대한 region(영역)간의 경계

![](https://i.imgur.com/zJ1nXb6.jpeg)
2픽셀 이미지를 가정한 시각화이다.
- x축&y축 : 각 픽셀의 강도(intensity; 0~255)
- 점(point): training set의 개별 data
- 점의 색깔 : training set의 label(=class)
- 배경의 색 : 새로운 test image가 들어왔을 때 해당 색깔의 label로 classify할 decision region space

동그라미 친 부분을 잘 살펴보면, boundary가 매우 noisy(들쑥날쑥)하고, 이상치에 대해서 영향을 많이 받는 것을 알 수 있다.(하나의 outlier에 대해서 노란색 noise가 발생하였다.) 

우리는 classifier가 이상치에 최대한 둔감하기를, 그리고
smooth한 boundary를 원한다. 어떻게 이처럼 만들 수 있을까?

-> 1개가 아닌 <font color="#c00000">여러 개의 이웃</font>의 label을 본 뒤, majority vote방법(다수결 방법)을 통해 카테고리를 할당하면 된다. 이때, 이웃의 수를 $k$로 표현하고 k-Nearest Neighbors라고 부른다.

![](https://i.imgur.com/f3ZBrDI.jpeg)
왼쪽 그래프는 가장 가까운 이웃 1개만을 살핀 경우, 오른쪽 그래프는 가장 가까운 이웃 3개를 살핀 경우이다.
![](https://i.imgur.com/O58qINO.jpeg)
더 많은 이웃을 살필 수록 decision boundary가 smooth해진다.
![](https://i.imgur.com/HIOCIwC.jpeg)
또한, 이상치에 대해 둔감해지며, robust한 classifier가 되는 것을 확인할 수 있다.
![](https://i.imgur.com/2hGN3JO.jpeg)
그래프의 흰 부분(tie)는 decision을 내리기 애매한 영역이다. 그래서 어떠한 category로도 예측을 내리기 애매하기 때문에, 이에 대한 후처리가 필요하다.
![](https://i.imgur.com/bE09rLJ.jpeg)
이미지 유사도 비교를 위해 Nearest Neighbor를 사용할 때, 지금까지 사용한 L1외에도 L2 Metric을 사용할 수도 있다.  

![](https://i.imgur.com/sznxxZY.jpeg)
k값이 같더라도, Distance Metric에 따라 decision boundary가 바뀐다. L1에서는 linear한 부분이 상당히 같은 각도(수평, 수직, 45도..등등)로 나오는데, L2 distance의 경우에는 여전히 linear하지만 다른 방향도 가지는 것을 확인할 수 있다.
> 하지만 아직, 두 distance가 어떤 semantic한 차이를 불러오는지에 대한 직관은 명확하지 않다고 한다.


![](https://i.imgur.com/i1H1ivR.jpeg)
distance metric으로 올바른 선택을 한다면, 어떤 종류의 데이터에도 k-NN을 적용할 수 있다.
![](https://i.imgur.com/nnJp1mC.jpeg)
그 예시로, k-NN은 유사 논문 검색의 알고리즘에도 활용되기도 한다. 실제로 arxiv-sanity라는 웹사이트에서 논문에 대한 표절 검사를 진행할 때, k-NN을 활용하였다고 한다. 거리계산식으로는 **tf-idf** distance metric이 사용되었다.
### Hyperparameter
![](https://i.imgur.com/adJ8Kdg.jpeg)
k-NN을 적용할 때에는 데이터로부터 얻을 수 없는 요소가 2가지 있다.
1. 가장 좋은 K의 값
2. 가장 좋은 distance metric(거리 계산식)

이들은 hyperparameter의 예시이다.

**hyperparameter(하이퍼파라미터)** : data로부터 학습할 수 없고, learning process의 시작 파트에서 사람이 설정해주어야하는 값. 학습이 진행됨에 따라서, 어떤 값이 가장 잘 맞는지 손수 찾아봐야한다.

하이퍼파라미터를 어떻게 설정할지는 주어진 데이터에 따라 다르다.
(이를 gold standard가 없다고 표현한다.) 그럼 어떻게 best 하이퍼파라미터를 정할 수 있을까?

이러한 하이퍼 파라미터 세팅에 대한 몇가지 전략이 있다.  

![fig 14. validation set](https://i.imgur.com/4T2WFBY.png)

> 1.training data의 accuracy가 가장 높은 하이퍼 파라미터를 선택하는 것

이러한 접근은 절대 하면 안된다. 
k-nn을 예시로 들었을 때, k=1이라고 하는 것과 같은데, 이 경우에는 knn classifier는 training data에 대해선 100%의 accuracy를 보이기 때문에, 범용 능력을 가지는지 확인할 수 없다.

> 2.training data와 test set 두 요소로 분리하는 것

두번째 아이디어는 training set과 test set 두 요소로 분리하는 것이다. 
첫번째 데이터에 비해 unseen data에 대한 성능을 확인(범용성을 확인)할 수 있지만, hyper parameter settting에서 test set에 대해 적합한 파라미터를 가지게 되며, **이 경우, test set은 더이상 unseen data로 유지되지 않는다.** 즉, 모델의 성능을 평가할 unseen data가 반드시 필요하다.

> 3.training set과 validation set, test set으로 분리하는 것

**<font color="#c00000">세번째 아이디어를 이용해야한다.</font>** 
여기서 evaluation(hyper parameter setting)은 validation set을 통해 진행하고, 이를 통해 하이퍼 파라미터를 업데이트 하면서 test set에 대한 unseen 상태를 유지해야한다. test set은 최종단계에서 오직 1번만 실행한다.

![fig 15. cross-validation se](https://i.imgur.com/4f1sBzW.png)
Cross-Validation은 데이터 셋을 fold라 불리는 덩어리(chunk)로 나누고, 나눠진 fold가 5개라면 5번의 iteration을 돌면서 각 iteration마다 validation set으로 이용하는 fold를 변경하는 것이다.(이 경우에도 test set은 건들면 안된다.) 이 방법은 이론상 제일 좋은 방법이지만, 연산량이 매우 크기 때문에 적용하기 힘든 경우가 많다.

![](https://i.imgur.com/qTkubE7.jpeg)
각 k값에 따른 cross-validation accuracy를 평가한 그래프로, k=7에서 주어진 데이터에 가장 잘 작동하는것을 보인다.  

### Universal Approximation
![](https://i.imgur.com/IEzC4Al.jpeg)
Nearest Neighbor 함수는 Universal Approximation하다는 특징을 가진다.
- **Universal Approximation** : 임의의 연속인 다변수 함수를 원하는 정도의 정확도로 근사할 수 있다

training sample의 수가 무한대로 많아지면, k-NN은 어떠한 함수도 표현할 수 있게 된다.
![](https://i.imgur.com/HradDD1.jpeg)
 feature의 개수가 1개인 상황을 가정해서 시각화해보자.
![](https://i.imgur.com/p3zRN6x.jpeg)

![](https://i.imgur.com/ItgUlUN.jpeg)

![](https://i.imgur.com/uNtX8YL.jpeg)
Training point가 늘어남(=샘플사이즈가 늘어나고 있다)에 따라서 True function에 더 잘 approximate하는 것을 볼 수 있다.
### Curse of Dimensionality
![](https://i.imgur.com/Ax089SP.jpeg)
단, k-NN에는 Curse of Dimensionality(차원의 저주)라는 또 다른 문제가 존재한다.

수학적 공간 차원(= feature 변수의 개수)이 늘어날 때, dimension을 같은 밀도(coverage)를 유지하기 위해서는 training sample의 수가 지수적으로 더 필요해진다는 내용이다.

![](https://i.imgur.com/3Rpuk1Q.jpeg)
32x32 크기에 binary값을 채우려고 할 때, 총 $2^{32 \times 32}$개의 수가 필요하다. 즉, Universal approximation theory를 만족시킬 정도로 데이터를 많이 뽑아내기 위해서는 천문학적으로 많은 수가 필요해지는 것이다.

![](https://i.imgur.com/Snb10in.jpeg)
즉, k-NN의 단점은 아래와 같이 정리할 수 있다.
1. test time이 느리다
2. distance metric이 이미지의 의미론(사람이 보는 high level처럼 보는 것)에 부합하지 않는다.
3. feature space를 채울 데이터가 너무나 많이 필요해지는 차원의 저주

위의 사진을 보면, Original 이미지와 변형을 가한 이미지 모두 L2 Distance가 같은 값을 보이며, 좋은 distance metric은 shifted image가 다른 이미지들보다 Original과 거리가 가깝다고 말해야한다.
![](https://i.imgur.com/HMrUEiA.jpeg)
![](https://i.imgur.com/WWs4Wxn.jpeg)
ConvNet을 통해 구한 feature를 이용해 k-NN을 수행하면 비교적 유사한 이미지를 잘 찾을 수 있다고 한다. raw pixel에는 k-NN이 잘 이용되지 않는다.

![](https://i.imgur.com/ffjMSjC.jpeg)

![](https://i.imgur.com/W8E0Xik.jpeg)



## Ref

- <https://ploradoaa.tistory.com/76>
- <https://sites.google.com/view/statml-smwu-2020s>