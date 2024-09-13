---
layout: single
title: "Intro & What is Machine Learning"
categories: Machine_Learning
tag: Machine Learning Specialization
---

> Coursera의 Machine Learning Specialization강의를 정리한 내용입니다.

## Intro

필자는 NLP에 대해 연구할 수 있는 기본기를 쌓는 것을 이 코스의 최종 목표로 가진다.

NLP는 Deep Learning의 한 분야이기에, 기본이 되는 Machine Learning에 대해서 먼저 공부할 것이다.

<br>

우리는 자신도 모르는 사이 머신러닝을 사용하고 있다.

구글을 통해 검색을 할 때 검색엔진들이 검색을 적절하게 하기 위해 웹 페이지에 순위를 매기는 것,

Instagram이나 facebook에 사진을 업로드 할 때 사진 속 친구들을 알아보고 Labeling하는 것,

유튜브의 추천서비스, 이메일에서의 스팸 차단 역시 머신러닝의 응용 분야이다.

<br>

이 수업에서는 최신 기술에 대해 배우고 기계 학습 알고리즘을 직접 구현하는 연습을 해볼 수 있다.

현재나 가까운 미래에 기계학습이 크게 활용될 것 같지 않은 업종을 생각하기는 어려우며, 많은 사람들이 사람처럼 지능적인 기계를 만들겠다는 꿈을 가지고 있다. 이를 **인공 일반 지능** 또는 **AGI**라고 한다.

>  AGI : 인간이 할 수 있는 어떠한 지적인 업무도 성공적으로 해낼 수 있는 기계의 지능

<br>

현재까지 창출된 가치 이외에도, 기계학습은 아직 창출되지 않은 훨씬 더 큰 가치를 지니고 있을 것이다.

즉, 머신러닝이 필요한 분야는 점차 확산되고 있다.

------

## What is Machine Learning?

Arthur Samuel(아서 사무엘)은 머신러닝을 아래와 같이 정의했다.

> "Field of study that gives computers the ability to learn without being explicitly programmed"
>
> "명시적으로 프로그래밍 하지 않고도 컴퓨터가 학습할 수 있는 능력을 부여하는 연구 분야"

<br>

아서 사무엘은 컴퓨터가 자기 자신을 상대로 수만개의 체커 게임을 플레이하도록 컴퓨터를 프로그래밍 하였는데, 어떤 포지션에서 이기고 어떤 포지션에서 패배하는지를 관찰하면서 어떤 포지션이 좋은지 나쁜지를 알게 되었다.

> Quiz.
>
> 만약 체커 프로그램이 10개의 게임만을 학습하는 경우, 1000개의 게임을 학습한 프로그램과 비교해서 성능은 어떻게 변화할까?
> => 일반적으로 **학습 알고리즘을 학습할 기회가 많을수록 성능이 더 좋아진다.**

아서 사무엘의 정의는 다소 비형식적인 것이었으며, 다음 챕터에서는 기계학습 알고리즘의 주요 유형이 무엇인지 더 자세하게 알아본다.

------

기계학습의 두 가지 주요 알고리즘은 Supervised learning(지도학습)과 Unsupervised learning(비지도학습)이다.

Supervised learning은 작업을 수행하는 방법을 **사용자가 학습시키는 것**이고, Unsupervised learning은 **컴퓨터가 스스로 학습하도록 유도하**는 것이다.

>  **Supervised** learning은 실제 응용 분야에서 가장 많이 사용되는 기계학습 유형으로 가장 빠른 발전과 혁신을 이루었다.

<br>

또한, 머신러닝 알고리즘의 다른 종류로는 Reinforcement learning(강화학습), Recommender systems(추천시스템)이 있다.