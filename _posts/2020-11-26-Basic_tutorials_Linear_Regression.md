---
layout: post
title: Basic_Tutorial(2)
subtitle: Linear Regression
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---

### 1. What is Linear Regression(LR)?

Linear Regression(선형회귀) : 선형회귀는 종속변수 y와 한 개 이상의 독립변수 x와의 선형 상관 관계를 모델링 하는 회기분석하는 기법이다. 한 개의 설명 변수에 기바한 경우에는 단순 선형 회귀, 둘 이상의 설명 변수에 기반한 경우에는 다중 선형 회귀라고 한다.<br>
선형 회귀는 선형 예측 함수를 사용해 회귀식을 모델링하며, 알려지지 않은 파라미터는 데이터로부터 추정한다. 이렇게 만들어진 회귀식을 선형 모델이라고 한다.
선형 회귀는 깊이있게 연구되고 널리 사용된 첫 번째 회귀분석 기법이다. 이는 알려지지 않은 파라미터에 대해 선형 관계를 갖는 모델을 세우는 것이, 비선형 관계를 갖는 모델을 세우는 것보다 용이하기 때문이다.<br>
일반적으로 최소제곱법(least square method)을 사용해 선형 회귀 모델을 세운다. 최소제곱법 외에 다른 기법으로도 선형 회귀 모델을 세울 수 있다. 손실 함수(loss fuction)를 최소화 하는 방식으로 선형 회귀 모델을 세울 수도 있다. 최소제곱법은 선형 회귀 모델 뿐 아니라, 비선형 회귀 모델에도 적용할 수 있다. 최소제곱법과 선형 회귀는 가깝게 연관되어 있지만, 그렇다고 해서 동의어는 아니다.<br>
<img src="/assets/img/Linear Regression/LR_1.png" width="700" height="500" align="left">    


### 2. Examples of Linear Regression
 
- 값을 예측하는 것이 목적일 경우, 선형 회귀를 사용해 데이터에 적합한 예측 모형을 개발한다. 개발한 선형 회귀식을 사용해 y가 없는 x값에 대해 y를 예측하기 위해 사용할 수 있다.
- 종속 변수 y와 이것과 연관된 독립 변수 X1, ..., Xp가 존재하는 경우에, 선형 회귀 분석을 사용해 Xj와 y의 관계를 정량화할 수 있다. Xj는 y와 전혀 관계가 없을 수도 있고, 추가적인 정보를 제공하는 변수일 수도 있다.

### 3. Implementation of Linear regression

[Linear Regression](https://github.com/20-2-SKKU-OSS/2020-2-OSS-10/tree/main/tutorials/01-basics/linear_regression){:target="_blank"}
