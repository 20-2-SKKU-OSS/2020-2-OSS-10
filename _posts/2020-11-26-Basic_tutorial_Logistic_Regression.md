---
layout: post
title: Basic_Tutorial(3)
subtitle: Logisitc Regression
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---

### 1. What is Logistic Regression(LR)?

로지스틱 회귀는 독립 변수의 선형결합을 이용하여 사건의 발생 가능성을 예측하는데 사용되는 통계 기법이다.<br>
로지스틱 회귀의 목적은 일반적인 회귀 분석의 목표와 동일하게 종속 변수와 독립 변수간의 관계를 구체적인 함수로 나타내어 향후 예측 모델에 사용하는 것이다. 이는 독립 변수의 선형 결합으로 종속 변수를 설명한다는 관점에서는 선형 회귀 분석과 유사하다. 하지만 로지스틱 회귀는 선형 회귀 분석과는 다르게 종속 변수가 범주형 데이터를 대상으로 하며 입력 데이터가 주어졌을 때 해당 데이터의 결과가 특정 분류로 나뉘기 때문에 일종의 분류 (classification) 기법으로도 볼 수 있다.

<img src="/assets/img/Logistic%20Regression/LoR_1.jpeg" width="700" height="500" align="left">    



### 2. Logistic Function

- 오즈(odds)  
  성공할 확률이 실패 확률에 비해 몇 배 더 높은가를 나타내며 그 식은 아래와 같다.  
<p align="center"><img src="/assets/img/Logistic%20Regression/odds.png" width="250" height="150"></p> 



- 로짓변환  
  오즈에 로그를 취한 함수로서 입력 값의 범위가 [0,1] 일 때 출력 값의 범위를 (-inf,inf)로 조정한다.  
<p align="center"><img src="/assets/img/Logistic%20Regression/logit.png" width="250" height="150"></p>
 
 
- 로지스틱 함수(Logistic Function)  
  로지스틱 함수의 그래프는 독립 변수 x가 주어졌을 때 종속 변수가 1의 범주에 속할 확률을 의미한다. 즉, p(y=1|x)를 의미한다.  
  로지스틱 함수는 로짓 변환을 통해 만들어지고, 그 형태는 다음과 같다.
  
  
<p align="center"><img src="/assets/img/Logistic%20Regression/logistic_fn.png" width="350" height="150"></p>


### 3. Implementation of Logistic regression

[Logistic Regression](https://github.com/20-2-SKKU-OSS/2020-2-OSS-10/tree/main/tutorials/01-basics/logistic_regression){:target="_blank"}
