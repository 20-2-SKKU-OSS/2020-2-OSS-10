---
layout: post
title: torchvision.datasets.MNIST
subtitle: MNIST Dataset 개요
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---

the MNIST(Modified National Institute of Standards and Technology) database of hadwritten digits 는 Yann LeCun(Courant Institute, NYU), Corinna Cortes(Google Labs, New York), Christopher J.C. Burges(Microsoft Research, Redmond) 가 제공하는 학습용 데이터이다. NIST 데이터의 하위 분류에 속하며, 손으로 쓴 숫자 이미지의 중심을 맞추고 같은 크기로 가공하는 후처리를 거친 상태로 제공된다.

데이터는 벡터(vector)와 다차원행렬(multidimensional matrix)의 형태로 제공된다.
다음과 같은 총 4개의 파일로 구성되어 있다.

- train-images-idx3-ubyte: training set images
- train-labels-idx1-ubyte: training set labels
- t10k-images-idx3-ubyte:  test set images
- t10k-labels-idx1-ubyte:  test set labels

총 60000개의 training data와 10000개의 test data로 구성되어 있다.
label의 값은 0-9 범위의 숫자이다.
각각의 image는 28×28 행렬이며, 각 항의 값은 0-255  숫자이다. 0이 흰색(background), 255가 검은색(foreground)을 의미한다. 흑백 데이터이므로 다른 색으로는 표현되지 않는다.

출처 : [MNIST](http://yann.lecun.com/exdb/mnist/)
