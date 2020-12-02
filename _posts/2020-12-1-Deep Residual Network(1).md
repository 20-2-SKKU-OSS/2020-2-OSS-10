---
layout: post
title: Deep Residual Network (1)
subtitle: Deep Residual Network의 개념
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---

이 포스트에서는 Deep Residual Network의 개념에 대해 Deep Residual Learning for Image Recognition 논문의 내용을 간단하게 정리한다.

![resnet1](https://github.com/20-2-SKKU-OSS/2020-2-OSS-10/blob/KyungheeKo-intermediate/assets/img/ResNet/resnet1.png?raw=true)

위 그래프는 CIFAR-10 데이터를 각각 20-layer와 56-layer의 plain network를 이용해 분류했을 때의 Training error와 test error를 나타낸 것이다. 이 그래프에서도 알 수 있듯이 단순한 형태의 Neural Network는 깊어지면 깊어질수록 어느 시점까지는 성능이 개선되지만, 이후로는 오히려 정확도가 떨어지는 결과를 보인다.

과적합(Overfitting) 문제라고 생각할 수도 있겠지만 이러한 현상이 일어나는 다른 원인이 있다. 바로 degradation problem이다. 이러한 문제가 발생하는 이유는 network가 깊어질 수록 weight의 분포가 균등하지 않게 되며, 역전파가 제대로 이루어지지 않는 vanishing/exploding gradient 가 발생하기 때문이다. 이로 인해 deep neural network에서는 오히려 정확도가 떨어지는 문제가 생긴다.

이 문제를 해결하기 위해 제안된 모델이 Residual Network 이다.

![resnet2](https://github.com/20-2-SKKU-OSS/2020-2-OSS-10/blob/KyungheeKo-intermediate/assets/img/ResNet/resnet2.png?raw=true)

Residual Network 의 기본 구조는 위 그림과 같다. 일반적인 network에서 input이 x, 얻을 수 있는 결과를 H(x)라고 할 때, F(x) = H(x) - x 의 형태로 network를 변형해 학습시켜 F(x) + x 가 H(x)에 근사하도록 하는 것이다. x를 더하는 것은 shortcut을 사용하기 때문에 복잡도 면에서는 큰 차이가 없다. 그러나 F(x) + x 를 사용하면 역전파 과정에서 미분을 하더라도 최소 기울기가 1이 되기 때문에 vanishing gradient 문제를 해결하면서 보다 안정적인 학습이 가능해진다.

![resnet3](https://github.com/20-2-SKKU-OSS/2020-2-OSS-10/blob/KyungheeKo-intermediate/assets/img/ResNet/resnet3.png?raw=true)

이 논문에서는 위와 같은 구조로 Residual Network를 설계했다. 그리고 이 구조를 적용해 학습을 진행한 결과 error 그래프는 다음과 같이 그려진다.

![resnet4](https://github.com/20-2-SKKU-OSS/2020-2-OSS-10/blob/KyungheeKo-intermediate/assets/img/ResNet/resnet4.png?raw=true)

왼쪽과 오른쪽 그래프를 비교해보면, 왼쪽의 Plain network는 깊이가 깊은 네트워크의 정확도가 더 낮게 나타나는 경향을 보이지만 Residual network는 같은 깊이에서도 네트워크의 정확도가 향상된 결과를 보이는 것을 알 수 있다.

출처 : [Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, 770-778](https://arxiv.org/pdf/1512.03385.pdf)
