---
layout: post
title: torchvision.datasets.CIFAR
subtitle: CIFAR Dataset 개요
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---

CIFAR(Canadian Institute For Advanced Research) dataset은 Alex Krizhevsky, Vinod Nair, Geoffrey Hinton 이 제공하는 학습용 데이터이다. [80 million tiny images dataset](http://groups.csail.mit.edu/vision/TinyImages/) 의 하위 분류에 속하며, CIFAR-10과 CIFAR-100은 각각 10개, 100개의 범주로 이미지를 구분해 제공한다.

각각의 이미지는 32×32 컬러 이미지로, 0-255의 RGB값을 각각 분리해 총 3개의 2차원 행렬로 표현된다. 이미지 총 개수는 60000개로 50000개의 train data 와 10000개의 test data로 구성되어 있다.

### CIFAR-10
분류 항목은 다음과 같으며, 각 항목마다 6000개의 이미지를 포함한다.
- airplane										
- automobile										
- bird										
- cat										
- deer										
- dog										
- frog										
- horse										
- ship										
- truck

### CIFAR-100
분류 항목은 다음과 같으며, 각 대분류마다 3000개, 소분류마다 600개의 이미지를 포함한다.
- aquatic mammals
  - beaver
  - dolphin
  - otter
  - seal
  - whale
- fish
  - aquarium fish
  - flatfish
  - ray
  - shark
  - trout
- flowers
  - orchids
  - poppies
  - roses
  - sunflowers
  - tulips
- food containers
  - bottles
  - bowls
  - cans
  - cups
  - plates
- fruit and vegetables
  - apples
  - mushrooms
  - oranges
  - pears
  - sweet peppers
- household electrical devices
  - clock
  - computer keyboard
  - lamp
  - telephone
  - television
- household furniture
  - bed
  - chair
  - couch
  - table
  - wardrobe
- insects
  - bee
  - beetle
  - butterfly
  - caterpillar
  - cockroach
- large carnivores
  - bear
  - leopard
  - lion
  - tiger
  - wolf
- large man-made outdoor things
  - bridge
  - castle
  - house
  - road
  - skyscraper
- large natural outdoor scenes
  - cloud
  - forest
  - mountain
  - plain
  - sea
- large omnivores and herbivores
  - camel
  - cattle
  - chimpanzee
  - elephant
  - kangaroo
- medium-sized mammals
  - fox
  - porcupine
  - possum
  - raccoon
  - skunk
- non-insect invertebrates
  - crab
  - lobster
  - snail
  - spider
  - worm
- people
  - baby
  - boy
  - girl
  - man
  - woman
- reptiles
  - crocodile
  - dinosaur
  - lizard
  - snake
  - turtle
- small mammals
  - hamster
  - mouse
  - rabbit
  - shrew
  - squirrel
- trees
  - maple
  - oak
  - palm
  - pine
  - willow
- vehicles 1
  - bicycle
  - bus
  - motorcycle
  - pickup truck
  - train
- vehicles 2
  - lawn-mower
  - rocket
  - streetcar
  - tank
  - tractor

출처 : [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)
