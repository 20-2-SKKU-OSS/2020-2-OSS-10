---
layout: post
title: Deep Residual Network (2)
subtitle: Deep Residual Network의 적용
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---
이 포스트에서는 Deep Residual Network를 적용한 [Pytorch Tutorial](https://github.com/yunjey/pytorch-tutorial.git) 코드를 분석해본다. ([코드 원본](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py))

[Deep Residual Network (1) 에서 정리한 논문](https://arxiv.org/pdf/1512.03385.pdf)의 4.2 항, CIFAR-10 에 대한 모델 구조를 구현한 코드이다. 또한 일부는 [Pytorch/vision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) 에서 제공하는 코드를 참조했다고 한다.

아래부터는 코드블럭과 그 아래 간단한 설명을 다는 형태로 분석한다.

---

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
```
가장 먼저 필요한 package 들을 불러온다. 이전 다른 코드에도 사용되었던, pytorch를 사용해 학습을 진행할 때에 가장 기본적으로 사용되는 package 들이므로 별도의 설명은 생략한다.  
<br>
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
학습에 사용할 device를 정의한다. gpu를 사용할 수 있다면 cuda, 그렇지 않은 경우에는 cpu로 정의한다. 이 과정을 거치는 것은 나중에 데이터와 모델을 device에 할당해주는 과정에서 매번 코드를 cuda 와 cpu로 변경하지 않고도 사용할 수 있게 하기 위함이다.  
  <br>
```python
num_epochs = 80
batch_size = 100
learning_rate = 0.001
```
학습할 epoch의 횟수, 데이터를 나누는 batch의 크기, learning rate 의 값을 설정해준다. 얼마든지 필요에 의해 변경해서 사용할 수 있는 변수들이다. 이 역시 변수를 미리 설정해 매번 번거롭게 숫자로 변경하는 작업을 하지 않도록 하기 위함이다.  
  
```python
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])
```
학습에 사용될 이미지를 전처리하는 모듈을 미리 정의해준다.
torchvision.transforms.Compose(transforms) : Transform objects list를 입력으로 받아 이들을 묶어서 순차적으로 실행해주는 함수이다. 단, torchscript를 지원하지 않으므로 필요한 경우에는 같은 기능을 하는 torch.nn.Sequential 을 사용하도록 하자.  
torchvision.transforms.Pad(padding, fill=0, padding_mode='constant') : 이미지의 상하좌우에 입력한 설정값만큼 padding을 삽입해주는 함수이다.  
torchvision.transforms.RandomHorizontalFlip(p=0.5) : 입력된 확률 값만큼 random하게 이미지를 좌우반전 시키는 함수이다.  
torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant') : 입력된 사이즈에 맞춰 이미지를 random하게 잘라주는 함수이다.  
torchvision.transforms.ToTensor : 최종 처리된 데이터를 Tensor 데이터로 변환해주는 함수이다.  
  <br>
```python
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                            train=False, 
                                            transform=transforms.ToTensor())
```
학습과 모델 성능 평가에 사용할 이미지 데이터인 CIFAR-10 데이터셋을 각각 train과 test로 나누어 불러온다. train dataset은 위에서 정의했던 전처리 모듈을 사용하고, test는 ToTensor를 이용해 Tensor 데이터로 변환만 해준다.  
  <br>
```python
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
```
불러온 데이터를 torch.utils.data.DataLoader를 이용해 배치 단위로 잘라서 순서대로 반환하도록 해준다. shuffle 값을 이용해 매 epoch마다 순서가 랜덤으로 바뀌도록 하면 학습에 도움이 된다.  
  <br>
```python
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
```
3x3 convolution layer를 정의한다.  
  <br>
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
```
Residual Block을 정의한다. 한 블럭에는 3x3 convolution layer, batch normalization, ReLU가 각각 2회씩 포함되어 있으며, 마지막에 residual을 더해주는 것을 잊지 않아야 한다.  
  <br>
```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
```
이제 위에서 정의한 블록을 이용해 우리가 사용할 ResNet 모델을 설계한다. 모델의 구조는 참조한 논문에 언급된 형태이다.  
  <br>
```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
criterion 과 optimizer를 정의한다. 사용되는 함수에는 여러가지가 있으나, 여기서는 CrossEntropyLoss 와 Adam을 사용하기로 한다.  
<br>
```python
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```
learning rate를 업데이트 하기 위한 함수를 정의한다.  
<br>
```python
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
```
모든 모델의 정의가 끝났으므로 이제 학습을 시작한다. epoch 수만큼 전체 데이터를 입력하고 전파, 역전파, 최적화 과정을 거듭한다. 학습이 어떻게 진행되고 있는지 현황을 확인하기 위해 일정 시점마다 loss를 출력해준다. 학습이 성공적으로 이루어지고 있다면 loss가 계속해서 감소하는 것을 확인할 수 있을 것이다.  
  <br>
```python
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
```
학습된 결과를 일반적인 경우에도 적용할 수 있는지 알아보기 위해 test 데이터를 이용해 최종 검증을 진행한다. 이 때에는 parameter를 갱신할 필요가 없는 test 데이터 이기 때문에 model.eval()과 with torch.no_grad()를 사용해준다. 검증을 마치면 최종적으로 정확도를 출력한다.  
<br>
```python
torch.save(model.state_dict(), 'resnet.ckpt')
```
결과가 만족스럽다면, 현재 상태를 저장할 수 있다. torch.save 함수를 이용해 state를 저장해둔다. 후에 다른 파일에서도 이 학습된 모델의 상태를 불러와서 다른 데이터를 분류하는 데에 사용할 수 있다.  
<br>

---

출처  

[Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, 770-778](https://arxiv.org/pdf/1512.03385.pdf)  
[Pytorch Tutorial](https://github.com/yunjey/pytorch-tutorial.git)  
[Pytorch Tutorial - Deep Residual Network](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py)  
[Pytorch/vision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)  
[Pytorch Docs - torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html)  
