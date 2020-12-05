---
layout: post
title: Intermediate Tutorial(BRNN) - (2)
subtitle: Implementation
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---


이 포스트에서는 Bidirectional Recurrent Nerual Network를 적용한 [Pytorch Tutorial](https://github.com/yunjey/pytorch-tutorial.git) 코드를 분석해본다. ([코드 원본](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py))

[Bidirectional Recurrent Nerual Network 에서 정리한 내용](https://joqjoq966.github.io/2020/11/30/BRNNs.html)을 바탕으로 구현한 코드이다. 또한 일부는 [Pytorch/vision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) 에서 제공하는 코드를 참조했다.

아래부터는 코드블럭과 그 아래 간단한 설명을 추가하는 형태로 분석하도록 한다.

---

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
```
가장 먼저 필요한 package 들을 불러옵니다. 이전 다른 코드에도 사용되었던, pytorch를 사용해 학습을 진행할 때에 가장 기본적으로 사용되는 package 들이므로 별도의 설명은 생략하도록 하겠습니다.  
<br>
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
학습에 사용할 device를 정의합니다. gpu를 사용할 수 있다면 cuda, 그렇지 않은 경우에는 cpu로 정의합니다. 이 과정을 거치는 것은 나중에 데이터와 모델을 device에 할당해주는 과정에서 매번 코드를 cuda 와 cpu로 변경하지 않고도 사용할 수 있게 하기 위함이다.  
  <br>
```python
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003
```
먼저, Hyper parameter들을 설정합니다.   
sequence_length : 원하는 sequence의 길이에 해당하는 수를 입력합니다.    
input_size : Input의 사이즈에 해당하는 수를 입력합니다.     
hidden_size : Hidden layer의 사이즈에 해당하는 수를 입력합니다.   
다른 parameter들도 원하는 값으로 입력합니다.   

  
```python
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())


```
학습과 모델 성능 평가에 사용할 이미지 데이터인 MNIST의 Data set을 불러옵니다.(Train dataset, Test dataset)  
  <br>
```python
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
```
불러온 데이터를 torch.utils.data.DataLoader를 이용해 배치 단위로 잘라서 순서대로 반환하도록 해줍니다. shuffle 값을 이용해 매 epoch마다 순서가 랜덤으로 바뀌도록 하면 학습에 도움이 됩니다.  
  <br>
```python
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
먼저, BiRNN class를 정의합니다. 그리고 model변수를 이용하여 Train할 클래스(데이터)를 만들어줍니다.  
또한, criterion 과 optimizer를 정의합니다. 사용되는 함수에는 여러가지가 있으나, 여기서는 CrossEntropyLoss 와 Adam을 사용하기로 한다.  
  <br>
```python
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
```
모델 학습을 진행합니다.  
  <br>
```python
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))  
```
학습된 모델을 평가합니다. 즉, 학습된 결과를 일반적인 경우에도 적용할 수 있는지 알아보기 위해 test 데이터를 이용해 최종 검증을 진행한다. 이 때에는 parameter를 갱신할 필요가 없는 test 데이터 이기 때문에 model.eval()과 with torch.no_grad()를 사용해준다. 검증을 마치면 최종적으로 정확도를 출력한다.  
<br>
```python
torch.save(model.state_dict(), 'resnet.ckpt')
```
결과가 만족스럽다면, 현재 상태를 저장할 수 있다. torch.save 함수를 이용해 state를 저장해둔다. 후에 다른 파일에서도 이 학습된 모델의 상태를 불러와서 다른 데이터를 분류하는 데에 사용할 수 있다.  
<br>

---

출처  

[Pytorch Tutorial](https://github.com/yunjey/pytorch-tutorial.git)  
[Pytorch Tutorial - Bidirectional Recurrent Nerual Network](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py)  
[Pytorch/vision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)  
[Pytorch Docs - torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html)
