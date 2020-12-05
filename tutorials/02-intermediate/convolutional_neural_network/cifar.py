import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


transform_train= transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=True,
                                        download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./cifar10_data', train=False, 
                           download=True, transform=transform_test)

validation_ratio=0.1
random_seed=17

num_train = len(trainset)
indices = list(range(num_train))
split=int(np.floor(validation_ratio * num_train))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
                                  sampler=train_sampler, num_workers=2)

validloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
                                  sampler=valid_sampler, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=128, 
                                  shuffle = False, num_workers=2)


classes=('plane','car','bird','cat',
         'deer','dog','frog','horse','ship','truck')

class Net(nn.Module):
  
  def __init__(self):
    super(Net, self).__init__()
    self.conv1=nn.Conv2d(3,6,5)
    self.bn1=nn.BatchNorm2d(6)
    self.pool=nn.MaxPool2d(2,2)
    self.conv2=nn.Conv2d(6,16,5)
    self.bn2=nn.BatchNorm2d(16)
    

    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84,64)
    self.fc4 = nn.Linear(64,42)
    self.fc5 = nn.Linear(42,10)

    self.bn3=nn.BatchNorm1d(120)
    self.bn4=nn.BatchNorm1d(84)
    self.bn5=nn.BatchNorm1d(64)
    self.bn6=nn.BatchNorm1d(42)
   
  

  def forward(self, x):
    x=self.pool(F.relu(self.bn1(self.conv1(x))))
    x=self.pool(F.relu(self.bn2(self.conv2(x))))
    x = x.view(-1, 16*5*5)
    x = F.relu(self.bn3(self.fc1(x)))
    x = F.relu(self.bn4(self.fc2(x)))
    x = F.relu(self.bn5(self.fc3(x)))
    x = F.relu(self.bn6(self.fc4(x)))
    x = self.fc5(x)
    return x

device=torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net=Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.0001)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

best_valid_acc=0
for epoch in range(350):  # loop over the dataset multiple times
  
        net.train()
        exp_lr_scheduler.step()     
        running_loss = 0
        total = 0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
   
    
            # print statistics
            running_loss += loss.item()
    
            # for train acc
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
      
    
        ### train loss/acc ###\n",
        print('[%d] train loss: %.3f %%, lr:%f' % 
             (epoch + 1, running_loss / 50000, optimizer.param_groups[0]['lr']))
        
        net.eval()
        ### valid acc ###\
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        valid_acc= 100.*correct/total
        if best_valid_acc<valid_acc:
            best_valid_acc= valid_acc
            torch.save(net.state_dict(), "./save_best.pth")
        print('[%d] valid_acc: %.3f %%, best: %.3f %%' %(epoch + 1, valid_acc, best_valid_acc))
        
    
print('Finished Training')

load_model=Net().to(device)
load_model.load_state_dict(torch.load("./save_best.pth"))

load_model.eval()

correct = 0
total = 0

# test part
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = load_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_acc=100.*correct/total        

print('test_acc: %.3f %%' %(test_acc))

