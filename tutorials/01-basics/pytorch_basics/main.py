import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 26 to 40)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189) 
# 8. Create tensor                          (Line 195 to 200)
# 9. Basic numpy operations                 (Line 195 to 200)
# 10. Visualize
# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# Create tensors.
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# Build a computational graph.
y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward()

# Print out the gradients.
print(x.grad)    # x.grad = 2 
print(w.grad)    # w.grad = 1 
print(b.grad)    # b.grad = 1 


# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #

# Create tensors of shape (10, 3) and (10, 2).
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Build a fully connected layer.
linear = nn.Linear(3, 2)
print ('w: ', linear.weight)
print ('b: ', linear.bias)

# Build loss function and optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass.
pred = linear(x)

# Compute loss.
loss = criterion(pred, y)
print('loss: ', loss.item())

# Backward pass.
loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad) 
print ('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
optimizer.step()

# You can also perform gradient descent at the low level.
# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

# Create a numpy array.
x = np.array([[1, 2], [3, 4]])

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
z = y.numpy()


# ================================================================== #
#                         4. Input pipeline                           #
# ================================================================== #

# Download and construct CIFAR-10 dataset.
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True, 
                                             transform=transforms.ToTensor(),
                                             download=True)

# Fetch one data pair (read data from disk).
image, label = train_dataset[0]
print (image.size())
print (label)

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, 
                                           shuffle=True)

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = data_iter.next()

# Actual usage of the data loader is as below.
for images, labels in train_loader:
    # Training code should be written here.
    pass


# ================================================================== #
#                5. Input pipeline for custom dataset                 #
# ================================================================== #

# You should build your custom dataset as below.
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0 

# You can then use the prebuilt data loader. 
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64, 
                                           shuffle=True)


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size())     # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))

# ================================================================== #
#                      8. Tensor                                     #
# ================================================================== #

# Create tensors (t1 and t2 are same)
t1 = torch.FloatTensor([0, 1, 2, 3, 4, 5, 6])
t2 = torch.tensor(np.arange(7))

# Print shape of tensors 
print(t1.shape, t2.shape)

# Broadcasting
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3])
print(m1 + m2)

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

# Mul vs MatMul
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # result 2 x 2
print('Shape of Matrix 2: ', m2.shape) # result 2 x 1
print(m1.matmul(m2)) # 2 x 1

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2
print(m1.mul(m2))

# Mean
t = torch.FloatTensor([[1, 2], [3, 4]]) 
print(t.type())
print(t)

print(t.mean())
print(t.mean(dim=0)) # Remove Dim 0
print(t.mean(dim=1))
print(t.mean(dim=-1))

# View
t = np.arange(12).reshape(-1, 2, 3)
print(t.shape)

floatT = torch.FloatTensor(t)
print(floatT.shape)

print(floatT.view([-1, 3]))

# ================================================================== #
#                      9. Basic numpy operations                     #
# ================================================================== #

# Create array with only '0' elements
a = np.zeros((2,2))
print(a)          # result "[[ 0.  0.]
                  #          [ 0.  0.]]"

# Create array with only '1' elements
b = np.ones((1,2))
print(b)          # result "[[ 1.  1.]
                  


# Fill array with specific number (in this case, 7)
c = np.full((2,2), 7)
print(c)          # result "[[ 7.  7.]
                  #          [ 7.  7.]]"


# Create 2x2 unit matrix
d = np.eye(2)
print(d)          # result "[[ 1.  0.]
                  #          [ 0.  1.]]"


# Create array filled with arbitrary values
e = np.random.random((2,2))
print(e)          # result "[[ 0.91940167  0.08143941]
                  #          [ 0.68744134  0.87236687]]"
                  # arbitrary values are filled

# Numpy array slicing    
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]
print(a[0,1]) # result "2"
b[0, 0] = 77 # b[0,0] and a[0,1] are same
print(a[0,1]) # result "77"
# Reshaping method
b = np.reshape(a, (2, 6))
print(b[0, 0]) # result "2"
b[0, 0] = 77
print(a[0, 0]) # result "77"

# ================================================================== #
#                      10. Visualize                                 #
# ================================================================== #

def vis_data(x,y = None, c = 'r'):
  if y is None:
    y = [None] * len(x)
  for x_, y_ in zip(x,y):
     if y_ is None:
        plt.plot(x_[0], x_[1], '*', markerfacecolor='none', markeredgecolor=c)
     else:
        plt.plot(x_[0], x_[1], c+'o' if y_ ==0 else c+'+')
plt.figure()
vis_data(x_train, y_train, c='r')
plt.show()
        
