### This example shows that if the train data set are randomly choosed number, how linear regression occurred. ###

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset
x_train = np.array([[0.44], [0.695], [0.954], [0.254], [0.198], [0.173], [0.226], [0.091], [0.408], [0.274], [0.487], [0.921], [0.233], [0.031], [0.585], [0.598], [0.162], [0.223], [0.062], [0.453]], dtype=np.float32)

y_train = np.array([[0.217], [0.302], [0.532], [0.839], [0.097], [0.466], [0.786], [0.821], [0.248], [0.418], [0.298], [0.214], [0.871], [0.0], [0.057], [0.676], [0.312], [0.408], [0.338], [0.428]], dtype=np.float32)

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
