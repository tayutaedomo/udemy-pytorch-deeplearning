import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline


data = pd.read_csv('climate/data.csv', skiprows=[0, 1, 2, 4], encoding='shift-jis')

temp_data = data['平均気温(℃)']
#temp_data
#data[1820:1830]

train_x = temp_data[:1826]
test_x = temp_data[1826:]
train_x = np.array(train_x)
test_x = np.array(test_x)

#train_x
#len(train_x)
#test_x
#len(test_x)


window_size = 180
tmp = []
train_X = []

for i in range(0, len(train_x) - window_size):
    tmp.append(train_x[i:i+window_size])

train_X = np.array(tmp)

#pd.DataFrame(train_X)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(180, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 180)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = Net()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    total_loss = 0
    input_x = []

    for i in range(100):
        index = np.random.randint(0, 1645)
        input_x.append(train_X[index])

    input_x = np.array(input_x, dtype="float32")
    input_x = Variable(torch.from_numpy(input_x))

    optimizer.zero_grad()
    output = model(input_x)

    loss = criterion(output, input_x)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    if (epoch+1) % 100 == 0:
        print(epoch+1, total_loss)

