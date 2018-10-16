import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import numpy as np
import torch.optim as optim


# %matplotlib inline

x = torch.rand((10000, 1))

y = 2 * x + 3


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


net = LinearRegressionModel()
citierion = nn.MSELoss()
optimeter = optim.SGD(net.parameters(), lr=1e-4)


num_epochs = 1000
for epoch in range(num_epochs):
    inputs = Variable(x)
    target = Variable(y)

    out = net(inputs)
    loss = citierion(out, target)
    optimeter.zero_grad()
    loss.backward()
    optimeter.step()
    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss: {:.6f}'
              .format(epoch+1, num_epochs, loss.data[0]))

if __name__ == '__main__':
    print("test")