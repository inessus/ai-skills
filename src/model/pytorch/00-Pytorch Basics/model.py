import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


if __name__ == "__main__":
    mm = Model()

    F.relu6()
    F.sigmoid()
    F.tanh()
    F.softplus
    F.softplus
    F.softmax()
    F.softmin()
    for k, v, in mm._backend.function_classes.items():
        print("{0:40s} {1}".format(k, v))