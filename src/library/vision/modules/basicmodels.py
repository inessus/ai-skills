import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.dense1 = nn.Linear(300, 300)
        self.dense2 = nn.Linear(300, 300)
        self.dense3 = nn.Linear(300, 3)

    def forward(self, x):
        a = torch.zeros(300)
        a[:len(x)] = x
        x = a
        x = F.dropout(F.relu(self.dense1(x)), 0.2)
        # x = nn.BatchNorm1d(x, 10)
        x = F.dropout(F.relu(self.dense2(x)), 0.3)
        # x = nn.BatchNorm1d(x, 10)
        x = F.softmax(self.dense3(x))
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(10))
        self.dense2 = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(10))
        self.dense3 = nn.Sequential(
            nn.Linear(300, 3),
            nn.Softmax())

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.dense1 = nn.Sequential()
        self.dense1.add_module("linear1", nn.Linear(300, 300))
        self.dense1.add_module("relu1", nn.ReLU())
        self.dense1.add_module("dropout1", nn.Dropout(0.2))
        self.dense1.add_module("batchnorm1d1", nn.BatchNorm1d(10))

        self.dense2 = nn.Sequential()
        self.dense2.add_module("linear2", nn.Linear(300, 300))
        self.dense2.add_module("relu2", nn.ReLU())
        self.dense2.add_module("dropout2", nn.Dropout(0.3))
        self.dense2.add_module("batchnorm1d2", nn.BatchNorm1d(10))

        self.dense3 = nn.Sequential()
        self.dense3.add_module("linear3", nn.Linear(300, 3))
        self.dense3.add_module("softmax", nn.Softmax())

    def forward(self, x):
        x = self.dense1(x)
        # x = x.view(x.size(0), -1)
        x = self.dense2(x)
        # x = x.view(x.size(0), -1)
        x = self.dense3(x)
        return x


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.dense1 = nn.Sequential(
            OrderedDict([
                    ("linear1", nn.Linear(300, 300)),
                    ("relu1", nn.ReLU()),
                    ("dropout1", nn.Dropout(0.2)),
                    ("batchnorm1d1", nn.BatchNorm1d(10))
            ]))
        self.dense2 = nn.Sequential(
            OrderedDict([
                    ("linear2", nn.Linear(300, 300)),
                    ("relu2", nn.ReLU()),
                    ("dropout2", nn.Dropout(0.3)),
                    ("batchnorm1d2", nn.BatchNorm1d(10))
            ]))

        self.dense3 = nn.Sequential(
            OrderedDict([
                ("linear3", nn.Linear(300, 3)),
                ("softmax", nn.Softmax())
            ]))


# 定义 Logistic Regression 模型
class Logstic_Regression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Logstic_Regression, self).__init__()
        self.in_dim = in_dim
        self.n_class = n_class
        self.logstic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将图片展开成 28x28
        out = self.logstic(x)
        return out


class FeedforwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedforwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, num_classes=0):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class ConvNet4(nn.Module):
    def __init__(self, num_classes=0):
        super(ConvNet4, self).__init__()
        self.layer1 = nn.Sequential(
            OrderedDict([
                    ("conv1", nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)),
                    ("batchnor1", nn.BatchNorm2d(16)),
                    ("relu1", nn.ReLU()),
                    ("maxpool1", nn.MaxPool2d(kernel_size=2, stride=2))
            ]))
        self.layer2 = nn.Sequential(
            OrderedDict([
                    ("conv1", nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)),
                    ("batchnor1", nn.BatchNorm2d(32)),
                    ("relu1", nn.ReLU()),
                    ("maxpool1", nn.MaxPool2d(kernel_size=2, stride=2))
            ]))

        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        """

        :param x: 32*1*28*28
        :return:
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out