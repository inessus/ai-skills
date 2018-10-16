import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import namedtuple
from IPython.display import Image

np.random.seed(2018)


class Fashion(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transofrm=None, download=False):
        self.urls = [
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
        ]
        super(Fashion, self).__init__(
            root, train=train, transform=transform, target_transform=target_transofrm, download=download
        )
        
def decode_label(l):
    return ["Top",
     "Trouser",
     "Pullover",
     "Dress",
     "Coat",
     "Sandal",
     "Shirt",
     "Sneaker",
     "Bag",
     "Ankle boot"
    ][l]

train_data = Fashion('/tmp/data', train=True, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                            ]))

test_data = Fashion('/tmp/data', train=False, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                            ]))
idxs = np.random.randint(100, size=8)
f, a = plt.subplots(2, 4, figsize=(10, 5))
for i in range(8):
    X = train_data.train_data[idxs[i]]
    Y = train_data.train_labels[idxs[i]]
    r, c = i // 4, i % 4
    a[r][c].set_title(decode_label(Y))
    a[r][c].axis('off')
    a[r][c].imshow(X.numpy())
plt.draw()

class FashionModel(nn.Module):
    def __init__(self):
        super(FashionModel, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x
print(FashionModel())
train_size = train_data.train_data.shape[0]
val_size, train_size = int(0.20 * train_size), int(0.80 * train_size)

test_size = test_data.test_data.shape[0]
batch_size = 100

train_loader = torch.utils.data.DataLoader(train_data, 
                                          batch_size=batch_size,
                                          sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(val_size, val_size+train_size)))
val_loader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=batch_size,
                                          sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(0, val_size)))
test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)
Metric = namedtuple('Metric', ['loss', 'train_error', 'val_error'])

def inference(model, loader, n_members):
    correct = 0
    for data, label in loader:
        X = Variable(data.view(-1, 784))
        Y = Variable(label)
        out = model(X)
        pred = out.data.max(1, keepdim=True)[1]
        predicted = pred.eq(Y.data.view_as(pred))
        correct += predicted.sum()
    return correct.numpy() / n_members

class Trainer():
    def __init__(self, model, optimizer, load_path=None):
        self.model = model
        if load_path is not None:
            self.model = torch.load(load_path)
        self.optimizer = optimizer
        
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def run(self, epochs):
        print("Start Training......")
        self.metrics = []
        for e in range(n_epochs):
            epoch_loss = 0
            correct = 0
            for batch_idx, (data, label) in enumerate(train_loader):
                self.optimizer.zero_grad()
                X = Variable(data.view(-1, 784))
                Y = Variable(label)
                out = self.model(X) # 模型输出
                pred = out.data.max(1, keepdim=True)[1]     # 选定行，逐列计算最大值，取其索引
                predicted = pred.eq(Y.data.view_as(pred))   # 根据标签和索引的匹配向量
                correct += predicted.sum()                  # 预测正确的个数

                loss = F.nll_loss(out, Y)                   # 根据模型输出和标签计算损失函数
                loss.backward()                             # 损失函数逆向求导
                self.optimizer.step()                       # 更新优化迭代器
                epoch_loss += loss.data[0]                  # 累计epochs累计误差(tensor -> value)
            total_loss = epoch_loss.numpy() / train_size    # 累计误差计算比率
            train_error = 1.0 - correct.numpy() / train_size # 正确个数计算比率
            val_error = 1.0 - inference(self.model,val_loader, val_size)
            print("epoch: {0}, loss: {1:.8f}".format(e+1, total_loss))
            self.metrics.append(Metric(loss=total_loss, train_error=train_error, val_error=val_error))

### LET's TRAIN ###
# A function to apply "normal" distribution on the parameters
def init_randn(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 1)

# we first initialize a Fashion Object and initilize the parameters "normally"
normalmodel = FashionModel()
normalmodel.apply(init_randn)

n_epochs = 8
print("SGD OPTIMIZER")
SGDOptimizer = torch.optim.SGD(normalmodel.parameters(), lr=0.01)
sgd_trainer = Trainer(normalmodel, SGDOptimizer)
sgd_trainer.run(n_epochs)
sgd_trainer.save_model('./sgd_model.pt')
print('')

print("ADAM OPTIMIZER")
normalmodel = FashionModel()
normalmodel.apply(init_randn)
AdamOptimizer = torch.optim.Adam(normalmodel.parameters(), lr=0.01)
adam_trainer = Trainer(normalmodel, AdamOptimizer)
adam_trainer.run(n_epochs)
adam_trainer.save_model('./adam_model.pt')
print('')

print("RMSPROP OPTIMIZER")
normalmodel = FashionModel()
normalmodel.apply(init_randn)
RMSPropOptimizer = torch.optim.RMSprop(normalmodel.parameters(), lr=0.01)
rms_trainer = Trainer(normalmodel, RMSPropOptimizer)
rms_trainer.run(n_epochs)
rms_trainer.save_model('./rmsprop_model.pt')
print('')

### TEST ###
model = FashionModel()
model.load_state_dict(torch.load('./sgd_model.pt'))
test_acc = inference(model, test_loader, test_size)
print("Test accuracy of model optimizer with SGD {0:.2f}".format(test_acc * 100))

model = FashionModel()
model.load_state_dict(torch.load('./adam_model.pt'))
test_acc = inference(model, test_loader, test_size)
print("Test accuracy of model optimizer with Adam {0:.2f}".format(test_acc * 100))

model = FashionModel()
model.load_state_dict(torch.load('./rmsprop_model.pt'))  
test_acc = inference(model, test_loader, test_size)
print("Test accuracy of model optimizer with RMSProp {0:.2f}".format(test_acc * 100))

def training_plot(metrics):
    plt.figure(1)
    plt.plot([m.loss for m in metrics], 'b')
    plt.title('Training Loss')
    plt.show()

training_plot(sgd_trainer.metrics)
training_plot(adam_trainer.metrics)
training_plot(rms_trainer.metrics)

def init_randn(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 1)

def init_custom(m):
    if type(m) == nn.Linear:
        rw = torch.randn(m.weight.data.size())
        m.weight.data.copy_(rw)

def init_xavier(m):
    if type(m) == nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)

n_epochs = 3

print("NORMAL INIT WEIGHTS")
AdamOptimizer = torch.optim.Adam(normalmodel.parameters(), lr=0.01)
normal_trainer = Trainer(normalmodel, AdamOptimizer)
normal_trainer.run(n_epochs)
normal_trainer.save_model('./normal_model.pt')
print('')

print("XAVIER INIT WEIGHTS")
xaviermodel = FashionModel()
xaviermodel.apply(init_xavier)
AdamOptimizer = torch.optim.Adam(normalmodel.parameters(), lr=0.01)
xavier_trainer = Trainer(xaviermodel, AdamOptimizer)
xavier_trainer.run(n_epochs)
xavier_trainer.save_model('./xavier_model.pt')
print('')


def training_plot(metrics):
    plt.figure(1)
    plt.plot([m.loss for m in metrics], 'b')
    plt.title('Training Loss')
    plt.show()

training_plot(normal_trainer.metrics)
training_plot(xavier_trainer.metrics)

