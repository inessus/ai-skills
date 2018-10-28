import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from library.vision.convert import Convert
from library.vision.logger import Logger


class BaseTrain(object):
    def __init__(self, net, optimizer=None, criterizon=None, data_loader=None, lr=1e-4, batch=32, clip=2.0, epoches=100, patience=5,
                 cuda=False, logger=None):
        self.net = net
        self.lr = lr
        self.criterion = criterizon if criterizon else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else optim.SGD(self.net.parameters(), lr=self.lr)
        self.logger = logger if logger else Logger('./logs')
        self.clip = clip
        self.batch = batch
        self.patience = patience
        self.epoches = epoches
        self.data_loader = data_loader
        self.cuda = cuda
        if cuda:
            self.net = net.cuda()

        self.since = time.time()

    def save(self, path="./net.ptch"):
        torch.save(self.net.state_dict(), path)

    def load(self, path="./net.ptch"):
        # with open(path, 'wb') as f
        # torch.load()
        pass

    def train(self, data_loader=None, show=True):
        # 开始训练
        for epoch in range(self.epoches):
            if show:
                print('*' * 10)
                print('epoch {}'.format(epoch + 1))
            self.since = time.time()

            for i, data in enumerate(data_loader, 1):
                img, label = data

                img = Convert.to(img, self.cuda)
                label = Convert.to(label, self.cuda)
                # 向前传播
                out = self.net(img)
                # out = out.view(-1, 1)
                # label = label.view(-1, 1)
                loss = self.criterion(out, label)
                # 向后传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 300 == 0:
                    print('[{}/{}] Loss: {:.6f} '.format(
                        epoch + 1, self.epoches, loss))

    def eval(self, test_loader):
        self.net.eval()
        eval_loss = 0.
        eval_acc = 0.
        for data in test_loader:
            img, label = data
            img = img.view(img.size(0), -1)
            if self.cuda:
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = self.net(img)
            loss = self.criterion(out, label)
            eval_loss += loss.data[0] * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.data[0]

        print('Time:{:.1f} s'.format(time.time() - self.since))
        print()
