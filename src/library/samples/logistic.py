import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

from library.vision.modules.basicmodels import Logstic_Regression
from library.utils.pipeline.basicpipeline import BasicPipeline, basic_validate, get_basic_grad_fn
from library.utils.trainer.basictrainer import BasicTrainer


def build_batchers(batch):
    # 下载训练集 MNIST 手写数字训练集
    train_dataset = datasets.MNIST(
        root='/media/webdev/store/data', train=True, transform=transforms.ToTensor(), download=True)

    test_dataset = datasets.MNIST(
        root='/media/webdev/store/data', train=False, transform=transforms.ToTensor())

    train_batcher = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_batcher = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    return train_batcher, val_batcher


def configure_net(input_size, output_size):
    net_args = {
        'in_dim': input_size,
        'n_class': output_size
    }
    net = Logstic_Regression(**net_args)
    return net, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size):
    opt_kwargs = {'lr': lr}

    train_params = {
        'optimizer': (opt, opt_kwargs),
        'clip_grad_norm': clip_grad,
        'batch_size': batch_size,
        'lr_decay': lr_decay
    }
    criterion = nn.CrossEntropyLoss()
    return criterion, train_params


def main(args):
    train_batcher, val_batcher = build_batchers(args.batch)
    net, net_args = configure_net(args.input_size, args.output_size)
    criterion, train_params = configure_training('sgd', args.lr, args.clip, args.decay, args.batch)
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    pipeline = BasicPipeline("line", net, train_batcher, val_batcher, args.batch, criterion, optimizer, args.clip)
    trainer = BasicTrainer(pipeline, args.path, args.ckpt_freq, args.patience)
    trainer.train()


# 定义超参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training of the Logistic Regression for MNIST')
    parser.add_argument('--path', default="/tmp/", help='root of the model')
    # model options
    parser.add_argument('--input_size', type=int, action='store', default=28*28, help='insize')
    parser.add_argument('--output_size', type=int, action='store', default=10, help='output')

    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3, help='学习率')
    parser.add_argument('--decay', type=float, action='store', default=0.5, help='学习率延迟率')
    parser.add_argument('--lr_p', type=int, action='store', default=0, help='patience学习率延迟率')
    parser.add_argument('--clip', type=float, action='store', default=2.0, help='梯度裁剪')
    parser.add_argument('--batch', type=int, action='store', default=32, help='训练批次尺寸')
    parser.add_argument('--epoches', type=int, action='store', default=100, help='epoches')
    parser.add_argument( '--ckpt_freq', type=int, action='store', default=3000, help='验证的批次频率')
    parser.add_argument('--patience', type=int, action='store', default=5, help='停止学习的容忍度')

    parser.add_argument('--debug', action='store_true', help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true', help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)

