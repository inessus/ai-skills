""" module providing basic training utilities"""

from os.path import join
from time import time
from datetime import timedelta
from itertools import starmap
from torch.utils.data import DataLoader
from cytoolz import curry, reduce

import torch
from torch.nn.utils import clip_grad_norm_


def get_basic_grad_fn(net, clip_grad, max_grad=1e2):
    """
        梯度裁剪函数
    :param net: 模型忘了
    :param clip_grad: 梯度最大范数
    :param max_grad:
    :return:
    """
    def f():
        grad_norm = clip_grad_norm_([p for p in net.parameters() if p.requires_grad], clip_grad)
        # grad_norm = grad_norm.item()
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log = {'grad_norm': grad_norm}
        return grad_log
    return f



@curry
def compute_loss(net, criterion, fw_args, loss_args):
    """
        调用网络，计算验证数据，然后和待验证数据计算误差
    :param net:  网络
    :param criterion: 误差函数
    :param fw_args: 网络参数
    :param loss_args: 损失参数
    :return:
    """
    # loss = criterion(*((net(*fw_args),) + loss_args))
    out = net(fw_args)
    loss = criterion(out, loss_args)
    return loss


@curry
def val_step(loss_step, fw_args, loss_args):
    """
        调用计算损失函数，获得验证数据，
    :param loss_step: 误差函数
    :param fw_args:  网络参数
    :param loss_args: 损失参数
    :return:
    """
    loss = loss_step(fw_args, loss_args)
    if loss.numel() == 1:
        return 1, loss.sum().item()
    else:
        return loss.size(0), loss.sum().item()


@curry
def basic_validate(net, criterion, val_batches):
    """
        验证三部曲，网络、标准和数据。 基本验证步骤
    :param net: 待验证网络
    :param criterion:  误差标准
    :param val_batches: 待验证数据，批次的XY
    :return:
    """
    print('\t running validation ... ', end='')
    net.eval()
    start = time()
    with torch.no_grad():
        validate_fn = val_step(compute_loss(net, criterion))
        n_data, tot_loss = reduce(
            lambda a, b: (a[0]+b[0], a[1]+b[1]),
            starmap(validate_fn, val_batches),
            (0, 0)
        )
    if n_data == 0:
        n_data = 0
    val_loss = tot_loss / n_data
    print('validation finished in {}'.format(timedelta(seconds=int(time()-start))))
    print('validation -------- loss: {:.4f} ... '.format(val_loss))
    return {'loss': val_loss}


class BasicPipeline(object):
    def __init__(self, name, net, train_batcher, val_batcher, batch_size, criterion, optim, clip=None, val_fn=None, grad_fn=None):
        """
            所谓的流水线，就是指的是 数据装载 预处理 切分词 向量化 分批 训练 求梯度 反向传播 下一epoch
        :param net: 模型忘了
        :param train_batcher: 训练批次器 本次是水桶数据装载器
        :param val_batcher: 验证批次器
        :param batch_size: 批次尺寸
        :param val_fn: 验证函数
        :param criterion: 误差标准
        :param optim: 优化器
        :param grad_fn: 梯度裁剪函数
        """
        self._name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._criterion = criterion
        self._opt = optim

        # grad_fn is calleble without input args that modifyies gradient
        self._grad_fn = grad_fn if grad_fn else get_basic_grad_fn(net, clip)

        # it should return a dictionary of logging values
        self._val_fn = val_fn if val_fn else basic_validate(net, criterion)

        self._n_epoch = 0  # epoch not very useful?
        self._batch_size = batch_size
        self._batches = self.batches() # 批处理器，开始调用

    def batches(self):
        if callable(self._train_batcher):
            batcher = self._train_batcher(self._batch_size) # 如果有桶
        else:
            batcher = self._train_batcher

        while True:
            for fw_args, bw_args in batcher:
                yield fw_args, bw_args
            self._n_epoch += 1

    def get_loss_args(self, net_out, bw_args):
        if isinstance(net_out, tuple):
            loss_args = net_out + bw_args
        else:
            loss_args = (net_out, ) + bw_args
        return loss_args

    def train_step(self):
        # forward pass of model
        self._net.train()
        fw_args, bw_args = next(self._batches)
        article, art_lens, abstract, extend_art, extend_vsize = fw_args
        # net_out = self._net(article, art_lens, abstract, extend_art, extend_vsize) # [B,T',V']
        net_out = self._net(*fw_args) # [B,T',V']

        # get logs and output for logging, backward
        log_dict = {}
        # loss_args = self.get_loss_args(net_out, bw_args)

        # backward and update ( and optional gradient monitoring )
        loss = self._criterion(net_out, bw_args).mean()
        loss.backward()
        log_dict['loss'] = loss.item()
        # log_dict['input'] = fw_args
        if self._grad_fn is not None:
            log_dict.update(self._grad_fn())
        self._opt.step()
        self._net.zero_grad()
        log = {'log_dict': log_dict, 'input': fw_args}

        return log

    def validate(self):
        if callable(self._val_batcher):
            batcher = self._val_batcher(self._batch_size) # 如果有桶
        else:
            batcher = self._val_batcher
        return self._val_fn(batcher)

    def checkpoint(self, save_path, step, val_metric=None):
        save_dict = {}
        if val_metric is not None:
            name = 'ckpt-{:6f}-{}'.format(val_metric, step)
            save_dict['val_metric'] = val_metric
        else:
            name = 'ckpt-{}'.format(step)

        save_dict['state_dict'] = self._net.state_dict()
        save_dict['optimizer'] = self._opt.state_dict()
        torch.save(save_dict, join(save_path, name))

    def terminate(self):
        try:
            self._train_batcher.terminate()
            self._val_batcher.terminate()
        except:
            pass

    @property
    def net(self):
        return self._net

    @property
    def name(self):
        return self._name

