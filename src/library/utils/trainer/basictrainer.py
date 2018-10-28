import os
import sys
import socket
from datetime import datetime
import tensorboardX
from os.path import join, exists
from torch.optim.lr_scheduler import ReduceLROnPlateau
from library.utils.pipeline.basicpipeline import BasicPipeline


class BasicTrainer(object):
    """ Basic trainer with minimal function and early stopping"""
    def __init__(self, pipeline, save_dir, ckpt_freq, patience, scheduler=None, val_mode='loss'):
        """

        :param pipeline: 流水线
        :param save_dir: 保存路径
        :param ckpt_freq: 检查点频率
        :param patience: 退出标准
        :param scheduler:
        :param val_mode: 验证模型
        """
        assert isinstance(pipeline, BasicPipeline)
        assert val_mode in ['loss', 'score']
        self._pipeline = pipeline
        self._save_dir = save_dir

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = join(join(save_dir, 'runs'), "{0}-{1}-{2}".format(current_time, socket.gethostname(), "name"))
        self._logger = tensorboardX.SummaryWriter(log_dir) # 日志保存
        if not exists(join(save_dir, 'ckpt')):
            os.makedirs(join(save_dir, 'ckpt'))

        self._ckpt_freq = ckpt_freq #
        self._patience = patience
        self._sched = scheduler
        self._val_mode = val_mode

        self._step = 0
        self._running_loss = None
        # state vars for early stopping
        self._current_p = 0
        self._best_val = None

    def log(self, dict):
        log_dict = dict['log_dict']
        input = dict['input']
        loss = log_dict['loss'] if 'loss' in log_dict else log_dict['reward']
        if self._running_loss is not None:
            self._running_loss = 0.99*self._running_loss + 0.01*loss
        else:
            self._running_loss = loss
        print('\rtrainer step: {}, {}: {:.4f}'.format(
            self._step,
            'loss' if 'loss' in log_dict else 'reward',
            self._running_loss), end="")
        # sys.stdout.flush()
        self._logger.add_graph(self._pipeline.net, (input,))
        for key, value in log_dict.items():
            self._logger.add_scalar(
                '{}_{}'.format(key, self._pipeline.name), value, self._step)

    def validate(self):
        val_log = self._pipeline.validate() # 验证的事情交给流水线
        for key, value in val_log.items():
            self._logger.add_scalar(
                'val_{}_{}'.format(key, self._pipeline.name),
                value, self._step
            )
        if 'reward' in val_log:
            val_metric = val_log['reward']
        else:
            val_metric = (val_log['loss'] if self._val_mode == 'loss'
                          else val_log['score'])
        return val_metric

    def checkpoint(self):
        val_metric = self.validate()
        self._pipeline.checkpoint(
            join(self._save_dir, 'ckpt'), self._step, val_metric)
        if isinstance(self._sched, ReduceLROnPlateau):
            self._sched.step(val_metric)
        elif self._sched:
            self._sched.step()
        stop = self.check_stop(val_metric)
        return stop

    def check_stop(self, val_metric):
        """
            损失或者分数，连续不满意的次数达到阈值，则停止
        :param val_metric:
        :return:
        """
        if self._best_val is None:
            self._best_val = val_metric
        elif ((val_metric < self._best_val and self._val_mode == 'loss')
              or (val_metric > self._best_val and self._val_mode == 'score')):
            self._current_p = 0
            self._best_val = val_metric
        else:
            self._current_p += 1
        return self._current_p >= self._patience

    def train(self):
        try:
            start = datetime.now()
            print('Start training')
            while True:
                log_dict = self._pipeline.train_step()
                self._step += 1
                self.log(log_dict)

                if self._step % self._ckpt_freq == 0:
                    stop = self.checkpoint()
                    if stop:
                        break
            print('Training finised in ', datetime.now()-start)
        finally:
            self._pipeline.terminate()