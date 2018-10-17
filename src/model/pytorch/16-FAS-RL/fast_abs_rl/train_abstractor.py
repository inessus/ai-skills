""" train the abstractor"""
import argparse
import json
import os
from os.path import join, exists
import pickle as pkl
<<<<<<< HEAD
=======
import platform

>>>>>>> f0bd9a5b01fe49a55f538ece70dac34e89887f1f
from cytoolz import compose

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.copy_summ import CopySumm
from model.util import sequence_loss
from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer

from data.data import JsonFileDataset
from data.batcher import coll_fn, prepro_fn
from data.batcher import convert_batch_copy, batchify_fn_copy
from data.batcher import BucketedGenerater

from utils import PAD, UNK, START, END
from utils import make_vocab, make_embedding

# NOTE: bucket size too large may sacrifice randomness,
#       to low may increase # of PAD tokens
BUCKET_SIZE = 6400

DATA_DIR = r'/Users/oneai/ai/data/cnndm'


class MatchDataset(JsonFileDataset):
    """
        优选文章和标题
        single article sentence -> single abstract sentence
        (dataset created by greedily matching ROUGE) 贪婪ROUGE搜索的数据库
    """

    def __init__(self, split):
        super().__init__(split, 'cnndm', DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, abs_sents, extracts = (js_data['article'], js_data['abstract'], js_data['extracted'])
        matched_arts = [art_sents[i] for i in extracts]  # 优选Id 换成句子
        return matched_arts, abs_sents[:len(extracts)]


def configure_net(vocab_size, emb_dim, n_hidden, bidirectional, n_layer):
    """
        网络参数配置
    :param vocab_size:  字典大小
    :param emb_dim:     嵌入维度
    :param n_hidden:    隐层尺寸
    :param bidirectional:   是否是双向
    :param n_layer:     层数
    :return:
    """

    net_args = {
        'vocab_size': vocab_size,
        'emb_dim': emb_dim,
        'n_hidden': n_hidden,
        'bidirectional': bidirectional,
        'n_layer': n_layer
    }
    net = CopySumm(**net_args)
    return net, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size):
    """
        训练参数配置
        supports Adam optimizer only
    :param opt:
    :param lr:
    :param clip_grad:
    :param lr_decay:
    :param batch_size:
    :return:
    """
    assert opt in ['adam']
    opt_kwargs = {'lr': lr}

    train_params = {
        'optimizer': (opt, opt_kwargs),
        'clip_grad_norm': clip_grad,
        'batch_size': batch_size,
        'lr_decay': lr_decay
    }
    nll = lambda logit, target: F.nll_loss(logit, target, reduce=False)

    def criterion(logits, targets):
        return sequence_loss(logits, targets, nll, pad_idx=PAD)

    return criterion, train_params


def build_batchers(word2id, cuda, debug):
    """

    :param word2id: 向量字典
    :param cuda: 是否使用cuda
    :param debug: 是否调试
    :return:
    """
    prepro = prepro_fn(args.max_art, args.max_abs)  # token函数定义

    def sort_key(sample):
        src, target = sample
        return (len(target), len(src))

    batchify = compose(
        batchify_fn_copy(PAD, START, END, cuda=cuda),   # 补码
        convert_batch_copy(UNK, word2id)    # 向量化
    )

    train_loader = DataLoader(
        MatchDataset('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn  # 拆包，压成一维，滤0，判优，打包
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)

    val_loader = DataLoader(
        MatchDataset('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn  # 拆包，压成一维，滤0，判优，打包
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)
    return train_batcher, val_batcher


def main(args):
    # create data batcher, vocabulary
    # batcher
    with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f)
    word2id = make_vocab(wc, args.vsize)
    train_batcher, val_batcher = build_batchers(word2id, args.cuda, args.debug)

    # 生成CopyNet
    net, net_args = configure_net(len(word2id), args.emb_dim, args.n_hidden, args.bi, args.n_layer)
    if args.w2v:
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        embedding, _ = make_embedding({i: w for w, i in word2id.items()}, args.w2v)
        net.set_embedding(embedding)

    # 配置训练参数 configure training setting
    criterion, train_params = configure_training('adam', args.lr, args.clip, args.decay, args.batch)

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {
        'net': 'base_abstractor',
        'net_args': net_args,
        'traing_params': train_params
    }
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)    # 缩进

    # 准备训练
    val_fn = basic_validate(net, criterion) # 基礎驗證
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)

    # 开始训练。。。。
    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


# --w2v=/Users/oneai/ai/data/cnndm/word2vec/word2vec.128d.226k.bin
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training of the abstractor (ML)')
    parser.add_argument('--path', required=True, help='模型根目录')

<<<<<<< HEAD
    parser.add_argument('--vsize', type=int, action='store', default=30000, help='字典大小')
=======
<<<<<<< HEAD
    parser.add_argument('--vsize', type=int, action='store', default=30000, help='字典大小')
=======
<<<<<<< HEAD
    parser.add_argument('--vsize', type=int, action='store', default=80000, help='字典大小')
=======
    parser.add_argument('--vsize', type=int, action='store', default=30000, help='字典大小')
>>>>>>> 4ea1f663ee6e652cc95a0830c027227b91c562f6
>>>>>>> a6c58ab3456443573d54820cd38bc860739e28c8
>>>>>>> f0bd9a5b01fe49a55f538ece70dac34e89887f1f
    parser.add_argument('--emb_dim', type=int, action='store', default=128, help='嵌入')
    parser.add_argument('--w2v', action='store', help='使用word2vec嵌入')
    parser.add_argument('--n_hidden', type=int, action='store', default=256, help='LSTM隐藏单元个数')
    parser.add_argument('--n_layer', type=int, action='store', default=1, help='LSTM层数')
    parser.add_argument('--no-bi', action='store_true', help='禁止双向LSTM encoder ')

    # 长度限制
    parser.add_argument('--max_art', type=int, action='store', default=100, help='单个文章句子的最大单词个数')
    parser.add_argument('--max_abs', type=int, action='store', default=30, help='单个标题句子的最大单词个数')
    # 训练选项
    parser.add_argument('--lr', type=float, action='store', default=1e-3, help='学习率')
    parser.add_argument('--decay', type=float, action='store', default=0.5, help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0, help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0, help='梯度裁剪最大值')
    parser.add_argument('--batch', type=int, action='store', default=32, help='批量尺寸')
    parser.add_argument('--ckpt_freq', type=int, action='store', default=3000, help='断点和验证的大小')
    parser.add_argument('--patience', type=int, action='store', default=5, help='patience for early stopping')

    parser.add_argument('--debug', action='store_true', help='运行debug模式，禁止进程创建')
    parser.add_argument('--no-cuda', action='store_true', help='禁止GPU训练')
    args = parser.parse_args()
    args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)
