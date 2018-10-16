""" train extractor (ML)"""
import argparse
import json
import os
from os.path import join, exists
import pickle as pkl

from cytoolz import compose

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.extract import ExtractSumm, PtrExtractSumm
from model.util import sequence_loss
from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer

from utils import PAD, UNK
from utils import make_vocab, make_embedding

from data.data import JsonFileDataset
from data.batcher import coll_fn_extract, prepro_fn_extract
from data.batcher import convert_batch_extract_ff, batchify_fn_extract_ff
from data.batcher import convert_batch_extract_ptr, batchify_fn_extract_ptr
from data.batcher import BucketedGenerater


BUCKET_SIZE = 6400


DATA_DIR = r'/Users/oneai/ai/data/cnndm'


class ExtractDataset(JsonFileDataset):
    """
        文章 -> 抽取索引
        article sentences -> extraction indices
        (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, 'cnndm', DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, extracts = js_data['article'], js_data['extracted']
        return art_sents, extracts


def build_batchers(net_type, word2id, cuda, debug):
    """
        构建输入桶，进行向量的预处理，补码，分片，token，向量化
    :param net_type:  网络类型 ff， rnn两种
    :param word2id:   向量字典
    :param cuda:      是否是用CUDA
    :param debug:     是否调试
    :return:
    """
    assert net_type in ['ff', 'rnn']
    prepro = prepro_fn_extract(args.max_word, args.max_sent)

    def sort_key(sample):
        src_sents, _ = sample
        return len(src_sents)

    batchify_fn = (batchify_fn_extract_ff if net_type == 'ff' else batchify_fn_extract_ptr)
    convert_batch = (convert_batch_extract_ff if net_type == 'ff' else convert_batch_extract_ptr)
    batchify = compose(batchify_fn(PAD, cuda=cuda), convert_batch(UNK, word2id))

    train_loader = DataLoader(
        ExtractDataset('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify, single_run=False, fork=not debug)

    val_loader = DataLoader(
        ExtractDataset('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify, single_run=True, fork=not debug)
    return train_batcher, val_batcher


def configure_net(net_type, vocab_size, emb_dim, conv_hidden, lstm_hidden, lstm_layer, bidirectional):
    """
        创建网络,定义配置参数
    :param net_type: 网络类型 ff，rnn两种
    :param vocab_size: 字典大小
    :param emb_dim:     嵌入尺寸
    :param conv_hidden:  卷积隐藏层大小
    :param lstm_hidden:  LSTM隐藏层大小
    :param lstm_layer:   LSTM隐藏层
    :param bidirectional:  双向LSTM
    :return:
    """
    assert net_type in ['ff', 'rnn']
    net_args = {
        'vocab_size': vocab_size,
        'emb_dim': emb_dim,
        'conv_hidden': conv_hidden,
        'lstm_hidden': lstm_hidden,
        'lstm_layer': lstm_layer,
        'bidirectional': bidirectional
    }

    net = (ExtractSumm(**net_args) if net_type == 'ff' else PtrExtractSumm(**net_args))
    return net, net_args


def configure_training(net_type, opt, lr, clip_grad, lr_decay, batch_size):
    """

    :param net_type:
    :param opt:
    :param lr:
    :param clip_grad:
    :param lr_decay:
    :param batch_size:
    :return:
    """
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    assert net_type in ['ff', 'rnn']
    opt_kwargs = {'lr': lr}

    train_params = {
        'optimizer': (opt, opt_kwargs),
        'clip_grad_norm': clip_grad,
        'batch_size': batch_size,
        'lr_decay': lr_decay
    }

    if net_type == 'ff':
        criterion = lambda logit, target: F.binary_cross_entropy_with_logits(logit, target, reduce=False)
    else:
        ce = lambda logit, target: F.cross_entropy(logit, target, reduce=False)

        def criterion(logits, targets):
            return sequence_loss(logits, targets, ce, pad_idx=-1)

    return criterion, train_params


def main(args):
    assert args.net_type in ['ff', 'rnn']
    # create data batcher, vocabulary
    # 批处理器
    with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f)
    word2id = make_vocab(wc, args.vsize)
    train_batcher, val_batcher = build_batchers(args.net_type, word2id, args.cuda, args.debug)

    # 生成网络
    net, net_args = configure_net(args.net_type, len(word2id), args.emb_dim, args.conv_hidden, args.lstm_hidden, args.lstm_layer, args.bi)
    if args.w2v:
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        embedding, _ = make_embedding({i: w for w, i in word2id.items()}, args.w2v)
        net.set_embedding(embedding)

    # 配置训练参数
    criterion, train_params = configure_training(args.net_type, 'adam', args.lr, args.clip, args.decay, args.batch)

    # 保存经验设计
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {
        'net': 'ml_{}_extractor'.format(args.net_type),
        'net_args': net_args,
        'traing_params': train_params
    }
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # 预训练
    val_fn = basic_validate(net, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=args.decay, min_lr=0, patience=args.lr_p)

    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline(meta['net'], net, train_batcher, val_batcher, args.batch, val_fn, criterion, optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path, args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


# --path=/Users/oneai/ai/data/cnndm --w2v=/Users/oneai/ai/data/cnndm/word2vec/word2vec.128d.226k.bin
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training of the feed-forward extractor (ff-ext, ML)')
    parser.add_argument('--path', required=True, help='root of the model')

    # model options
    parser.add_argument('--net-type', action='store', default='rnn', help='model type of the extractor (ff/rnn)')
    parser.add_argument('--vsize', type=int, action='store', default=30000, help='字典带下')
    parser.add_argument('--emb_dim', type=int, action='store', default=128, help='词嵌入维度')
    parser.add_argument('--w2v', action='store',help='使用预训练word2vec嵌入')
    parser.add_argument('--conv_hidden', type=int, action='store', default=100, help='卷积网络隐藏单元数量')
    parser.add_argument('--lstm_hidden', type=int, action='store', default=256, help='lSTM隐藏单元数量')
    parser.add_argument('--lstm_layer', type=int, action='store', default=1, help='LSTM Encoder层数')
    parser.add_argument('--no-bi', action='store_true', help='禁止双向 LSTM encoder')

    # length limit
    parser.add_argument('--max_word', type=int, action='store', default=100, help='单个 article sentence最大长度')
    parser.add_argument('--max_sent', type=int, action='store', default=60, help='单个 article article')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3, help='学习率')
    parser.add_argument('--decay', type=float, action='store', default=0.5, help='学习率延迟率')
    parser.add_argument('--lr_p', type=int, action='store', default=0, help='patience学习率延迟率')
    parser.add_argument('--clip', type=float, action='store', default=2.0, help='梯度裁剪')
    parser.add_argument('--batch', type=int, action='store', default=32, help='训练批次尺寸')
    parser.add_argument( '--ckpt_freq', type=int, action='store', default=3000, help='number of update steps for checkpoint and validation')
    parser.add_argument('--patience', type=int, action='store', default=5, help='patience for early stopping')

    parser.add_argument('--debug', action='store_true', help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true', help='disable GPU training')
    args = parser.parse_args()
    args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    main(args)
