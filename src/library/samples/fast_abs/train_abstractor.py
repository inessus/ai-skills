""" trainer the abstractor"""
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


from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer

from library.utils.datasets.dictionary import make_vocab, PAD, START, UNK, END
from library.utils.datasets.jsonfile import JsonFileDataset
from library.utils.datasets.batcher import coll_fn, prepro_token_fn
from library.utils.datasets.batcher import convert_id_batch_copy, batchify_pad_fn_copy
from library.utils.datasets.batcher import BucketedGenerater
from library.utils.transforms.sequence import sequence_loss
from library.text.modules.base.embedding import make_embedding
from library.text.modules.copynet import CopySumm
from library.utils.pipeline.basicpipeline import basic_validate, BasicPipeline, get_basic_grad_fn
from library.utils.trainer.basictrainer import BasicTrainer

# NOTE: bucket size too large may sacrifice randomness,
#       to low may increase # of PAD tokens
BUCKET_SIZE = 6400  # 集装箱大小

# DATA_DIR = r'/Users/oneai/ai/data/cnndm'
# DATA_DIR = r'/Users/oneai/ai/data/bytecup'
DATA_DIR = "/media/webdev/store/competition/bytecup2018/data/"


class MatchDataset(JsonFileDataset):
    """
        优选文章和标题  根据Rouge贪婪搜索匹配出来的 文章和句子
        single article sentence -> single abstract sentence
        (dataset created by greedily matching ROUGE) 贪婪ROUGE搜索的数据库
    """

    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        """
        {
            'id': 'ee8871b15c50d0db17b0179a6d2beab35065f1e9',
            'article': ["editor 's note : in our behind the scenes series , cnn correspondents share their experiences in covering news and analyze the stories behind the events . here , soledad o'brien takes users inside a jail where many of the inmates are mentally ill .", "an inmate housed on the `` forgotten floor , '' where many mentally ill inmates are housed in miami before trial .", "miami , florida -lrb- cnn -rrb- -- the ninth floor of the miami-dade pretrial detention facility is dubbed the `` forgotten floor . '' here , inmates with the most severe mental illnesses are incarcerated until they 're ready to appear in court .", "most often , they face drug charges or charges of assaulting an officer -- charges that judge steven leifman says are usually `` avoidable felonies . '' he says the arrests often result from confrontations with police . mentally ill people often wo n't do what they 're told when police arrive on the scene -- confrontation seems to exacerbate their illness and they become more paranoid , delusional , and less likely to follow directions , according to leifman .", "so , they end up on the ninth floor severely mentally disturbed , but not getting any real help because they 're in jail .", "we toured the jail with leifman . he is well known in miami as an advocate for justice and the mentally ill . even though we were not exactly welcomed with open arms by the guards , we were given permission to shoot videotape and tour the floor . go inside the ` forgotten floor ' ''", "at first , it 's hard to determine where the people are . the prisoners are wearing sleeveless robes . imagine cutting holes for arms and feet in a heavy wool sleeping bag -- that 's kind of what they look like . they 're designed to keep the mentally ill patients from injuring themselves . that 's also why they have no shoes , laces or mattresses .", 'leifman says about one-third of all people in miami-dade county jails are mentally ill . so , he says , the sheer volume is overwhelming the system , and the result is what we see on the ninth floor .', "of course , it is a jail , so it 's not supposed to be warm and comforting , but the lights glare , the cells are tiny and it 's loud . we see two , sometimes three men -- sometimes in the robes , sometimes naked , lying or sitting in their cells .", "`` i am the son of the president . you need to get me out of here ! '' one man shouts at me .", 'he is absolutely serious , convinced that help is on the way -- if only he could reach the white house .', "leifman tells me that these prisoner-patients will often circulate through the system , occasionally stabilizing in a mental hospital , only to return to jail to face their charges . it 's brutally unjust , in his mind , and he has become a strong advocate for changing things in miami .", 'over a meal later , we talk about how things got this way for mental patients .', "leifman says 200 years ago people were considered `` lunatics '' and they were locked up in jails even if they had no charges against them . they were just considered unfit to be in society .", 'over the years , he says , there was some public outcry , and the mentally ill were moved out of jails and into hospitals . but leifman says many of these mental hospitals were so horrible they were shut down .', 'where did the patients go ? nowhere . the streets . they became , in many cases , the homeless , he says . they never got treatment .', 'leifman says in 1955 there were more than half a million people in state mental hospitals , and today that number has been reduced 90 percent , and 40,000 to 50,000 people are in mental hospitals .', "the judge says he 's working to change this . starting in 2008 , many inmates who would otherwise have been brought to the `` forgotten floor '' will instead be sent to a new mental health facility -- the first step on a journey toward long-term treatment , not just punishment .", "leifman says it 's not the complete answer , but it 's a start . leifman says the best part is that it 's a win-win solution . the patients win , the families are relieved , and the state saves money by simply not cycling these prisoners through again and again .", 'and , for leifman , justice is served . e-mail to a friend .'],
            'abstract': ["mentally ill inmates in miami are housed on the `` forgotten floor ''", "judge steven leifman says most are there as a result of `` avoidable felonies ''", "while cnn tours facility , patient shouts : `` i am the son of the president ''", "leifman says the system is unjust and he 's fighting for change ."],
            'extracted': [1, 3, 9, 11],
            'score': [0.5384615384615384, 0.6, 0.5294117647058824, 0.6153846153846154]
        }
        :param i:
        :return:  高分文章摘要对应
        (
            ["an inmate housed on the `` forgotten floor , '' where many mentally ill inmates are housed in miami before trial .", "most often , they face drug charges or charges of assaulting an officer -- charges that judge steven leifman says are usually `` avoidable felonies . '' he says the arrests often result from confrontations with police . mentally ill people often wo n't do what they 're told when police arrive on the scene -- confrontation seems to exacerbate their illness and they become more paranoid , delusional , and less likely to follow directions , according to leifman .", "`` i am the son of the president . you need to get me out of here ! '' one man shouts at me .", "leifman tells me that these prisoner-patients will often circulate through the system , occasionally stabilizing in a mental hospital , only to return to jail to face their charges . it 's brutally unjust , in his mind , and he has become a strong advocate for changing things in miami ."],
            ["mentally ill inmates in miami are housed on the `` forgotten floor ''", "judge steven leifman says most are there as a result of `` avoidable felonies ''", "while cnn tours facility , patient shouts : `` i am the son of the president ''", "leifman says the system is unjust and he 's fighting for change ."]
        )

        """
        js_data = super().__getitem__(i)
        art_sents, abs_sents = ([".".join(js_data['article'])], js_data['abstract'])
        # matched_arts = [art_sents[i] for i in extracts]  # 优选Id 换成句子
        return art_sents, abs_sents


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
        'vocab_size': vocab_size,   # W, 30004
        'emb_dim': emb_dim,         # E, 128
        'n_hidden': n_hidden,       # N, 256
        'bidirectional': bidirectional, # true
        'n_layer': n_layer          # L 1
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
    nll = lambda logit, target: F.nll_loss(logit, target, reduce=False) # [384, 30033] [384]

    def criterion(logits, targets):
        return sequence_loss(logits, targets, nll, pad_idx=PAD)

    return criterion, train_params


def build_batchers(word2id, cuda, debug):
    """
        转配DataLoader BucketedGenerater
    :param word2id: 向量字典
    :param cuda: 是否使用cuda
    :param debug: 是否调试
    :return:
    """
    prepro = prepro_token_fn(args.max_art, args.max_abs)  # 切分词，截断

    def sort_key(sample):
        """
            排序使用长度作为关键值
        :param sample:
        :return:
        """
        src, target = sample
        return (len(target), len(src))

    batchify = compose(
        batchify_pad_fn_copy(PAD, START, END, cuda=cuda),   # 填补标记 后进行
        convert_id_batch_copy(UNK, word2id)                # id化 先进行
     )

    train_loader = DataLoader(
        MatchDataset('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn  # 集装箱拆包，压成一维，滤0，判优，打包
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
    # batcher 字典，在整理数据的时候生成的词频字典
    with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f)
    word2id = make_vocab(wc, args.vsize) # W 30000
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
    parser.add_argument('--path', required=True, help='模型存储目录')

    parser.add_argument('--vsize', type=int, action='store', default=30000, help='字典大小')
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
    parser.add_argument('--patience', type=int, action='store', default=5, help='停止条件')

    parser.add_argument('--debug', action='store_true', help='运行debug模式，禁止进程创建')
    parser.add_argument('--no-cuda', action='store_true', help='禁止GPU训练')
    args = parser.parse_args()
    args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda


    main(args)

    """
    python train_abstractor.py --path=/Users/oneai/ai/data/cnndm/abstractor --w2v=/Users/oneai/ai/data/cnndm/word2vec/word2vec.128d.226k.bin
    python train_abstractor.py --path=/Users/oneai/ai/data/bytecup/abstractor --w2v=/Users/oneai/ai/data/bytecup/word2vec/word2vec.128d.1207k.bin
    python train_abstractor.py --path=/media/webdev/store/competition/cnndm/abstractor --w2v=/media/webdev/store/competition/cnndm/word2vec/word2vec.128d.1207k.bin
    python train_abstractor.py --path=/media/webdev/store/competition/bytecup2018/data/abstractor --w2v=/media/webdev/store/competition/bytecup2018/data/word2vec/word2vec.128d.1747k.bin
    CopySumm:
        embedding: Embedding(30004, 128, padding_idx=0)
        _enc_lstm: LSTM(128, 256, bidirectional=True)
        _dec_lstm: 
            MultiLayerLSTMCells
                _cells: ModuleList((0): LSTMCell(256, 256))
        _dec_h: Linear(in_features=512, out_features=256, bias=False)
        _dec_c: Linear(in_features=512, out_features=256, bias=False)
        _projection: 
            Sequential:
                (0): Linear(in_features=512, out_features=256, bias=True)
                (1): Tanh()
                (2): Linear(in_features=256, out_features=128, bias=False)
        _copy: _CopyLinear()
    
    """