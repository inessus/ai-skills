""" batching """
import random
from collections import defaultdict

from toolz.sandbox import unzip
from cytoolz import curry, concat, compose

import torch
import torch.multiprocessing as mp


# Batching functions
def coll_fn(data):
    """
        (callable, optional): merges a list of samples to form a mini-batch.
        拆包，压成一维，滤0，判优，打包
    :param data:
    :return:
    """
    source_lists, target_lists = unzip(data)
    # NOTE: independent filtering works because
    #       source and targets are matched properly by the Dataset
    sources = list(filter(bool, concat(source_lists)))
    targets = list(filter(bool, concat(target_lists)))
    assert all(sources) and all(targets)
    return sources, targets


def coll_fn_extract(data):
    """
        判优， 过滤，打包
    :param data:
    :return:
    """
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts = d
        return source_sents and extracts
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch


@curry
def tokenize(max_len, texts):
    """
        文章》句子》单词  把文章tokenize成指定位数的单词向量
    :param max_len: 单词最大个数
    :param texts:  文章输入
    :return: 指定维数的单词向量
    """
    return [t.lower().split()[:max_len] for t in texts]


def conver2id(unk, word2id, words_list):
    """
        文章》vec 把tokenize的文章，进行向量化
    :param unk:
    :param word2id:
    :param words_list:
    :return:
    """
    word2id = defaultdict(lambda: unk, word2id) # 没看出作用来
    return [[word2id[w] for w in words] for words in words_list]


@curry
def prepro_fn(max_src_len, max_tgt_len, batch):
    """
        对原始数据进行预处理，进行并发执行（拆包、截断X,Y的长度tokenize、打包）
    :param max_src_len: 最大源长度 100
    :param max_tgt_len: 最大目标长度 30
    :param batch:   批次原始数据数据 (23126*[****], 23126*[*****])
    :return:
    """
    sources, targets = batch
    sources = tokenize(max_src_len, sources)
    targets = tokenize(max_tgt_len, targets)
    batch = list(zip(sources, targets))
    return batch


@curry
def prepro_fn_extract(max_src_len, max_src_num, batch):
    """
        tokenized源数据 并筛选extracts数据
    :param max_src_len: tokenize最大长度
    :param max_src_num: 截取长度，也是extracts的长度
    :param batch:
    :return:
    """
    def prepro_one(sample):
        """
        :param sample: 单个抽取样本数据 将extracts数据长度小于max_src_num的数据过滤出来
        :return:
        """
        source_sents, extracts = sample
        tokenized_sents = tokenize(max_src_len, source_sents)[:max_src_num]
        cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents), extracts))
        return tokenized_sents, cleaned_extracts
    batch = list(map(prepro_one, batch))
    return batch


@curry
def convert_batch(unk, word2id, batch):
    """
        拆包、向量化，打包
    :param unk:
    :param word2id:  word2vec训练的结果，单词变向量
    :param batch:  批次数据
    :return:
    """
    sources, targets = unzip(batch)
    sources = conver2id(unk, word2id, sources)
    targets = conver2id(unk, word2id, targets)
    batch = list(zip(sources, targets))
    return batch


@curry
def convert_batch_copy(unk, word2id, batch):
    """
        拆包, 处理OOV，向量化， 打包
    :param unk:
    :param word2id:  word2vec训练的结果，单词变向量
    :param batch:  批次数据
    :return:
    """
    sources, targets = map(list, unzip(batch))
    ext_word2id = dict(word2id)
    for source in sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, sources)
    sources = conver2id(unk, word2id, sources)
    tar_ins = conver2id(unk, word2id, targets)
    targets = conver2id(unk, ext_word2id, targets)
    batch = list(zip(sources, src_exts, tar_ins, targets))
    return batch


@curry
def convert_batch_extract_ptr(unk, word2id, batch):
    """
        拆包、向量化、打包
    :param unk:
    :param word2id:
    :param batch:
    :return:
    """
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        return id_sents, extracts
    batch = list(map(convert_one, batch))
    return batch


@curry
def convert_batch_extract_ff(unk, word2id, batch):
    """
        拆包、向量化，OOV ID统计，打包
    :param unk:
    :param word2id:
    :param batch:
    :return:
    """
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        binary_extracts = [0] * len(source_sents)
        for ext in extracts:
            binary_extracts[ext] = 1
        return id_sents, binary_extracts
    batch = list(map(convert_one, batch))
    return batch


@curry
def pad_batch_tensorize(inputs, pad, cuda=True):
    """
        pad_batch_tensorize
        找到最长文章，拉齐，填pad
    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len = max(len(ids) for ids in inputs)   # 找到最长那个文章
    tensor_shape = (batch_size, max_len)
    tensor = tensor_type(*tensor_shape) # 按照最长的标准构造空间
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        tensor[i, :len(ids)] = tensor_type(ids)
    return tensor


@curry
def batchify_fn(pad, start, end, data, cuda=True):
    """
        tensorizer 补码， 源长度 ， 目标加开始 tensorizer ， 目标加结束
    :param pad:
    :param start:
    :param end:
    :param data:
    :param cuda:
    :return:
    """
    sources, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]  # 源句子长度列表
    tar_ins = [[start] + tgt for tgt in targets]    # 添加开始标志
    targets = [tgt + [end] for tgt in targets]  # 添加结束标志

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)

    fw_args = (source, src_lens, tar_in)
    loss_args = (target, )
    return fw_args, loss_args



@curry
def batchify_fn_copy(pad, start, end, data, cuda=True):
    """

    :param pad:
    :param start:
    :param end:
    :param data:
    :param cuda:
    :return:
    """
    sources, ext_srcs, tar_ins, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    ext_vsize = ext_src.max().item() + 1
    fw_args = (source, src_lens, tar_in, ext_src, ext_vsize)
    loss_args = (target, )
    return fw_args, loss_args


@curry
def batchify_fn_extract_ptr(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))

    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    fw_args = (sources, src_nums, tar_in)
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_ff(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))

    tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    target = tensor_type(list(concat(targets)))

    fw_args = (sources, src_nums)
    loss_args = (target, )
    return fw_args, loss_args


def _batch2q(loader, prepro, q, single_run=True):
    """
        分批数据转换成队列的伺服进程，用途提高数据处理效率， 先装数据，后装批次
    :param loader: 数据装载器，加载数据 DataLoader(coll_fn)
    :param prepro: 预处理函数         prepro_fn
    :param q:  进程共享队列， 用于存放批次数据
    :param single_run: 是否只运行一次？
    :return:
    """
    epoch = 0
    while True:
        for batch in loader:
            q.put(prepro(batch))
        if single_run:
            break
        epoch += 1
        q.put(epoch)  # 批次结束信号，数据驱动的信号
    q.put(None)


class BucketedGenerater(object):
    """
        桶生成器，用一个进行作为伺服读数据的进程，开放预处理接口，保障数据读取的并行化操作
    """
    def __init__(self, loader, prepro, sort_key, batchify, single_run=True, queue_size=8, fork=True):
        """
        :param loader: 数据装载器，已经能对数据进行分片操作
        :param prepro: 预处理， 分批数据的预处理需要
        :param sort_key: 排序键值
        :param batchify: 批处理
        :param single_run: 执行完一个epoch就结束
        :param queue_size: 桶生成器的大小尺寸
        :param fork:   桶生成器是否需要格外的进程保障
        """
        self._loader = loader  # DataLoader
        self._prepro = prepro  # prepro_fn
        self._sort_key = sort_key
        self._batchify = batchify
        """
            batchify = compose(
                batchify_fn_copy(PAD, START, END, cuda=cuda),   # 补码
                convert_batch_copy(UNK, word2id)    # 向量化
            )
        """
        self._single_run = single_run
        if fork:
            ctx = mp.get_context('forkserver')
            self._queue = ctx.Queue(queue_size)  # 通信队列
        else:
            # for easier debugging
            self._queue = None
        self._process = None

    def __call__(self, batch_size: int):
        def get_batches(hyper_batch):
            """
                主要实现超级ｂａｔｃｈ功能
            :param hyper_batch:
            :return:
            """
            indexes = list(range(0, len(hyper_batch), batch_size))
            if not self._single_run:
                # random shuffle for training batches
                random.shuffle(hyper_batch)
                random.shuffle(indexes)
            hyper_batch.sort(key=self._sort_key)
            for i in indexes:
                batch = self._batchify(hyper_batch[i:i+batch_size])
                yield batch

        if self._queue is not None:
            ctx = mp.get_context('forkserver') # 输入数据生产者空间
            self._process = ctx.Process(
                target=_batch2q,
                args=(self._loader, self._prepro,
                      self._queue, self._single_run)
            )
            self._process.start()
            while True:
                d = self._queue.get()
                if d is None:
                    break
                if isinstance(d, int):  # 本桶水打到一个epoch整形
                    print('\nepoch {} done'.format(d))
                    continue
                yield from get_batches(d)
            self._process.join()    # 乖乖等主人
        else:
            i = 0
            while True:
                for batch in self._loader:
                    yield from get_batches(self._prepro(batch))
                if self._single_run:
                    break
                i += 1
                print('\nepoch {} done'.format(i))

    def terminate(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join()
