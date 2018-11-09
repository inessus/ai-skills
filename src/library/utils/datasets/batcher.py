""" batching """
import random
from collections import defaultdict

from toolz.sandbox import unzip
from cytoolz import curry, concat, compose

import torch
import torch.multiprocessing as mp


#1 =====================================================================================================================
# 3.1 第一步收集函数 DataLoader调用并发执行
def coll_fn(data):
    """
        收集函数， (callable, optional): merges a list of samples to form a mini-batch.
        拆包，压成一维，滤0，判优，打包  6400（）
    :param data: 是一个长度为桶长度的list 每个list 包含一对 数据集返回的值
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
        DataLoader装载数据集使用， 判优， 过滤，打包
        (
            [
                "editor 's note : in our behind the scenes series , cnn correspondents share their experiences in covering news and analyze the stories behind the events . here , soledad o'brien takes users inside a jail where many of the inmates are mentally ill .",
                "an inmate housed on the `` forgotten floor , '' where many mentally ill inmates are housed in miami before trial .", "miami , florida -lrb- cnn -rrb- -- the ninth floor of the miami-dade pretrial detention facility is dubbed the `` forgotten floor . '' here , inmates with the most severe mental illnesses are incarcerated until they 're ready to appear in court .",
                "most often , they face drug charges or charges of assaulting an officer -- charges that judge steven leifman says are usually `` avoidable felonies . '' he says the arrests often result from confrontations with police . mentally ill people often wo n't do what they 're told when police arrive on the scene -- confrontation seems to exacerbate their illness and they become more paranoid , delusional , and less likely to follow directions , according to leifman .",
                "so , they end up on the ninth floor severely mentally disturbed , but not getting any real help because they 're in jail .", "we toured the jail with leifman . he is well known in miami as an advocate for justice and the mentally ill . even though we were not exactly welcomed with open arms by the guards , we were given permission to shoot videotape and tour the floor . go inside the ` forgotten floor ' ''",
                "at first , it 's hard to determine where the people are . the prisoners are wearing sleeveless robes . imagine cutting holes for arms and feet in a heavy wool sleeping bag -- that 's kind of what they look like . they 're designed to keep the mentally ill patients from injuring themselves . that 's also why they have no shoes , laces or mattresses .",
                'leifman says about one-third of all people in miami-dade county jails are mentally ill . so , he says , the sheer volume is overwhelming the system , and the result is what we see on the ninth floor .',
                "of course , it is a jail , so it 's not supposed to be warm and comforting , but the lights glare , the cells are tiny and it 's loud . we see two , sometimes three men -- sometimes in the robes , sometimes naked , lying or sitting in their cells .",
                "`` i am the son of the president . you need to get me out of here ! '' one man shouts at me .",
                'he is absolutely serious , convinced that help is on the way -- if only he could reach the white house .',
                "leifman tells me that these prisoner-patients will often circulate through the system , occasionally stabilizing in a mental hospital , only to return to jail to face their charges . it 's brutally unjust , in his mind , and he has become a strong advocate for changing things in miami .",
                'over a meal later , we talk about how things got this way for mental patients .',
                "leifman says 200 years ago people were considered `` lunatics '' and they were locked up in jails even if they had no charges against them . they were just considered unfit to be in society .",
                'over the years , he says , there was some public outcry , and the mentally ill were moved out of jails and into hospitals . but leifman says many of these mental hospitals were so horrible they were shut down .',
                'where did the patients go ? nowhere . the streets . they became , in many cases , the homeless , he says . they never got treatment .', 'leifman says in 1955 there were more than half a million people in state mental hospitals , and today that number has been reduced 90 percent , and 40,000 to 50,000 people are in mental hospitals .',
                "the judge says he 's working to change this . starting in 2008 , many inmates who would otherwise have been brought to the `` forgotten floor '' will instead be sent to a new mental health facility -- the first step on a journey toward long-term treatment , not just punishment .",
                "leifman says it 's not the complete answer , but it 's a start . leifman says the best part is that it 's a win-win solution . the patients win , the families are relieved , and the state saves money by simply not cycling these prisoners through again and again .",
            'and , for leifman , justice is served . e-mail to a friend .'],
            [1, 3, 9, 11]
        )
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
        切分词  文章》句子》单词  把文章tokenize成指定位数的单词向量
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
    word2id = defaultdict(lambda: unk, word2id) # 超纲词全部是0
    return [[word2id[w] for w in words] for words in words_list]


#2 =====================================================================================================================
# 3.2 第二步 切词打包 BucketedGenerator调用，异步处理
@curry
def prepro_token_fn(max_src_len, max_tgt_len, batch):
    """
        对原始数据进行预处理，切分词，截断
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


#3 =====================================================================================================================
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


# 3.3 第三步 被加载器分批后的ID化 BucketedGenerator 惰性调用
@curry
def convert_id_batch_copy(unk, word2id, batch):
    """
        将一个批次32对XY 中word转换成id 对于太小的字典需要在本批次内扩充
        (
            ['-lrb-', 'cnn', '-rrb-', '--', 'harold', 'pinter', ',', 'the', 'nobel', 'prize-winning', 'playwright', 'and', 'screenwriter', 'whose', 'absurdist', 'and', 'realistic', 'works', 'displayed', 'a', 'despair', 'and', 'defiance', 'about', 'the', 'human', 'condition', ',', 'has', 'died', ',', 'according', 'to', 'british', 'media', 'reports', '.', 'he', 'was', '78', '.'],
            ['harold', 'pinter', 'died', 'on', 'christmas', 'eve', ',', 'his', 'wife', 'tells', 'british', 'media', '.']
        ) ... (32对， 已经批次化)

        (
            [55, 127, 53, 59, 9592, 1, 6, 4, 5977, 21222, 20640, 9, 20167, 687, 1, 9, 6472, 1039, 4930, 8, 9863, 9, 13095, 63, 4, 535, 750, 6, 32, 239, 6, 146, 7, 277, 392, 579, 5, 20, 13, 6783, 5],
            [55, 127, 53, 59, 9592, 30004, 6, 4, 5977, 21222, 20640, 9, 20167, 687, 30005, 9, 6472, 1039, 4930, 8, 9863, 9, 13095, 63, 4, 535, 750, 6, 32, 239, 6, 146, 7, 277, 392, 579, 5, 20, 13, 6783, 5],
            [9592, 1, 239, 17, 773, 3568, 6, 25, 254, 2023, 277, 392, 5],
            [9592, 30004, 239, 17, 773, 3568, 6, 25, 254, 2023, 277, 392, 5]
        ) ... (32对， 多加了两句)
   :param unk: 超纲词标记
    :param word2id:  word2vec训练的结果，单词变向量
    :param batch:  批次数据 32句话对
    :return:
    """
    sources, targets = map(list, unzip(batch))
    ext_word2id = dict(word2id) # 扩充字典
    for source in sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, sources) # 扩充字典后的 id化
    sources = conver2id(unk, word2id, sources)
    tar_ins = conver2id(unk, word2id, targets)
    targets = conver2id(unk, ext_word2id, targets)
    batch = list(zip(sources, src_exts, tar_ins, targets)) # 文章， 扩展文章，概括标题，扩展概括标题
    return batch


@curry
def convert_batch_extract_ptr(unk, word2id, batch):
    """
        (
           [['-lrb-', 'instyle', '-rrb-', '--', 'it', 'all', 'boils', 'down', 'to', 'this', '.', 'it', 'does', "n't", 'really', 'matter', 'all', 'that', 'much', 'what', 'hot', ',', 'nubile', 'french', 'maverick', 'has', 'set', 'the', 'fashion', 'world', 'on', 'fire', '.', 'or', 'which', 'milanese', 'visionary', 'has', 'a', 'new', 'fabric', 'technique', 'discovered', 'during', 'a', 'life-changing', 'trip', 'to', 'angkor', 'wat', 'that', "'s", 'sure', 'to', 'bring', 'back', 'sixties', 'minimalism', 'with', 'a', 'twist', '.', 'or', 'that', 'so-and-so', 'has', 'signed', 'a', 'deal', 'to', 'develop', 'boutique', 'spa', 'hotels', 'around', 'the', 'globe', 'in', 'former', 'monasteries', '.', 'because', ',', 'in', 'the', 'end', ',', 'he', "'s", 'ralph', 'lauren', ',', 'and', 'we', "'re", 'not', '.'],
            ['ralph', 'lauren', 'has', 'his', 'eye', 'on', 'china', 'and', 'japan', '.'],
            ['for', 'four', 'decades', 'no', 'other', 'designer', 'has', 'had', 'a', 'greater', 'impact', ',', 'not', 'only', 'on', 'the', 'way', 'american', 'men', 'and', 'women', 'dress', 'but', 'also', 'on', 'the', 'way', 'they', 'imagine', ',', 'seek', 'and', 'indulge', 'in', 'the', 'good', 'life', ',', 'than', 'the', 'former', 'tie', 'salesman', 'from', 'the', 'bronx', '.'],
            ['``', 'those', 'ties', 'were', 'handmade', ',', 'by', 'the', 'way', ',', "''", 'recalls', 'lauren', '.', '``', 'back', 'then', ',', 'ties', ',', 'even', 'designer', 'ones', ',', 'did', "n't", 'sell', 'for', 'more', 'than', '$', '5', 'apiece', '.', 'mine', 'were', '$', '12', 'to', '$', '15', '.', 'such', 'luxury', 'in', 'something', 'so', 'simple', 'was', 'revolutionary', '.', "''"],
            ['and', 'ironic', '.', 'because', 'while', 'no', 'other', 'designer', 'logo', 'exemplifies', 'aspiration', 'in', 'the', 'home', 'of', 'the', 'free', 'and', 'the', 'brave', 'like', 'the', 'mallet-wielding', 'guy', 'on', 'the', 'pony', ',', 'lauren', 'originally', 'named', 'his', 'company', 'polo', 'because', '``', 'it', 'was', 'the', 'sport', 'of', 'kings', '.', 'it', 'was', 'glamorous', ',', 'sexy', 'and', 'international', '.', "''", 'see', 'his', 'designs', "''"],
            ['in', 'the', 'beginning', 'a', 'few', 'people', 'questioned', 'if', 'it', 'was', 'named', 'after', 'marco', 'polo', '--', 'but', 'today', 'the', 'fact', 'that', 'virtually', 'none', 'of', 'lauren', "'s", 'millions', 'of', 'devoted', 'customers', 'has', 'ever', 'even', 'seen', 'a', 'polo', 'match', 'is', 'immaterial', '.', 'lauren', 'instinctively', 'caught', 'something', 'that', 'was', 'in', 'the', 'air', 'before', 'any', 'of', 'his', 'competitors', 'had', 'a', 'chance', 'to', 'grab', 'it', '--', 'the', 'desire', ',', 'not', 'just', 'to', 'be', 'a', 'success', 'but', 'to', 'look', 'like', 'one', 'before', 'you', "'d", 'even', 'achieved', 'your', 'goal', '.'], ['what', "'s", 'more', ',', 'lauren', 'made', 'it', 'look', 'as', 'easy', 'as', 'fred', 'astaire', 'dancing', 'down', 'a', 'staircase', '.'],
            ['``', 'what', 'matters', 'the', 'most', 'to', 'me', 'are', 'clothes', 'that', 'are', 'consistent', 'and', 'accessible', ',', "''", 'says', 'the', 'designer', '.'], ['``', 'when', 'i', 'look', 'at', 'the', 'people', 'i', "'ve", 'admired', 'over', 'the', 'years', ',', 'the', 'ultimate', 'stars', ',', 'like', 'frank', 'sinatra', ',', 'cary', 'grant', 'and', 'astaire', ',', 'the', 'ones', 'who', 'last', 'the', 'longest', 'are', 'the', 'ones', 'whose', 'style', 'has', 'a', 'consistency', ',', 'whose', 'naturalness', 'is', 'part', 'of', 'their', 'excitement', '.', 'and', 'when', 'you', 'think', 'of', 'the', 'blur', 'of', 'all', 'the', 'brands', 'that', 'are', 'out', 'there', ',', 'the', 'ones', 'you', 'believe', 'in', 'and', 'the', 'ones', 'you', 'remember', ',', 'like', 'chanel', 'and', 'armani', ',', 'are', 'the', 'ones', 'that', 'stand', 'for', 'something', '.', 'fashion', 'is', 'about', 'establishing', 'an', 'image', 'that', 'consumers', 'can', 'adapt'],
            ['however', ',', 'with', 'a', 'media', 'that', 'is', 'insatiable', 'for', 'the', 'new', ',', 'the', 'now', 'and', 'the', 'next', ',', 'being', 'steadfast', 'does', "n't", 'always', 'make', 'for', 'good', 'copy', '.'],
            ['``', 'the', 'spotlight', 'is', 'always', 'going', 'to', 'search', 'for', 'the', 'newcomer', ',', "''", 'lauren', 'admits', '.', '``', 'and', 'that', "'s", 'fine', '.', 'but', 'the', 'key', 'to', 'longevity', 'is', 'to', 'keep', 'doing', 'what', 'you', 'do', 'better', 'than', 'anyone', 'else', '.', 'we', 'work', 'real', 'hard', 'at', 'that', '.', 'it', "'s", 'about', 'getting', 'your', 'message', 'out', 'to', 'the', 'consumer', '.', 'it', "'s", 'about', 'getting', 'their', 'trust', ',', 'but', 'also', 'getting', 'them', 'excited', ',', 'again', 'and', 'again', '.', 'my', 'clothes', '--', 'the', 'clothes', 'we', 'make', 'for', 'the', 'runway', '--', 'are', "n't", 'concepts', '.', 'they', 'go', 'into', 'stores', '.', 'our', 'stores', '.', 'thankfully', ',', 'we'],
            ['``', 'what', 'i', 'rely', 'on', 'is', 'people', 'walking', 'into', 'my', 'store', 'saying', ',', '`', 'i', 'want', 'your', 'clothes', '.', "'", "''"],
            ['well', ',', 'if', 'all', 'of', 'lauren', "'s", 'customers', 'shouted', 'that', 'together', ',', 'he', 'would', 'go', 'deaf', 'faster', 'than', 'he', 'could', 'pull', 'on', 'one', 'of', 'his', 'classic', 'pullovers', '.'],
            ['lauren', "'s", 'effortless', 'luxury', 'is', 'all', 'over', 'the', 'red', 'carpet', ',', 'on', 'ski', 'slopes', 'and', 'boats', ',', 'at', 'wimbledon', 'and', 'elsewhere', '.', 'it', 'furnishes', 'living', 'rooms', 'and', 'graces', 'dinner', 'tables', '.', 'it', "'s", 'on', 'the', 'bed', ',', 'in', 'the', 'bed', 'and', 'under', 'the', 'bed', '--', 'and', 'now', 'sits', 'on', 'coffee', 'tables', ',', 'thanks', 'to', 'the', 'tome', 'ralph', 'lauren', '-lrb-', 'rizzoli', '-rrb-', ',', 'celebrating', 'his', '40-years-and-growing', 'career', '.'], ['but', 'far', 'from', 'giving', 'his', 'customary', 'over-the-head', 'wave', 'and', 'riding', 'off', 'into', 'his', 'colorado-ranch', 'sunset', ',', 'the', 'designer', 'is', 'going', 'even', 'more', 'global', '.'], ['``', 'americans', 'have', 'a', 'real', 'inferiority', 'about', 'their', 'own', 'style', '.', 'we', "'ve", 'brought', 'sportswear', 'to', 'the', 'world', ',', 'and', 'yet', 'we', 'have', 'a', 'long', 'way', 'to', 'go', '.', "''"], ['already', 'in', 'milan', ',', 'london', ',', 'paris', 'and', 'moscow', ',', 'lauren', 'has', 'more', 'stores', 'planned', 'for', 'china', ',', 'japan', '...', 'oh', ',', 'everywhere', '.', '``', 'there', 'are', "n't", 'enough', 'americans', 'out', 'there', ',', "''", 'he', 'says', '.', 'who', 'better', 'to', 'start', 'with', 'than', 'ralph', '?', 'just', 'as', 'long', 'as', 'he', 'does', "n't", 'let', 'on', 'that', 'most', 'of', 'us', 'still', 'ca', "n't", 'play', 'a', 'lick', 'of', 'polo', '.', 'e-mail', 'to', 'a', 'friend', '.'],
            ['get', 'a', 'free', 'trial', 'issue', 'of', 'instyle', '-', 'click', 'here', '!'],
            ['copyright', '©', '2007', 'time', 'inc.', '.', 'all', 'rights', 'reserved', '.']],
            [2, 3, 0, 7]
        )
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


#4 =====================================================================================================================
@curry
def pad_batch_tensorize(inputs, pad, cuda=True):
    """
        pad_batch_tensorize
        找到最长文章，拉齐，填pad ，tensor化
    :param inputs: List of size B containing torch tensors of shape [T, ...] T32
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


# 3.4 第三部 批次处理，整理向量 BucketedGenerator 惰性调用
@curry
def batchify_pad_fn_copy(pad, start, end, data, cuda=True):
    """
        针对批次数据 添加开始、结束和填充，拉齐，tensor
        copy 表示重新申请tensor内存
        (
            [55, 127, 53, 59, 9592, 1, 6, 4, 5977, 21222, 20640, 9, 20167, 687, 1, 9, 6472, 1039, 4930, 8, 9863, 9, 13095, 63, 4, 535, 750, 6, 32, 239, 6, 146, 7, 277, 392, 579, 5, 20, 13, 6783, 5],
            [55, 127, 53, 59, 9592, 30004, 6, 4, 5977, 21222, 20640, 9, 20167, 687, 30005, 9, 6472, 1039, 4930, 8, 9863, 9, 13095, 63, 4, 535, 750, 6, 32, 239, 6, 146, 7, 277, 392, 579, 5, 20, 13, 6783, 5],
            [9592, 1, 239, 17, 773, 3568, 6, 25, 254, 2023, 277, 392, 5],
            [9592, 30004, 239, 17, 773, 3568, 6, 25, 254, 2023, 277, 392, 5]
        ) ... (32对， 多加了两句)

        source, src_lens, tar_in, ext_src, ext_vsize
        32*60   32        32*11   32*60    30043

        target
tensor([[  774, 23380,  2469,  1406,  1155,    11,  3885,    11, 15956,     5, 3],
        [ 1178,   114,    32,    48,    11,  4972,   468,    22,   502,     5, 3],
        [  150,  2654,   105,  3102,   564,     7,    35, 22618,    81,    36, 3],
        ......
        [ 1490,    24,     4,   142,    50,   258,    18,   358,  3466,     5, 3],
        [  114,    30,    48,  1099,  1237,     7,  6184,    69,  7851,     5, 3],
        [ 3650, 19674,   141,    40,   709,   886,     7,    34,  2975,     5, 3]], device='cuda:0' , 32*11)


    :param pad: 填充码值
    :param start: 开始标记
    :param end: 结束标记
    :param data: 批次数据 32*(正常源ids,超纲源ids，正常目标ids，超纲目标ids)
    :param cuda: 是否转换为cuda
    :return:
    """
    sources, ext_srcs, tar_ins, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins] # 目标每句话加开始标记
    targets = [tgt + [end] for tgt in targets] # 超纲目标每句加结束标志

    source = pad_batch_tensorize(sources, pad, cuda) # 文章 tensor化
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda) # 概括标题 tensor化
    target = pad_batch_tensorize(targets, pad, cuda) # 扩展概括标题 tensor化
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda) # 扩展文章tensor化

    ext_vsize = ext_src.max().item() + 1 # 超纲字典个数30045
    fw_args = (source, src_lens, tar_in, ext_src, ext_vsize) # 文章[B,T],文长 B，概括标题[B,T'], 扩展文章[B,T],扩展字典大小
    # art_lens', 'abstract', 'extend_art', and 'extend_vsize'
    # fw_args = (src_lens, tar_in, ext_src, ext_vsize) # 文章[B,T],文长 B，概括标题[B,T'], 扩展文章[B,T],扩展字典大小
    loss_args = (target, ) # 扩展概括标题[B,T'']
    return fw_args, loss_args


@curry
def batchify_fn_extract_ptr(pad, data, cuda=True):
    """
    (
        [
            [55, 127, 53, 59, 21, 12, 48, 218, 204, 139, 3654, 22471, 3868, 76, 4, 1019, 939, 6, 37, 68, 38, 152, 1998, 1, 22, 94, 2227, 5],
            [4966, 10, 4, 170, 273, 9, 1019, 1947, 111, 946, 1984, 11, 28476, 6, 1019, 6, 96, 22471, 12, 17702, 229, 11142, 5],
            [31, 4, 2146, 10029, 6, 7, 362, 13820, 111, 24912, 38, 1747, 3637, 6, 7, 2265, 11, 4, 9227, 111, 4, 1284, 4656, 3485, 8726, 6, 127, 12, 16724, 38, 3110, 17, 4, 10125, 10, 517, 5],
            [19527, 6, 1019, 11491, 946, 9, 9247, 12021, 135, 4, 18196, 27809, 47, 4, 7960, 4999, 895, 10, 19527, 6, 99, 10, 1480, 47, 805, 7, 1633, 49, 756, 17, 348, 6, 21317, 293, 5],
            [6422, 16820, 23, 549, 4, 1025, 7, 4, 114, 12, 2861, 11518, 1438, 13, 1745, 79, 39, 83, 30, 7973, 6, 40, 152, 1393, 7, 6695, 5, 35, 50, 524, 263, 88, 1834, 128, 45, 225, 6, 36, 39, 84, 21317, 5, 35, 50, 570, 6695, 6, 9, 50, 570, 75, 21, 240, 5, 111, 1135, 64, 50, 166, 126, 45, 18, 117, 144, 5, 36],
            [37, 67, 13, 99, 2503, 14, 2067, 1, 6, 44, 13, 1992, 39, 43, 373, 680, 5, 52, 39, 401, 33, 159, 13, 1493, 6, 39, 2316, 506, 6, 3634, 8, 4961, 10, 4064, 9, 23, 24, 35, 2103, 58, 6, 1519, 5, 36, 689, 21317, 12, 228, 17, 157, 110, 7, 19527, 5],
            [2898, 992, 6, 1019, 2381, 9, 10052, 4228, 11526, 16081, 11, 4, 14186, 10, 49, 20371, 101, 11, 2898, 992, 27, 3654, 22471, 5157, 4, 159, 2164, 59, 6177, 3069, 62, 7, 1006, 9, 6406, 60, 87, 1748, 112, 7, 6843, 21, 104, 6, 21317, 23, 5, 509, 4, 1, 754, 49, 1681, 3988, 612, 36],
            [4680, 4, 2826, 2540, 17, 9, 40, 2184, 6, 37, 40, 38, 42, 100, 805, 110, 76, 4, 8866, 6, 42, 137, 7, 9825, 4, 6168, 10, 49, 429, 315, 5],
            [4, 325, 23, 40, 47, 4423, 22, 2257, 29, 68, 396, 128, 79, 93, 5, 35, 50, 162, 71, 10, 168, 68, 16, 38, 3317, 9, 23760, 108, 40, 214, 99, 5947, 17, 49, 1219, 65, 49, 2383, 38, 62, 11, 2854, 6, 36, 2381, 11526, 84, 21317, 5], [35, 50, 75, 54, 162, 618, 63, 117, 310, 16, 239, 11, 2898, 992, 9, 63, 4, 630, 16, 117, 107, 18, 2885, 2528, 5, 28, 222, 1273, 88, 372, 9, 28, 30, 7, 423, 91, 144, 71, 85, 318, 5, 36, 689, 121, 4, 1, 2333, 21317, 52, 40, 118, 40, 83, 54, 1874, 5], [1, 532, 6, 1019, 16851, 918, 2202, 1, 532, 9, 4637, 7, 4083, 96, 4, 1284, 5, 39, 152, 216, 42, 177, 353, 39, 32, 8, 101, 7, 546, 7, 6, 1, 23, 5],
            [35, 21, 12, 102, 8, 4600, 5, 58, 279, 26, 165, 159, 52, 58, 3910, 60, 7, 543, 9, 58, 201, 102, 15, 301, 28, 157, 7, 162, 45, 318, 19, 126, 36, 39, 23, 5],
            [130, 95, 39, 12, 95, 9678, 17, 3912, 9, 80, 370, 21, 143, 274, 125, 5, 35, 50, 75, 54, 137, 177, 86, 50, 201, 157, 7, 30, 4, 267, 7, 442, 117, 2629, 5, 21, 12, 16, 753, 5, 91, 394, 32, 54, 48, 334, 7, 156, 139, 50, 113, 6, 36, 39, 23, 5, 689, 1, 12, 228, 17, 121, 4083, 895, 9, 763, 38, 402, 7, 188, 5],
            [4083, 6, 1019, 327, 152, 2405, 26, 4, 4083, 4277, 610, 38, 402, 7, 725, 811, 14, 49, 951, 6, 230, 898, 293, 6, 37, 40, 141, 21, 18, 466, 3731, 5, 509, 3742, 283, 1111, 2270, 644, 7, 188, 24912, 36], [1175, 1197, 23, 4, 6605, 189, 13, 402, 7, 874, 7, 25, 72, 2134, 63, 49, 101, 5],
            [35, 28, 80, 754, 93, 4, 2196, 9, 411, 21, 1795, 6, 36, 20, 23, 5, 35, 28, 75, 54, 607, 7, 1, 21, 6, 78, 40, 177, 45, 18, 8, 378, 700, 847, 6, 37, 21, 12, 80, 541, 17, 2183, 130, 95, 6, 36, 20, 84, 4, 786, 5, 689, 87, 237, 633, 38, 1299, 230, 898, 5],
            [1081, 4289, 6, 1019, 1973, 943, 12, 1337, 13, 17, 10197, 27, 20, 524, 164, 125, 11, 8, 3637, 11, 1081, 4289, 6, 1019, 6, 46, 466, 62, 10, 2854, 7, 1361, 22471, 5], [35, 28, 232, 48, 6177, 168, 1843, 62, 9, 1495, 93, 17, 246, 125, 6, 36, 20, 84, 1, 5],
            [4, 14410, 810, 23, 67, 64, 34, 471, 1843, 7, 685, 62, 7, 24912, 6, 37, 67, 47, 1185, 63, 370, 660, 5985, 13, 1738, 6, 1, 23, 5, 689, 1, 228, 17, 121, 8, 1393, 2579, 5909, 13, 1134, 5],
            [9298, 6, 2406, 24016, 38, 749, 1505, 11, 9298, 137, 27, 907, 10, 1001, 763, 607, 7, 119, 4, 488, 110, 17, 6, 1, 293, 5],
            [2710, 8979, 1, 84, 4, 786, 39, 13, 35, 245, 1, 128, 36, 16, 2110, 1051, 290, 10, 33, 43, 488, 6, 37, 39, 152, 799, 8, 15087, 80, 7, 145, 2632, 5], [35, 40, 201, 130, 240, 8, 153, 46, 58, 442, 165, 642, 7, 689, 165, 1, 11623, 14, 196, 248, 6, 36, 39, 23, 5, 35, 40, 201, 2760, 7, 751, 58, 128, 86, 58, 75, 54, 442, 21, 6, 37, 83, 40, 379, 464, 16, 40, 75, 54, 214, 4, 3510, 17, 5, 21, 75, 54, 145, 1265, 6, 58, 177, 126, 36], [2581, 1001, 84, 1, 16, 99, 8073, 47, 1398, 112, 8, 494, 259, 10, 946, 37, 16, 21, 2623, 7, 30, 347, 110, 7, 1402, 29, 292, 5, 689, 1, 12, 228, 17, 121, 68, 38, 466, 2102, 5],
            [12299, 6, 2406, 82, 11, 12299, 38, 14544, 8, 20419, 9511, 7, 1467, 12628, 9, 98, 720, 106, 4, 488, 18, 62, 6, 286, 84, 1, 5],
            [4, 554, 13, 224, 108, 10, 690, 2727, 11, 4, 323, 363, 29, 4, 1284, 6, 4, 786, 23, 5],
            [895, 84, 1, 40, 103, 54, 102, 4, 3969, 1224, 37, 47, 14, 4, 2313, 86, 21, 891, 720, 104, 5, 689, 1, 228, 17, 4, 554, 17, 12299, 5],
            [14669, 6, 69, 1208, 8, 114, 44, 2202, 19527, 32, 307, 60, 17, 407, 157, 101, 6, 666, 5615, 7, 145, 14669, 49, 69, 1333, 6, 1, 293, 5],
            [5646, 3825, 9, 33, 72, 136, 299, 40, 523, 34, 334, 7, 2170, 62, 4, 1284, 9, 2436, 4075, 11, 2854, 37, 52, 22471, 193, 62, 4, 488, 9, 4, 343, 40, 727, 7, 278, 7, 61, 114, 11, 69, 1208, 5],
            [35, 21, 12, 396, 52, 58, 177, 569, 135, 58, 6, 36, 3825, 84, 1, 5, 689, 121, 1, 18, 402, 7, 188, 4, 3825, 114, 5]
        ],
        [6, 12, 0]
    )
    :param pad:
    :param data:
    :param cuda:
    :return:
    """
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
    :param prepro: 预处理函数         prepro_token_fn
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
        :param prepro: 预处理， 切分词
        :param sort_key: 排序键值
        :param batchify: 批处理
        :param single_run: 执行完一个epoch就结束
        :param queue_size: 桶生成器的大小尺寸
        :param fork:   桶生成器是否需要格外的进程保障
        """
        self._loader = loader  # DataLoader
        self._prepro = prepro  # prepro_token_fn
        self._sort_key = sort_key
        self._batchify = batchify
        """
            batchify = compose(
                batchify_pad_fn_copy(PAD, START, END, cuda=cuda),   # 补码
                convert_id_batch_copy(UNK, word2id)    # 向量化
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
                for batch in self._loader: # 集装箱装载6400
                    yield from get_batches(self._prepro(batch))
                if self._single_run:
                    break
                i += 1
                print('\nepoch {} done'.format(i))

    def terminate(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join()
