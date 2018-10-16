import re
import torch
import itertools
import unicodedata
from cytoolz import take

from .dictionary import Voc, MAX_LENGTH, PAD_token, EOS_token, SOS_token


def printLines(file, n=10):
    """
        打印文件的函数。相当于head命令
    :param file: 要读入的文件
    :param n:  打印的最大长度
    :return:
    """
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


def fileFilter(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
        return list(take(n, lines))


# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    """
        把文件按照关键字，组成二级字典
    :param fileName: 要处理的文件
    :param fields:  文件中的字典关键字
    :return:
    """
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            value = line.split(" +++$+++")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = value[i]

            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines form readLines into conversations based on movie_conversations.txt
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            lineIds = eval(convObj['utteranceIDs'])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    """
        从谈话数据集中提取对话数据列表
    :param conversations: 谈话数据集，
    :return:[(q0,a0),(q1,a1),....,(qn, an)]
    """
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the line of the conversation
        for i in range(len(conversation["lines"]) - 1):  #
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i + 1]["text"].strip()
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


def shave_marks(txt):
    """去掉全部变音符号"""
    norm_txt = unicodedata.normalize('NFD', txt)  # 把所有字符分解成基字符和组合记号
    shaved = ''.join(c for c in norm_txt if not unicodedata.combining(c))  # 过滤掉所有组合记号。
    return unicodedata.normalize('NFC', shaved)  # 重组所有字符


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'  # Nonspacing_Mark 非间距组合字符
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    """
        标准化字符串， 小写、修剪、字符转换、去非字母
    :param s:
    :return:
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    """
        读取datafile的文件，unicode标准化后加入字典
    :param datafile:
    :param corpus_name:
    :return:
    """
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    """
        给出一对句子，判断长度小于预定阈值
    :param p:
    :return:
    """
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH


# Filter pairs using filterPair condition
def filterPairs(pairs):
    """
        过滤一篇文章
    :param pairs:
    :return:
    """
    return [pair for pair in pairs if filterPair(pair)]


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    """

    :param corpus: 语料库目录
    :param corpus_name: 语料库名称
    :param datafile: 整理的文件格式
    :param save_dir:
    :return:
    """
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])  # 源句子
        voc.addSentence(pair[1])  # 目标句子
    print("Counted words:", voc.num_words)
    return voc, pairs


def trimRareWords(voc, pairs, MIN_COUNT):
    """

    :param voc: 字典
    :param pairs:  整理的token
    :param MIN_COUNT: 裁剪的最小阈值
    :return:
    """
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence  单词在不在字典中
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


def indexesFromSentence(voc, sentence):
    """
        将每句话，通过字典，转换成ID
    :param voc: 映射字典
    :param sentence: 句子
    :return:
    """
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    """
        求掩码矩阵
    :param l:
    :param value:
    :return:
    """
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    """
        5条拍好序的语句，向量化，补齐，求每条语句长度，Tensor
    :param l: 输入语句
    :param voc: 字典
    :return:
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    """
        5条排好序的语句，向量化，求最大长度，补齐，求掩码矩阵
    :param l:
    :param voc:
    :return:
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    """
        批量语句，本题5句， 按照长度排序，分类input和output

    :param voc: 修正的字典
    :param pair_batch: 批处理对
    :return:
    """
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len
