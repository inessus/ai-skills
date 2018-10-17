import re
import unicodedata
import pickle as pkl

# Default word tokens
PAD = 0     # Used for padding short sentences
UNK = 1
START = 2   # Start-of-sentence token
END = 3     # End-of-sentence token
MAX_LENGTH = 10  # Maximum sentence length to consider


class VocV1:
    def __init__(self, name):
        """
            词汇表
        :param name:
        """
        self.name = name
        self.trimmed = False
        self.word2index = {}  # 单词的token
        self.word2count = {}  # 单词的计数
        self.num_words = 0  # Count PAD, UNK, START, END
        self.index2word = {}  # ID 代表的单词
        self.init()

    def init(self):
        self.word2index = {'<pad>': PAD, '<unk>': UNK, '<start>': START, '<end>': END}  # 单词的token
        self.word2count = {}  # 单词的计数
        self.index2word = {PAD: "<pad>", UNK: '<unk>', START: "<start>", END: "<end>"}  # ID 代表的单词
        self.num_words = 4  # Count SOS, EOS, PAD
        self.trimmed = False

    def addArticle(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        sentences = [[self.normalizeString(s) for s in l.split('\t')] for l in lines]
        self.init()
        for sentence in sentences:
            self.addSentence(sentence)

    def addSentence(self, sentence):
        """
            添加一个句子
        :param sentence: 一整个句子
        :return:
        """
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """
            字典加单词
        :param word:
        :return:
        """
        if word not in self.word2index:  # 不在字典中
            self.word2index[word] = self.num_words  # 分配编号
            self.word2count[word] = 1  # 单词计数
            self.index2word[self.num_words] = word  # ID到单词
            self.num_words += 1  # 词库个数增加
        else:
            self.word2count[word] += 1  # 已经添加过，记录计数

    def trim_by_count(self, min_count):
        """
            当单词频率值时，表明单词太生僻了，需要修剪掉，
            Remove words below a certain count threhold
        :param min_count:
        :return:
        """
        if self.trimmed:  # 不重复修剪
            return

        self.trimmed = True

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.init()

        for word in keep_words:
            self.addWord(word)

    def make_vocab(self, wc_path, vocab_size):
        """
            单词频率文件，创建建立词汇表
        :param wc:
        :param vocab_size:
        :return:
        """
        with open(wc_path, 'rb') as f:
            wc = pkl.load(f)
            self.init()
            for (w, _) in wc.most_common(vocab_size):
                self.addWord(w)

        return self.word2index

    def shave_marks(self, txt):
        """
            去掉全部变音符号
        :param txt:
        :return:
        """
        norm_txt = unicodedata.normalize('NFD', txt)  # 把所有字符分解成基字符和组合记号
        shaved = ''.join(c for c in norm_txt if not unicodedata.combining(c))  # 过滤掉所有组合记号。
        return unicodedata.normalize('NFC', shaved)  # 重组所有字符

    def unicodeToAscii(self, s):
        """
        Turn a Unicode string to plain ASCII, thanks to
        http://stackoverflow.com/a/518232/2809427
        :param s:
        :return:
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'  # Nonspacing_Mark 非间距组合字符
        )

    def normalizeString(self, s):
        """
            标准化字符串， 小写、修剪、字符转换、去非字母
            Lowercase, trim, and remove non-letter characters
        :param s:
        :return:
        """
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s