

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 10  # Maximum sentence length to consider


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}  # 单词的token
        self.word2count = {}  # 单词的计数
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}  # ID 代表的单词
        self.num_words = 3  # Count SOS, EOS, PAD

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

    # Remove words below a certain count threhold
    def trim(self, min_count):
        """
            当单词频率值时，表明单词太生僻了，需要修剪掉，
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
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)

