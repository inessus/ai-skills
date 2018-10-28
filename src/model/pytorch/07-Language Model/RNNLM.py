# 1 单词ID化 将数据中以单词为单位，将单词转化为id
# 2 每句尾部添加标志符号 然后构建一个长向量序列
# 3 句子ID化 将使用单词id代替 长向量序列的单词
# 4 单词向量化， 每个单词ID映射到指定维度向量

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm
from datetime import datetime
import os

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if not word in self.word2idx:  # 字典默认检索keys
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)
    
class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
        self.ids = None
    
    def get_data(self, path, batch_size=20):
        #Add words to the dictionary  将单词向量化，这样可以大大降低数据维度，单词个数毕竟是有限的
        with open(path, 'r') as f:
            tokens = 0
            for line in f:  # 读取每一行
                words = line.split() + ['<eos>'] # 获取每个单词，在句子后面添加<eos>
                tokens += len(words) # 单词总数计数（含<eos>）
                for word in words:
                    self.dictionary.add_word(word)
        # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        self.ids = ids
        num_batches = ids.size(0) // batch_size  # 切分成批次
        ids = ids[:num_batches*batch_size]       # 
        return ids.view(batch_size, -1)



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start = datetime.now()
loopstart = start

# Hyper-parameters
embed_size = 128   # 向量大小
hidden_size = 1024 # 隐层尺寸
num_layers = 1
num_epochs = 5
num_samples = 1000 # number of words to be sampled
batch_size = 20
seq_length = 30   # 进入模型的长度
learning_rate = 0.002

corpus = Corpus()

ids = corpus.get_data(r'./src/model/pytorch/07-Language Model/data/trainer.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length

#RNN based language model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # 1万个单词， 维度128
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, h):
        #Embed word ids to vectors
        x = self.embed(x)
        
        #Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        
        #Reshape output to (batch_size * sequence_length, hidden_size)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        
        #Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)
    

model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]

# Train the model
for epoch in range(num_epochs):
    # Set initial hidden and cell_states
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))
    for i in range(0, ids.size(1) - seq_length, seq_length):
        # Get mini-batch inputs and targets
        inputs = ids[:, i:i+seq_length].to(device)  # 列向量按照seq_length划分成每个序列 作为mini-batch作为输入
        targets = ids[:, (i+1):(i+1)+seq_length].to(device) # 前后两个序列间存在关联，视为目标

        # Forward pass
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1) // seq_length

        if step % 100 == 0:
            cur = datetime.now()
            print("Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity {:5.2f} {}"
                  .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item()), cur-loopstart))
            loopstart = cur
                    
with torch.no_grad():
    with open('sample.txt', 'w') as f:
        # Set initial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                torch.zeros(num_layers, 1, hidden_size).to(device))
        
        # Select one word id randomly
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            # Forward propagate RNN 
            output, state = model(input, state)

            # Sample a word id
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # Fill input with sampled word id for the next time step
            input.fill_(word_id)

            # File write
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))

#model.load_state_dict(torch.load("model.ckpt"))
# Save the model checkpoints
torch.save(model.state_dict(), 'model.ckpt')