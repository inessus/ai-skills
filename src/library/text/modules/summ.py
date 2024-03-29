import torch
from torch import nn
from torch.nn import init

from library.text.modules.base.rnn import lstm_encoder, MultiLayerLSTMCells
from library.text.modules.base.attention import step_attention
from library.utils.transforms.sequence import sequence_mean, len_mask


INIT = 1e-2


class Seq2SeqSumm(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_hidden, bidirectional, n_layer, dropout=0.0):
        """
            Seq2Seq 网络 带注意力的网络
        :param vocab_size:  字典大小 V
        :param emb_dim: 嵌入尺寸  E
        :param n_hidden: 隐藏层大小 N
        :param bidirectional: 是否双向
        :param n_layer: 层数 L
        :param dropout: dropout层参数
        """
        super().__init__()
        # embedding weight parameter is shared between encoder, decoder,
        # and used as final projection layer to vocab logit
        # can initialize with pretrained word vectors
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)  # (V, E)  会被替换掉
        self._enc_lstm = nn.LSTM(emb_dim, n_hidden, n_layer, bidirectional=bidirectional, dropout=dropout)  # (128*256*1, True)

        # initial encoder LSTM states are learned parameters
        state_layer = n_layer * (2 if bidirectional else 1)
        self._init_enc_h = nn.Parameter(torch.Tensor(state_layer, n_hidden))    # ((2*L)*N) (2*256)
        self._init_enc_c = nn.Parameter(torch.Tensor(state_layer, n_hidden))    # ((2*L)*N) (2*256)
        init.uniform_(self._init_enc_h, -INIT, INIT)
        init.uniform_(self._init_enc_c, -INIT, INIT)

        # vanillat lstm / LNlstm
        self._dec_lstm = MultiLayerLSTMCells(2*emb_dim, n_hidden, n_layer, dropout=dropout) # (2*E,N,L)(2*128, 256, 1)

        # project encoder final states to decoder initial states
        enc_out_dim = n_hidden * (2 if bidirectional else 1) # (2*N)
        self._dec_h = nn.Linear(enc_out_dim, n_hidden, bias=False)  # (2*N, N) (512, 256)
        self._dec_c = nn.Linear(enc_out_dim, n_hidden, bias=False)  # (2*N, N) (512, 256)

        # multiplicative attention
        self._attn_wm = nn.Parameter(torch.Tensor(enc_out_dim, n_hidden))   # (512, 256)
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))      # (256, 256)
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)

        # project decoder output to emb_dim, then
        # apply weight matrix from embedding layer
        self._projection = nn.Sequential(
            nn.Linear(2*n_hidden, n_hidden),            # (2*256, 256)
            nn.Tanh(),
            nn.Linear(n_hidden, emb_dim, bias=False)    # (256, 128)
        )
        # functional object for easier usage   注意力解码器，或者被CopyNet解码替换掉
        self._decoder = AttentionalLSTMDecoder(
            self._embedding, self._dec_lstm,
            self._attn_wq, self._projection
        )

    def forward(self, article, art_lens, abstract):
        # [128*32*256], ( [(1*128*256),(1*128*256)],(256*128))
        attention, init_dec_states = self.encode(article, art_lens)

        mask = len_mask(art_lens, attention.device).unsqueeze(-2)

        # ([128*32*256], (XXXXXX)), ( [(1*128*256),(1*128*256)],(256*128)) , (32*128)
        logit = self._decoder((attention, mask), init_dec_states, abstract)
        return logit

    def encode(self, article, art_lens=None):
        """
            针对数据的文章tensor进行编码
        :param article: ROUGE优选文章，[B,T] 每个批次Tensor化的时候长度不同，最大T值为100
        :param art_lens:
        :return:
        """
        #
        size = (self._init_enc_h.size(0), len(art_lens) if art_lens else 1, self._init_enc_h.size(1)) # 2，32，256

        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )

        # 32, tunnel L_size, 256
        enc_art, final_states = lstm_encoder(article, self._enc_lstm, art_lens,init_enc_states, self._embedding)
        if self._enc_lstm.bidirectional:
            h, c = final_states
            final_states = (
                torch.cat(h.chunk(2, dim=0), dim=2),
                torch.cat(c.chunk(2, dim=0), dim=2)
            ) # 将两层网络的值在dim2上拼接 双向神经网络的状态拼接([L,B,2N],(L, B, 2N)]
        init_h = torch.stack([self._dec_h(s) for s in final_states[0]], dim=0) # [L,B,N]
        init_c = torch.stack([self._dec_c(s) for s in final_states[1]], dim=0) # [L,B,N]
        init_dec_states = (init_h, init_c)
        attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1)    # [T,B,2N]*[2N,N] =[B,T,N]'(32, 128, 256)X(256, 256)=[128, 32, 256]

        # (256, 128)
        init_attn_out = self._projection(torch.cat([init_h[-1], sequence_mean(attention, art_lens, dim=1)], dim=1))

        # [128, 32, 256], ( [(1, 128, 256),(1, 128, 256)],(256, 128))
        return attention, (init_dec_states, init_attn_out)

    def batch_decode(self, article, art_lens, go, eos, max_len):
        """

             greedy decode support batching
        :param article:
        :param art_lens:
        :param go:
        :param eos:
        :param max_len:
        :return:
        """
        batch_size = len(art_lens)
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask)
        tok = torch.LongTensor([go]*batch_size).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            outputs.append(tok[:, 0])
            attns.append(attn_score)
        return outputs, attns

    def decode(self, article, go, eos, max_len):
        """

        :param article:
        :param go:
        :param eos:
        :param max_len:
        :return:
        """
        attention, init_dec_states = self.encode(article)
        attention = (attention, None)
        tok = torch.LongTensor([go]).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
        return outputs, attns

    def set_embedding(self, embedding):
        """
            使用word2vec，等替换嵌入层
        embedding is the weight matrix
        :param embedding:
        :return:
        """
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)


class AttentionalLSTMDecoder(object):
    def __init__(self, embedding, lstm, attn_w, projection):
        """
            注意力解码器
        :param embedding:  # (30004, 128)
        :param lstm: (256, 256, 1) 解码器 LSTM 网络
        :param attn_w: (256, 256)  注意力参数
        :param projection: [521, 256]=>[256, 128] 映射矩阵
        """
        super().__init__()
        self._embedding = embedding
        self._lstm = lstm
        self._attn_w = attn_w   # 解码器输出　权值
        self._projection = projection

    def __call__(self, attention, init_states, target):
        """
            注意力的解码极值，针对每一个输出的对象，计算相应的概率
        :param attention: (注意力[B,T,N]，掩码[B,1,T]，扩展文章[B, T]，扩展字典尺寸)
        :param init_states:(([L,B,N],[L,B,N]),[B,E]) ( [(1, 128, 256), (1, 128, 256)],(256, 128)) ,
        :param target: 概括标题[B, T'] (32, 7)
        :return:
        """
        max_len = target.size(1)
        states = init_states
        logits = []
        for i in range(max_len): # 对于所有要解码长度,依次生成
            tok = target[:, i:i+1] # [B,1]
            logit, states, _ = self._step(tok, states, attention)
            logits.append(logit)
        logit = torch.stack(logits, dim=1) # [B,T',V']
        return logit

    def _step(self, tok, states, attention):
        """

        :param tok: 概括标题的一个单词id
        :param states: (([L,B,N],[L,B,N]),[B,E]) ( [(1, 128, 256), (1, 128, 256)],(256, 128))
        :param attention: (注意力[B,T,N]，掩码[B,1,T]，扩展文章[B, T]，扩展字典尺寸)
        :return:
        """
        prev_states, prev_out = states
        lstm_in = torch.cat([self._embedding(tok).squeeze(1), prev_out], dim=1) # [128 + 128, 128]
        states = self._lstm(lstm_in, prev_states)
        lstm_out = states[0][-1]
        query = torch.mm(lstm_out, self._attn_w)    # (32 * 128)*(128, 256)
        attention, attn_mask = attention
        context, score = step_attention(query, attention, attention, attn_mask)
        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))
        states = (states, dec_out)
        logit = torch.mm(dec_out, self._embedding.weight.t())
        return logit, states, score

    def decode_step(self, tok, states, attention):
        """

        :param tok:
        :param states:
        :param attention:
        :return:
        """
        logit, states, score = self._step(tok, states, attention)
        out = torch.max(logit, dim=1, keepdim=True)[1]
        return out, states, score
