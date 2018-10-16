import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attn


cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')


class EncoderRNN(nn.Module):
    """
    INPUT:
        input_seq : 分批的输入序列 shape=(max_length, batch_size)
        input_lengths: 批次列表中句子的长度 shape = (batch_size)
        hidden: 隐藏状态，此例中表示语义向量 shape=(n_layers, batch_size, hidden_size)
    OUTPUT:
        outputs: gru最后的隐层中输出的特征，（双向输出之和），shape=(max_length, batch_size, hidden_size)
        hidden: GRU更新状态 shape=(n_layers, batch_size, hidden_size)
    """

    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        """

        :param input_seq: (T, B) (10, 64)
        :param input_lengths: (B)
        :param hidden:
        :return:
        """
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)    # (T, B, N)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)  # [(no zero number, N), (T)][(375, N),(T)]
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden) # ((I, N*2),(T)) (L*2,B,N)   ((375,1000),(10)) (4, 64, 500)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs) # (T, B, N*2)(10, 64, 10000)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] # (T, B, N)(10, 64, 500)
        # Return output and final hidden state
        return outputs, hidden


class LuongAttnDecoderRNN(nn.Module):
    """

        带注意力的解码器
    """

    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        """

        :param attn_model:
        :param embedding:
        :param hidden_size:
        :param output_size:
        :param n_layers:
        :param dropout:
        """
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model  # 注意力方法
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)  # 生成注意力

    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        :param input_step: (1, B) (1, 64)
        :param last_hidden: (L, B, N) (2, 64, 500)
        :param encoder_outputs: (T, B, N) (10, 64, 500)
        :return:
        """
        """此处，输入多加入了编码器的状态太"""
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded) # (1, B, N) (1, 64, 500)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden) # (B, N) (L, B, N) (64, 500)(2, 64, 500)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs) # B, 1, T (64, 1, 10)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        """

        :param input_seq:
        :param input_length:
        :param max_length:
        :return:
        """
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

