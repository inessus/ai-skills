'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import torch
import torch.nn.functional as F
from torch.autograd import Variable
'''
Bahdanau, D., Cho, K., & Bengio, Y. (2014). 
Neural machine translation by jointly learning to align and translate. 
arXiv preprint arXiv:1409.0473.
Luong, M. T., Pham, H., & Manning, C. D. (2015). 
Effective approaches to attention-based neural machine translation. 
arXiv preprint arXiv:1508.04025.
Paulus, R., Xiong, C., & Socher, R. (2017). 
A deep reinforced model for abstractive summarization. 
arXiv preprint arXiv:1705.04304.
See, A., Liu, P. J., & Manning, C. D. (2017). 
Get To The Point: Summarization with Pointer-Generator Networks. 
arXiv preprint arXiv:1704.04368.
'''
class AttentionEncoder(torch.nn.Module):
    
    def __init__(
        self,
        hidden_size,
        attn_method,
        coverage,
    ):
        super(AttentionEncoder, self).__init__()
        self.method = attn_method.lower()
        self.hidden_size = hidden_size
        self.coverage = coverage
        
        if self.method == 'luong_concat':
            self.attn_en_in = torch.nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=True).cuda()
            self.attn_de_in = torch.nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=False).cuda()
            self.attn_cv_in = torch.nn.Linear(1, self.hidden_size, bias=False).cuda()
            self.attn_warp_in = torch.nn.Linear(self.hidden_size, 1, bias=False).cuda()
        if self.method == 'luong_general':
            self.attn_in = torch.nn.Linear(
                self.hidden_size, 
                self.hidden_size,
                bias=False).cuda()

    def forward(self, dehy, enhy, past_attn):
        # attention score
        if self.method == 'luong_concat':
            attn_agg = self.attn_en_in(enhy) + self.attn_de_in(dehy.unsqueeze(1))
            if self.coverage[:4] == 'asee':
                attn_agg = attn_agg + self.attn_cv_in(past_attn.unsqueeze(2))
            attn_agg = F.tanh(attn_agg)
            attn_ee = self.attn_warp_in(attn_agg).squeeze(2)
        else:
            if self.method == 'luong_general':
                enhy_new = self.attn_in(enhy)
                attn_ee = torch.bmm(enhy_new, dehy.unsqueeze(2)).squeeze(2)
            else:
                attn_ee = torch.bmm(enhy, dehy.unsqueeze(2)).squeeze(2)
        # coverage and attention weights
        if self.coverage == 'temporal':
            attn_ee = torch.exp(attn_ee)
            attn = attn_ee/past_attn
            nm = torch.norm(attn, 1, 1).unsqueeze(1)
            attn = attn/nm
        else:
            attn = F.softmax(attn_ee, dim=1)
        # context vector
        attn2 = attn.unsqueeze(1)
        c_encoder = torch.bmm(attn2, enhy).squeeze(1)
        
        return c_encoder, attn, attn_ee
'''
Intra-decoder

Paulus, R., Xiong, C., & Socher, R. (2017). 
A deep reinforced model for abstractive summarization. 
arXiv preprint arXiv:1705.04304.
'''
class AttentionDecoder(torch.nn.Module):
    
    def __init__(
        self,
        hidden_size,
        attn_method
    ):
        super(AttentionDecoder, self).__init__()
        self.method = attn_method.lower()
        self.hidden_size = hidden_size
        
        if self.method == 'luong_concat':
            self.attn_en_in = torch.nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=True).cuda()
            self.attn_de_in = torch.nn.Linear(
                self.hidden_size,
                self.hidden_size,
                bias=False).cuda()
            self.attn_warp_in = torch.nn.Linear(self.hidden_size, 1, bias=False).cuda()
        if self.method == 'luong_general':
            self.attn_in = torch.nn.Linear(
                self.hidden_size, 
                self.hidden_size,
                bias=False).cuda()

    def forward(self, dehy, past_hy):
        # attention score
        if self.method == 'luong_concat':
            attn_agg = self.attn_en_in(past_hy) + self.attn_de_in(dehy.unsqueeze(1))
            attn_agg = F.tanh(attn_agg)
            attn = self.attn_warp_in(attn_agg).squeeze(2)
        else:
            if self.method == 'luong_general':
                past_hy_new = self.attn_in(past_hy)
                attn = torch.bmm(past_hy_new, dehy.unsqueeze(2)).squeeze(2)
            else:
                attn = torch.bmm(past_hy, dehy.unsqueeze(2)).squeeze(2)
        attn = F.softmax(attn, dim=1)
        # context vector
        attn2 = attn.unsqueeze(1)
        c_decoder = torch.bmm(attn2, past_hy).squeeze(1)
        
        return c_decoder, attn
'''
LSTM decoder
'''    
class LSTMDecoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        attn_method,
        coverage,
        batch_first,
        pointer_net,
        attn_decoder
    ):
        super(LSTMDecoder, self).__init__()
        # parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = num_layers
        self.batch_first = batch_first
        self.attn_method = attn_method.lower()
        self.coverage = coverage
        self.pointer_net = pointer_net
        self.attn_decoder = attn_decoder
        
        self.lstm_ = torch.nn.LSTMCell(
            self.input_size+self.hidden_size, 
            self.hidden_size).cuda()
        self.encoder_attn_layer = AttentionEncoder(
            hidden_size=self.hidden_size,
            attn_method=self.attn_method, 
            coverage=self.coverage).cuda()
        # intra-decoder
        if self.attn_decoder:           
            self.decoder_attn_layer = AttentionDecoder(
                hidden_size=self.hidden_size,
                attn_method=self.attn_method).cuda()
            self.attn_out = torch.nn.Linear(
                self.hidden_size*3,
                self.hidden_size,
                bias=True).cuda()
        else:
            self.attn_out = torch.nn.Linear(
                self.hidden_size*2,
                self.hidden_size,
                bias=True).cuda()
        # pointer generator network
        if self.pointer_net:
            if self.attn_decoder:   
                self.pt_out = torch.nn.Linear(
                    self.input_size+self.hidden_size*3, 1).cuda()
            else:
                self.pt_out = torch.nn.Linear(
                    self.input_size+self.hidden_size*2, 1).cuda()
        
    def forward(
        self, idx, input_, hidden_, h_attn, 
        encoder_hy, past_attn, p_gen, past_dehy):
        
        if self.batch_first:
            input_ = input_.transpose(0,1)
        batch_size = input_.size(1)
        
        output_ = []
        out_attn = []
        
        loss_cv = Variable(torch.zeros(1)).cuda()
        batch_size = input_.size(1)
        for k in range(input_.size(0)):
            x_input = torch.cat((input_[k], h_attn), 1)
            hidden_ = self.lstm_(x_input, hidden_)
            # attention encoder
            c_encoder, attn, attn_ee = self.encoder_attn_layer(
                hidden_[0], encoder_hy, past_attn)
            # attention decoder
            if self.attn_decoder:
                if k + idx == 0:
                    c_decoder = Variable(torch.zeros(
                        batch_size, self.hidden_size)).cuda()
                else:
                    c_decoder, attn_de = self.decoder_attn_layer(
                        hidden_[0], past_dehy)
                past_dehy = past_dehy.transpose(0, 1) # seqL*batch*hidden
                de_idx = past_dehy.size(0)
                if k + idx == 0:
                    past_dehy = hidden_[0].unsqueeze(0) # seqL*batch*hidden
                    past_dehy = past_dehy.transpose(0, 1) # batch*seqL*hidden
                else:
                    past_dehy = past_dehy.contiguous().view(-1, self.hidden_size) # seqL*batch**hidden
                    past_dehy = torch.cat((past_dehy, hidden_[0]), 0) # (seqL+1)*batch**hidden
                    past_dehy = past_dehy.view(de_idx+1, batch_size, self.hidden_size) # (seqL+1)*batch*hidden
                    past_dehy = past_dehy.transpose(0, 1) # batch*(seqL+1)*hidden
                h_attn = self.attn_out(torch.cat((c_encoder, c_decoder, hidden_[0]), 1))
            else:
                h_attn = self.attn_out(torch.cat((c_encoder, hidden_[0]), 1))
            # coverage
            if self.coverage == 'asee_train':
                lscv = torch.cat((past_attn.unsqueeze(2), attn.unsqueeze(2)), 2)
                lscv = lscv.min(dim=2)[0]
                try:
                    loss_cv = loss_cv + torch.mean(lscv)
                except:
                    loss_cv = torch.mean(lscv)
            if self.coverage[:4] == 'asee':
                past_attn = past_attn + attn
            if self.coverage == 'temporal':
                if k + idx == 0:
                    past_attn = past_attn*0.0
                past_attn = past_attn + attn_ee
            # output
            output_.append(h_attn)
            out_attn.append(attn)
            # pointer
            if self.pointer_net:
                if self.attn_decoder:
                    pt_input = torch.cat((input_[k], hidden_[0], c_encoder, c_decoder), 1)
                else:
                    pt_input = torch.cat((input_[k], hidden_[0], c_encoder), 1)
                p_gen[:, k] = F.sigmoid(self.pt_out(pt_input))
                    
        len_seq = input_.size(0)
        batch_size, hidden_size = output_[0].size()
        output_ = torch.cat(output_, 0).view(
            len_seq, batch_size, hidden_size)
        out_attn = torch.cat(out_attn, 0).view(
            len_seq, attn.size(0), attn.size(1))
        
        if self.batch_first:
            output_ = output_.transpose(0,1)
   
        return output_, hidden_, h_attn, out_attn, past_attn, p_gen, past_dehy, loss_cv
'''
GRU decoder
'''
class GRUDecoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        attn_method,
        coverage,
        batch_first,
        pointer_net,
        attn_decoder
    ):
        super(GRUDecoder, self).__init__()
        # parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = num_layers
        self.batch_first = batch_first
        self.attn_method = attn_method.lower()
        self.coverage = coverage
        self.pointer_net = pointer_net
        self.attn_decoder = attn_decoder
        
        self.gru_ = torch.nn.GRUCell(
            self.input_size+self.hidden_size, 
            self.hidden_size).cuda()
        self.encoder_attn_layer = AttentionEncoder(
            hidden_size=self.hidden_size,
            attn_method=self.attn_method,
            coverage=self.coverage).cuda()
        # intra-decoder
        if self.attn_decoder:
            self.decoder_attn_layer = AttentionDecoder(
                hidden_size=self.hidden_size,
                attn_method=self.attn_method).cuda()
            self.attn_out = torch.nn.Linear(
                self.hidden_size*3,
                self.hidden_size,
                bias=True).cuda()
        else:
            self.attn_out = torch.nn.Linear(
                self.hidden_size*2,
                self.hidden_size,
                bias=True).cuda()
        # pointer generator network
        if self.pointer_net:
            if self.attn_decoder:   
                self.pt_out = torch.nn.Linear(
                    self.input_size+self.hidden_size*3, 1).cuda()
            else:
                self.pt_out = torch.nn.Linear(
                    self.input_size+self.hidden_size*2, 1).cuda()
            
    def forward(
        self, idx, input_, hidden_, h_attn, 
        encoder_hy, past_attn, p_gen, past_dehy):
            
        if self.batch_first:
            input_ = input_.transpose(0,1)
        batch_size = input_.size(1)
        
        output_ = []
        out_attn = []
        
        loss_cv = Variable(torch.zeros(1)).cuda()
        batch_size = input_.size(1)
        for k in range(input_.size(0)):
            x_input = torch.cat((input_[k], h_attn), 1)
            hidden_ = self.gru_(x_input, hidden_)
            # attention encoder
            c_encoder, attn, attn_ee = self.encoder_attn_layer(
                hidden_, encoder_hy, past_attn)
            # attention decoder
            if self.attn_decoder:
                if k + idx == 0:
                    c_decoder = Variable(torch.zeros(
                        batch_size, self.hidden_size)).cuda()
                else:
                    c_decoder, attn_de = self.decoder_attn_layer(
                        hidden_, past_dehy)
                past_dehy = past_dehy.transpose(0, 1) # seqL*batch*hidden
                de_idx = past_dehy.size(0)
                if k + idx == 0:
                    past_dehy = hidden_.unsqueeze(0) # seqL*batch*hidden
                    past_dehy = past_dehy.transpose(0, 1) # batch*seqL*hidden
                else:
                    past_dehy = past_dehy.contiguous().view(-1, self.hidden_size) # seqL*batch**hidden
                    past_dehy = torch.cat((past_dehy, hidden_), 0) # (seqL+1)*batch**hidden
                    past_dehy = past_dehy.view(de_idx+1, batch_size, self.hidden_size) # (seqL+1)*batch*hidden
                    past_dehy = past_dehy.transpose(0, 1) # batch*(seqL+1)*hidden
                h_attn = self.attn_out(torch.cat((c_encoder, c_decoder, hidden_), 1))
            else:
                h_attn = self.attn_out(torch.cat((c_encoder, hidden_), 1))
            # coverage
            if self.coverage == 'asee_train':
                lscv = torch.cat((past_attn.unsqueeze(2), attn.unsqueeze(2)), 2)
                lscv = lscv.min(dim=2)[0]
                try:
                    loss_cv = loss_cv + torch.mean(lscv)
                except:
                    loss_cv = torch.mean(lscv)
            if self.coverage[:4] == 'asee':
                past_attn = past_attn + attn
            if self.coverage == 'temporal':
                if k + idx == 0:
                    past_attn = past_attn*0.0
                past_attn = past_attn + attn_ee
            # output
            output_.append(h_attn)
            out_attn.append(attn)
            if self.pointer_net:
                if self.attn_decoder:
                    pt_input = torch.cat((input_[k], hidden_, c_encoder, c_decoder), 1)
                else:
                    pt_input = torch.cat((input_[k], hidden_, c_encoder), 1)
                p_gen[:, k] = F.sigmoid(self.pt_out(pt_input))
            
        len_seq = input_.size(0)
        batch_size, hidden_size = output_[0].size()
        output_ = torch.cat(output_, 0).view(
            len_seq, batch_size, hidden_size)
        out_attn = torch.cat(out_attn, 0).view(
            len_seq, attn.size(0), attn.size(1))
        
        if self.batch_first:
            output_ = output_.transpose(0,1)

        return output_, hidden_, h_attn, out_attn, past_attn, p_gen, past_dehy, loss_cv
'''
sequence to sequence model
''' 
class Seq2Seq(torch.nn.Module):
    
    def __init__(
        self,
        src_emb_dim=128,
        trg_emb_dim=128,
        src_hidden_dim=256,
        trg_hidden_dim=256,
        src_vocab_size=999,
        trg_vocab_size=999,
        src_nlayer=2,
        trg_nlayer=1,
        batch_first=True,
        src_bidirect=True,
        dropout=0.0,
        attn_method='vanilla',
        coverage='vanilla',
        network_='lstm',
        pointer_net=False,
        shared_emb=True,
        attn_decoder=False,
        share_emb_weight=False
    ):
        super(Seq2Seq, self).__init__()
        # parameters
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_nlayer = src_nlayer
        self.trg_nlayer = trg_nlayer
        self.batch_first = batch_first
        self.src_bidirect = src_bidirect
        self.dropout = dropout
        self.attn_method = attn_method.lower()
        self.coverage = coverage.lower()
        self.network_ = network_.lower()
        self.pointer_net = pointer_net
        self.shared_emb = shared_emb
        self.attn_decoder = attn_decoder
        self.share_emb_weight = share_emb_weight
        # bidirection encoder
        self.src_num_directions = 1
        if self.src_bidirect:
            self.src_hidden_dim = src_hidden_dim // 2
            self.src_num_directions = 2
        # source embedding and target embedding
        if self.shared_emb:
            self.embedding = torch.nn.Embedding(
                self.trg_vocab_size,
                self.src_emb_dim).cuda()
            torch.nn.init.uniform(self.embedding.weight, -1.0, 1.0)
        else:
            self.src_embedding = torch.nn.Embedding(
                self.src_vocab_size,
                self.src_emb_dim).cuda()
            torch.nn.init.uniform(self.src_embedding.weight, -1.0, 1.0)
            self.trg_embedding = torch.nn.Embedding(
                self.trg_vocab_size,
                self.trg_emb_dim).cuda()
            torch.nn.init.uniform(self.trg_embedding.weight, -1.0, 1.0)
        # network structure
        if self.network_ == 'lstm':
            # encoder
            self.encoder = torch.nn.LSTM(
                input_size=self.src_emb_dim,
                hidden_size=self.src_hidden_dim,
                num_layers=self.src_nlayer,
                batch_first=self.batch_first,
                dropout=self.dropout,
                bidirectional=self.src_bidirect).cuda()
            # decoder
            self.decoder = LSTMDecoder(
                input_size=self.trg_emb_dim,
                hidden_size=self.trg_hidden_dim,
                num_layers=self.trg_nlayer,
                attn_method=self.attn_method,
                coverage=self.coverage,
                batch_first=self.batch_first,
                pointer_net=self.pointer_net,
                attn_decoder=self.attn_decoder
            ).cuda()
        elif self.network_ == 'gru':
            # encoder
            self.encoder = torch.nn.GRU(
                input_size=self.src_emb_dim,
                hidden_size=self.src_hidden_dim,
                num_layers=self.src_nlayer,
                batch_first=self.batch_first,
                dropout=self.dropout,
                bidirectional=self.src_bidirect).cuda()
            # decoder
            self.decoder = GRUDecoder(
                input_size=self.trg_emb_dim,
                hidden_size=self.trg_hidden_dim,
                num_layers=self.trg_nlayer,
                attn_method=self.attn_method,
                coverage=self.coverage,
                batch_first=self.batch_first,
                pointer_net=self.pointer_net,
                attn_decoder=self.attn_decoder
            ).cuda()
        # encoder to decoder
        self.encoder2decoder = torch.nn.Linear(
            self.src_hidden_dim*self.src_num_directions,
            self.trg_hidden_dim).cuda()
        # decoder to vocab
        if self.share_emb_weight:
            self.decoder2proj = torch.nn.Linear(
                self.trg_hidden_dim,
                self.src_emb_dim,
                bias=False).cuda()
            self.proj2vocab = torch.nn.Linear(
                self.src_emb_dim,
                self.trg_vocab_size,
                bias=True).cuda()
            self.proj2vocab.weight.data = self.embedding.weight.data
        else:
            self.decoder2vocab = torch.nn.Linear(
                self.trg_hidden_dim,
                self.trg_vocab_size,
                bias=True).cuda()
        
    def forward(self, input_src, input_trg):
        # parameters
        src_seq_len = input_src.size(1)
        trg_seq_len = input_trg.size(1)
        # embedding
        if self.shared_emb:
            src_emb = self.embedding(input_src)
            trg_emb = self.embedding(input_trg)
        else:
            src_emb = self.src_embedding(input_src)
            trg_emb = self.trg_embedding(input_trg)

        batch_size = input_src.size(1)
        if self.batch_first:
            batch_size = input_src.size(0)
        # Variables
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers*self.src_num_directions,
            batch_size, self.src_hidden_dim)).cuda()
        if self.coverage == 'temporal':
            past_attn = Variable(torch.ones(
                batch_size, src_seq_len)).cuda()
        else:
            past_attn = Variable(torch.zeros(
                batch_size, src_seq_len)).cuda()
        h_attn = Variable(torch.zeros(
            batch_size, self.trg_hidden_dim)).cuda()
        p_gen = Variable(torch.zeros(
            batch_size, trg_seq_len)).cuda()
        past_dehy = Variable(torch.zeros(1, 1)).cuda()
        # network
        if self.network_ == 'lstm':
            c0_encoder = Variable(torch.zeros(
                self.encoder.num_layers*self.src_num_directions,
                batch_size, self.src_hidden_dim)).cuda()
            # encoder
            encoder_hy, (src_h_t, src_c_t) = self.encoder(
                src_emb, (h0_encoder, c0_encoder))

            if self.src_bidirect:
                h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
                c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
            else:
                h_t = src_h_t[-1]
                c_t = src_c_t[-1]
                        
            decoder_h0 = self.encoder2decoder(h_t)
            decoder_h0 = F.tanh(decoder_h0)
            decoder_c0 = c_t
            # decoder
            trg_h, (_, _), _, attn_, _, p_gen, _, loss_cv = self.decoder(
                0, trg_emb,
                (decoder_h0, decoder_c0),
                h_attn, encoder_hy,
                past_attn, p_gen, past_dehy)
        elif self.network_ == 'gru':
            # encoder
            encoder_hy, src_h_t = self.encoder(
                src_emb, h0_encoder)

            if self.src_bidirect:
                h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            else:
                h_t = src_h_t[-1]
                        
            decoder_h0 = self.encoder2decoder(h_t)
            decoder_h0 = F.tanh(decoder_h0)
            # decoder
            trg_h, _, _, attn_, _, p_gen, _, loss_cv = self.decoder(
                0, trg_emb,
                decoder_h0, h_attn,
                encoder_hy, past_attn, p_gen, past_dehy)
        # prepare output
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0)*trg_h.size(1), trg_h.size(2))
        # consume a lot of memory.
        if self.share_emb_weight:
            decoder_proj = self.decoder2proj(trg_h_reshape)
            decoder_output = self.proj2vocab(decoder_proj)
        else:
            decoder_output = self.decoder2vocab(trg_h_reshape)
        decoder_output = decoder_output.view(
            trg_h.size(0), trg_h.size(1), decoder_output.size(1))

        return decoder_output, attn_, p_gen, loss_cv
    
    def forward_encoder(self, input_src):
        # parameters
        src_seq_len = input_src.size(1)
        # embedding
        if self.shared_emb:
            src_emb = self.embedding(input_src)
        else:
            src_emb = self.src_embedding(input_src)
            
        batch_size = input_src.size(1)
        if self.batch_first:
            batch_size = input_src.size(0)
        # Variables
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers*self.src_num_directions,
            batch_size, self.src_hidden_dim)).cuda()
        if self.coverage == 'temporal':
            past_attn = Variable(torch.ones(
                batch_size, src_seq_len)).cuda()
        else:
            past_attn = Variable(torch.zeros(
                batch_size, src_seq_len)).cuda()
        h_attn = Variable(torch.zeros(
            batch_size, self.trg_hidden_dim)).cuda()
        past_dehy = Variable(torch.zeros(1, 1)).cuda()
        # network
        if self.network_ == 'lstm':
            c0_encoder = Variable(torch.zeros(
                self.encoder.num_layers*self.src_num_directions,
                batch_size, self.src_hidden_dim)).cuda()
            # encoder
            encoder_hy, (src_h_t, src_c_t) = self.encoder(
                src_emb, 
                (h0_encoder, c0_encoder))

            if self.src_bidirect:
                h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
                c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
            else:
                h_t = src_h_t[-1]
                c_t = src_c_t[-1]
                        
            decoder_h0 = self.encoder2decoder(h_t)
            decoder_h0 = F.tanh(decoder_h0)
            decoder_c0 = c_t
                    
            return encoder_hy, (decoder_h0, decoder_c0), h_attn, past_attn, past_dehy
        
        elif self.network_ == 'gru':
            # encoder
            encoder_hy, src_h_t = self.encoder(
                src_emb, h0_encoder)

            if self.src_bidirect:
                h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            else:
                h_t = src_h_t[-1]
                        
            decoder_h0 = self.encoder2decoder(h_t)
            decoder_h0 = F.tanh(decoder_h0)
                
            return encoder_hy, decoder_h0, h_attn, past_attn, past_dehy
    
    def forward_onestep_decoder(
        self,
        idx,
        input_trg,
        hidden_decoder,
        h_attn,
        encoder_hy,
        past_attn,
        past_dehy
    ):
        if self.shared_emb:
            trg_emb = self.embedding(input_trg)
        else:
            trg_emb = self.trg_embedding(input_trg)

        batch_size = input_trg.size(1)
        if self.batch_first:
            batch_size = input_trg.size(0)
        # pointer weight
        p_gen = Variable(torch.zeros(batch_size, 1)).cuda()
        # decoder
        if self.network_ == 'lstm':
            trg_h, hidden_decoder, h_attn, attn_, past_attn, p_gen, past_dehy, loss_cv = self.decoder(
                idx, trg_emb, hidden_decoder, h_attn,
                encoder_hy, past_attn, p_gen, past_dehy)
        if self.network_ == 'gru':
            trg_h, hidden_decoder, h_attn, attn_, past_attn, p_gen, past_dehy, loss_cv = self.decoder(
                idx, trg_emb, hidden_decoder, h_attn,
                encoder_hy, past_attn, p_gen, past_dehy)
        # prepare output
        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size(0) * trg_h.size(1), trg_h.size(2))
        if self.share_emb_weight:
            decoder_proj = self.decoder2proj(trg_h_reshape)
            decoder_output = self.proj2vocab(decoder_proj)
        else:
            decoder_output = self.decoder2vocab(trg_h_reshape)
        decoder_output = decoder_output.view(
            trg_h.size(0), trg_h.size(1), decoder_output.size(1))

        return decoder_output, hidden_decoder, h_attn, past_attn, p_gen, attn_, past_dehy

    def cal_dist(self, input_src, logits_, attn_, p_gen, src_vocab2id):
        # parameters
        src_seq_len = input_src.size(1)
        trg_seq_len = logits_.size(1)
        batch_size = input_src.size(0)
        vocab_size = len(src_vocab2id)
                
        attn_ = attn_.transpose(0, 1)
        # calculate index matrix
        pt_idx = Variable(torch.FloatTensor(torch.zeros(1, 1, 1))).cuda()
        pt_idx = pt_idx.repeat(batch_size, src_seq_len, vocab_size)
        pt_idx.scatter_(2, input_src.unsqueeze(2), 1.0)
        
        return p_gen.unsqueeze(2)*logits_ + (1.0-p_gen.unsqueeze(2))*torch.bmm(attn_, pt_idx)
    
    def cal_dist_explicit(self, input_src, logits_, attn_, p_gen, vocab2id, ext_id2oov):
        # parameters
        src_seq_len = input_src.size(1)
        trg_seq_len = logits_.size(1)
        batch_size = input_src.size(0)
        vocab_size = len(vocab2id) + len(ext_id2oov)
        
        # extend current structure
        logits_ex = Variable(torch.zeros(1, 1, 1)).cuda()
        logits_ex = logits_ex.repeat(batch_size, trg_seq_len, len(ext_id2oov))
        if len(ext_id2oov) > 0:
            logits_ = torch.cat((logits_, logits_ex), -1)
        # pointer
        attn_ = attn_.transpose(0, 1)
        # calculate index matrix
        pt_idx = Variable(torch.FloatTensor(torch.zeros(1, 1, 1))).cuda()
        pt_idx = pt_idx.repeat(batch_size, src_seq_len, vocab_size)
        pt_idx.scatter_(2, input_src.unsqueeze(2), 1.0)
        
        return p_gen.unsqueeze(2)*logits_ + (1.0-p_gen.unsqueeze(2))*torch.bmm(attn_, pt_idx)
