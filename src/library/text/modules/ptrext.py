import torch.nn as nn

from library.text.modules.base.encoder import ConvSentEncoder, LSTMEncoder


class PtrExtractSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden, lstm_hidden, lstm_layer, bidirectional, n_hop=1, dropout=0.0):
        """

        :param emb_dim:
        :param vocab_size:
        :param conv_hidden:
        :param lstm_hidden:
        :param lstm_layer:
        :param bidirectional:
        :param n_hop:
        :param dropout:
        """
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._extractor = LSTMPointerNet(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop
        )

    def forward(self, article_sents, sent_nums, target):
        """

        :param article_sents:
        :param sent_nums:
        :param target:
        :return:
        """
        enc_out = self._encode(article_sents, sent_nums)
        bs, nt = target.size()
        d = enc_out.size(2)
        ptr_in = torch.gather(
            enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d)
        )
        output = self._extractor(enc_out, sent_nums, ptr_in)
        return output

    def extract(self, article_sents, sent_nums=None, k=4):
        """

        :param article_sents:
        :param sent_nums:
        :param k:
        :return:
        """
        enc_out = self._encode(article_sents, sent_nums)
        output = self._extractor.extract(enc_out, sent_nums, k)
        return output

    def _encode(self, article_sents, sent_nums):
        """

        :param article_sents:
        :param sent_nums:
        :return:
        """
        if sent_nums is None:  # test-time excode only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent)
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)], dim=0)
                   if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        lstm_out = self._art_enc(enc_sent, sent_nums)
        return lstm_out

    def set_embedding(self, embedding):
        """

        :param embedding:
        :return:
        """
        self._sent_enc.set_embedding(embedding)