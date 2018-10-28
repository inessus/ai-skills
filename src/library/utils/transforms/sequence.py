import math

import torch
from torch.nn import functional as F


#################### general sequence helper #########################
def len_mask(lens, device):
    """ users are resposible for shaping
    Return: tensor_type [B, T]
    """
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.ByteTensor(batch_size, max_len).to(device)
    mask.fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l].fill_(1)
    return mask


def sequence_mean(sequence, seq_lens, dim=1):
    """
        输入向量进行了转置 [128, 32, 256] [B,T,N]
        batch_size个向量 每个向量求平均数，组成一个向量
    """
    if seq_lens:
        assert sequence.size(0) == len(seq_lens)  # 通道， 相当于句子长度
        sum_ = torch.sum(sequence, dim=dim, keepdim=False)  # 每行求和，保留batch_size维度 (B,N)
        mean = torch.stack([s / l for s, l in zip(sum_, seq_lens)], dim=0)  # 除以长度 后堆叠成一个向量
    else:
        mean = torch.mean(sequence, dim=dim, keepdim=False)  # 简单直接的求平均函数
    return mean


def sequence_loss(logits, targets, xent_fn=None, pad_idx=0):
    """
        损失函数计算了，默认使用交叉熵计算 logits [B,T',V']
        挑选序列，进行整理，计算商值      targets [B,T']
    """
    assert logits.size()[:-1] == targets.size()

    mask = targets != pad_idx
    target = targets.masked_select(mask)  # 去掉pad的值，其实pad本身就是0
    logit = logits.masked_select(
        mask.unsqueeze(2).expand_as(logits)  # [B,T',V']
    ).contiguous().view(-1, logits.size(-1))
    if xent_fn:
        loss = xent_fn(logit, target)  # [BT',V'] [BT']
    else:
        loss = F.cross_entropy(logit, target)
    assert (not math.isnan(loss.mean().item())
            and not math.isinf(loss.mean().item()))
    return loss


#################### LSTM helper #########################
def reorder_sequence(sequence_emb, order, batch_first=False):
    """
    sequence_emb: [T, B, N] if not batch_first
    order: list of sequence length
    根据索引顺序选择重新排列顺序， batch_first代表梯度优先
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]

    order = torch.LongTensor(order).to(sequence_emb.device)
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)

    return sorted_


def reorder_lstm_states(lstm_states, order):
    """
    lstm_states: (H, C) of tensor ([2L, B, N],[2L, B, N])
    order: list of sequence length
    """
    assert isinstance(lstm_states, tuple)
    assert len(lstm_states) == 2
    assert lstm_states[0].size() == lstm_states[1].size()
    assert len(order) == lstm_states[0].size()[1]

    order = torch.LongTensor(order).to(lstm_states[0].device)
    sorted_states = (lstm_states[0].index_select(index=order, dim=1),
                     lstm_states[1].index_select(index=order, dim=1))

    return sorted_states