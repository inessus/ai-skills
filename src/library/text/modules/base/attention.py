""" attention functions """
from torch.nn import functional as F


def dot_attention_score(key, query):
    """
        注意力第一次点乘操作
        [B, T, N], [B, 1, N] -> [B, 1, Tk]
    :param key:
    :param query:
    :return:
    """
    return query.matmul(key.transpose(1, 2))


def prob_normalize(score, mask):
    """ [B, 1, Tk]
    user should handle mask shape"""
    score = score.masked_fill(mask == 0, -1e18)
    norm_score = F.softmax(score, dim=-1)
    return norm_score


def attention_aggregate(value, score):
    """
        注意力的第二次点乘
    :param value:
    :param score:
    :return:
    """
    """[B, T, N], [B, 1, T] -> [B,1,N]"""
    output = score.matmul(value)
    return output


def step_attention(query, key, value, mem_mask=None):
    """
        点乘注意力计算　Attention(Q,K,V)=softmax(QK')V
    :param query: [B,N] (32, 256) 解码器输出的矩阵点乘后的query值
    :param key:   [B,T,N] attention
    :param value: [B,T,N]
    :param mem_mask:
    :return:
    """
    """ query[(Bs), B, D], key[B, T, D], value[B, T, D]"""
    score = dot_attention_score(key, query.unsqueeze(-2))
    if mem_mask is None:
        norm_score = F.softmax(score, dim=-1)
    else:
        norm_score = prob_normalize(score, mem_mask)
    output = attention_aggregate(value, norm_score) # [B,1,N]
    return output.squeeze(-2), norm_score.squeeze(-2) # [B,N], [B,T]
