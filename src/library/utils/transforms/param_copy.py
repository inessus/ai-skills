import copy


def transfer_weights(model_from, model_to):
    """
        实现了从model_from到model to的相同网络参数的拷贝。
        复制一个目标参数，把没有的都填上，就可以使用了，
    :param model_from:
    :param model_to:
    :return:
    """
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys() :
        if (not k in wf) | (k == 'fc.weight') | (k == 'fc.bias'): # 例如两个都有fc层，那么要求fc层的参数是一样的 就是参数维度不匹配啦。
            wf[k] = wt[k]
    model_to.load_state_dict(wf)

