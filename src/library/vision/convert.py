import torch
from torch.autograd import Variable
from scipy.sparse.csr import csr_matrix
from collections import Iterable


class Convert():
    def __init__(self):
        pass

    @staticmethod
    def to(a, cuda=False):
        if isinstance(a, Variable):
            ret = a.double()
        elif not isinstance(a, Iterable):
            ret = Variable(torch.tensor(int(a)))
        elif isinstance(a, csr_matrix):
            ret = Variable(torch.from_numpy(a.toarray()).float())
        else:
            ret = Variable(torch.tensor(a).float())
        if cuda:
            ret = ret.cuda()
        return ret

# X = X.float()
# or cast your complete model to DoubleTensor as
#
# model = model.double()
# You can also set the default type for all tensors using
#
# pytorch.set_default_tensor_type('torch.DoubleTensor')