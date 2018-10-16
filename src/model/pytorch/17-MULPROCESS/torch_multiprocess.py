import torch
import torch.multiprocessing as mp


def test_cuda_small_tensor(self):
    # Check multiple small tensors which will likely use the same
    # underlying cached allocation
    ctx = mp.get_context('spawn')
    tensors = []
    for i in range(5):
        tensors += [torch.arange(i*5, (i + 1) * 5)]

    inq = ctx.Queue()
    outq = ctx.Queue()
    inq.put(tensors)

    pass