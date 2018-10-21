import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, Subset, random_split


class MyDataset(Dataset):
    def __init__(self, t=0, name="myDataset"):
        super(MyDataset, self).__init__()
        self.nums = []
        if t == 0:
            self.nums = [torch.randn(1).item() for _ in range(100)]
        elif t == 1:
            self.nums = list(range(230))
        elif t == 2:
            self.nums = torch.linspace(-1, 1, 250).data.numpy()
        self.name = name
        self.t = t

    def __getitem__(self, i):
        return self.nums[i]

    def __len__(self):
        return len(self.nums)


if __name__ == "__main__":
    loss = nn.L1Loss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(x, target)
    output.backward()


    torch.Tensor(1)
    ds0 = MyDataset(0, "type_0")
    ds1 = MyDataset(1, "type_1")
    ds2 = MyDataset(2, "type_2")
    ds = ds0 + ds1
    ds = ds + ds2
    print(ds.datasets[0].datasets[0].name,ds.datasets[0].datasets[1].name,ds.datasets[1].name)
    print(len(ds))
    dss = random_split(ds, [310, 270]) # 第二个参数是长度，累积和是数据集长度
