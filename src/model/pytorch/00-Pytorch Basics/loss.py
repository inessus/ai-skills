import torch
import torch.nn as nn


def L1Loss():
    loss = nn.L1Loss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(x, target)
    output.backward()


def PoissonNLLLoss():
    loss = nn.PoissonNLLLoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(x, target)
    output.backward()
    pass


def MSELoss():
    loss = nn.MSELoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(x, target)
    output.backward()
    pass


def BCEWithLogitsLoss():
    loss = nn.BCEWithLogitsLoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(x, target)
    output.backward()
    pass


def HingeEmbeddingLoss():
    loss = nn.HingeEmbeddingLoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(x, target)
    output.backward()
    pass


def MultiLabelMarginLoss():
    loss = nn.MultiLabelMarginLoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(x, target)
    output.backward()
    pass


def SmoothL1Loss():
    m = nn.Sigmoid()
    loss = nn.SmoothL1Loss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(m(x), target)
    output.backward()
    pass


def SoftMarginLoss():
    m = nn.Sigmoid()
    loss = nn.SoftMarginLoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(m(x), target)
    output.backward()
    pass


def CosineEmbeddingLoss():
    m = nn.Sigmoid()
    loss = nn.CosineEmbeddingLoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(m(x), target)
    output.backward()
    pass


def MarginRankingLoss():
    m = nn.Sigmoid()
    loss = nn.MarginRankingLoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(m(x), target)
    output.backward()
    pass


def TripletMarginLoss():
    m = nn.Sigmoid()
    loss = nn.TripletMarginLoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(m(x), target)
    output.backward()
    pass


def NLLLoss():
    m = nn.Sigmoid()
    loss = nn.NLLLoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(m(x), target)
    output.backward()
    pass


def BCELoss():
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(m(x), target)
    output.backward()
    pass


def CrossEntropyLoss():
    m = nn.Sigmoid()
    loss = nn.CrossEntropyLoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(m(x), target)
    output.backward()
    pass


def MultiLabelSoftMarginLoss():
    m = nn.Sigmoid()
    loss = nn.MultiLabelSoftMarginLoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(m(x), target)
    output.backward()
    pass


def MultiMarginLoss():
    m = nn.Sigmoid()
    loss = nn.MultiMarginLoss()
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(m(x), target)
    output.backward()
    pass


if __name__ == "__main__":
    # L1Loss()
    # PoissonNLLLoss()
    # MSELoss()
    # BCEWithLogitsLoss()
    # HingeEmbeddingLoss()
    # MultiLabelMarginLoss()
    # SmoothL1Loss()
    # SoftMarginLoss()
    # CosineEmbeddingLoss()
    # MarginRankingLoss()
    # TripletMarginLoss()
    # NLLLoss()
    # BCELoss()
    # CrossEntropyLoss()
    # MultiLabelSoftMarginLoss()
    MultiMarginLoss()