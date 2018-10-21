import torch
import torch.nn as nn

input_size = 11
hidden_size = 120
batch = 28
layers = 2
tunnel = 4

input = torch.randn(batch, tunnel, input_size)

rnn = nn.GRU(input_size, hidden_size, layers)
h0 = torch.randn(layers, tunnel, _size)
output, hn = rnn(input, h0)
output.size(), hn.size()
# output, [batch, tunnel, hidden_size], [layers, tunnel, hidden_size]
torch.Tensor(23)