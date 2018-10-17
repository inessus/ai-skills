import torch
import torch.nn.functional as F


# Luong attention layer
class Attn(torch.nn.Module):
    """
        全局注意力
    """

    def __init__(self, method, hidden_size):
        """

        :param method:
        :param hidden_size:
        """
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        """
            输出维度是最大的那个，本质上是一个小向量向大向量的广播
        :param hidden: 1*B*N  解码器循环网络的输出
        :param encoder_output: L*B*N 编码器循环网络的输出
        :return:
        """
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        """

        :param hidden:
        :param encoder_output:
        :return:
        """
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        """

        :param hidden:
        :param encoder_output:
        :return:
        """
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        """

        :param hidden:
        :param encoder_outputs:
        :return:
        """
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot': # (1*B*N) (T*B*N)   (1*64*500)*(10*64*500)
            attn_energies = self.dot_score(hidden, encoder_outputs) # (T*B)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t() # B, T

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)