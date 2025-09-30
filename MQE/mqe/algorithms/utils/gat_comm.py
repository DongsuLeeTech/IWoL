import torch
import torch.nn as nn
import torch.nn.functional as F
from mqe.algorithms.utils.util import init, check
from mqe.algorithms.utils.gat import GraphAttention

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class GAT_Comm(nn.Module):
    def __init__(self, input_size, output_size, gat_hidden_size, gat_num_heads):
        super(GAT_Comm, self).__init__()
        dropout = 0
        negative_slope = 0.2
        self.comm_0 = GraphAttention(input_size, gat_hidden_size, dropout, negative_slope, gat_num_heads, average=False, normalize=True)
        self.comm_1 = GraphAttention(gat_hidden_size * gat_num_heads, output_size, dropout, negative_slope, 1, average=True, normalize=True)
        self.ln0 = nn.LayerNorm(output_size)
        self.ln1 = nn.LayerNorm(input_size + output_size)
        self.ln2 = nn.LayerNorm(output_size)
        self.linear = init_(nn.Linear(input_size + output_size, output_size), activate=False)
        self.message_encoder = nn.Sequential(
            init_(nn.Linear(output_size, output_size), activate=True),
            nn.GELU(),
            init_(nn.Linear(output_size, output_size), activate=False)
        )
        self.output_head = nn.Sequential(
            init_(nn.Linear(output_size, output_size), activate=True),
            nn.GELU(),
            nn.LayerNorm(output_size)
        )

    def forward(self, input, graph):
        """
        Compute the message for each agent after the communication.
        :param input: (np.ndarray / torch.Tensor) input feature into network.
        :param graph: the graph for communication
        """
        message = self.comm_0(input, graph)
        message = F.elu(message)
        message = self.comm_1(message, graph)
        # message = self.ln0(F.elu(message))
        message = self.ln1(torch.cat([input, message], dim=-1))
        message = self.linear(message)
        output = self.ln2(message + self.message_encoder(message))
        output = self.output_head(output)

        return output