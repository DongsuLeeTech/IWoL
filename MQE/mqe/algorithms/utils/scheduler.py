import torch
import torch.nn as nn
import torch.nn.functional as F
from mqe.algorithms.utils.cnn import CNNBase
from mqe.algorithms.utils.mlp import MLPBase


class Scheduler(nn.Module):
    """
    Scheduler for communication
    """
    def __init__(self, config, obs_shape, hidden_size, num_heads, negative_slope):
        super(Scheduler, self).__init__()
        self.num_agents = 2
        self.config = config
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.negative_slope = negative_slope
        obs_shape = (obs_shape,) if isinstance(obs_shape, int) else obs_shape
        base = MLPBase # CNNBase if len(obs_shape) == 3 else MLPBase
        self.obs_encoder = base(self.config, obs_shape)
        self.a_i = nn.Parameter(torch.zeros(size=(num_heads, hidden_size, 1)))
        self.a_j = nn.Parameter(torch.zeros(size=(num_heads, hidden_size, 1)))
        # more steep negative slope means that when you get negative values, no communication is more probable
        self.leakyrelu = nn.LeakyReLU(self.negative_slope)

        initial_temperature = 1.0
        final_temperature = 0.1
        decay_rate = 0.99

        self.temperature = max(final_temperature, initial_temperature * (decay_rate ** config['num_env_steps']))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialization for the parameters of the scheduler
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.a_i, gain=gain)
        nn.init.xavier_normal_(self.a_j, gain=gain)

    def forward(self, obs):
        h = self.obs_encoder(obs).view(-1, self.num_agents, self.hidden_size)
        h_expanded = h.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        coeff_i = torch.matmul(h_expanded, self.a_i)
        coeff_j = torch.matmul(h_expanded, self.a_j)
        coeff = coeff_i + coeff_j.transpose(2, 3)

        e = coeff.mean(dim=1).squeeze(-1)
        e = torch.stack([self.leakyrelu(e), torch.zeros_like(e)], dim=-1)

        diff_graph = F.gumbel_softmax(e, hard=True)[:, :, :, 0].squeeze(-1)
        return diff_graph



# class Scheduler(nn.Module):
#     """
#     Scheduler for communication
#     """
#     def __init__(self, args, obs_shape, hidden_size):
#         super(Scheduler, self).__init__()
#         self.num_agents = args.num_agents
#         self.hidden_size = hidden_size
#         base = CNNBase if len(obs_shape) == 3 else MLPBase
#         self.obs_encoder = base(args, obs_shape)
#         self.scheduler_mlp = nn.Sequential(
#             nn.Linear(hidden_size * 2, hidden_size // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size // 2, hidden_size // 4),
#             nn.ReLU(),
#             nn.Linear(hidden_size // 4, 2))

#     def forward(self, obs):
#         # obs: [batch_size * num_agents, hidden_size]
#         # e: [batch_size, num_agents, hidden_size]
#         e = self.obs_encoder(obs).view(-1, self.num_agents, self.hidden_size)
#         # hard_attn_input: [batch_size, num_agents, num_agents, 2 * hidden_size]
#         hard_attn_input = torch.cat([e.unsqueeze(2).repeat(1, 1, self.num_agents, 1), e.unsqueeze(1).repeat(1, self.num_agents, 1, 1)], dim=-1)
#         hard_attn_output = F.gumbel_softmax(self.scheduler_mlp(hard_attn_input), hard=True)
#         # output: [batch_size, num_agents, num_agents]
#         output = torch.narrow(hard_attn_output, 3, 1, 1).squeeze(-1)
#         return output