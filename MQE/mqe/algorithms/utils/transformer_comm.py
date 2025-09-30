import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical, Normal
from mqe.algorithms.utils.util import check, init

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class Transformer_Comm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_comm_hops, num_heads, num_agents,
                 causal_masked=False, fixed_masked=True, mask_threshold=0.5, device=torch.device("cpu")):
        super(Transformer_Comm, self).__init__()
        
        self.state_encoder = nn.Sequential(nn.LayerNorm(input_size),
                                           init_(nn.Linear(input_size, hidden_size), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList(
            [EncodeBlock(hidden_size, num_heads, num_agents, causal_masked, fixed_masked, mask_threshold).to(device)
             for _ in range(num_comm_hops)]
        )
        self.head = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size), activate=True), nn.GELU(), nn.LayerNorm(hidden_size),
                                  init_(nn.Linear(hidden_size, output_size)))
        self.diagonal = torch.eye(num_agents, num_agents, device=device)
        self.no_diagonal = torch.ones(num_agents, num_agents, device=device) - self.diagonal
        self.to(device)
    
    def forward(self, input, graph, pos_embed=None):
        # force self-communicate
        graph = graph * self.no_diagonal + self.diagonal
        hidden_state = self.state_encoder(input)
        if pos_embed is not None:
            hidden_state = hidden_state + pos_embed
        x = self.ln(hidden_state)
        for encode_block in self.blocks:
            x, att, mask = encode_block(x, graph)
        output = self.head(x)
        
        return output, att, mask

    def message_encoding(self, input):
        hidden_state = self.state_encoder(input)
        return hidden_state

    def comm_(self, hidden_state, graph, pos_embed=None):
        # force self-communicate
        graph = graph * self.no_diagonal + self.diagonal
        if pos_embed is not None:
            hidden_state = hidden_state + pos_embed
        x = self.ln(hidden_state)
        for encode_block in self.blocks:
            x = encode_block(x, graph)
        output = self.head(x)

        return output

class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, causal_masked=False, fixed_masked=True, mask_threshold=0.5):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.causal_masked = causal_masked
        self.n_head = n_head
        self.fixed_masked = fixed_masked
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        self.mask_threshold = mask_threshold
        if self.fixed_masked:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("causal_mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                                 .view(1, 1, n_agent + 1, n_agent + 1))
        else:
            # Add this line in the __init__ method
            # self.mask_logits = nn.Parameter(torch.tril(torch.ones(1, n_head, n_agent + 1, n_agent + 1)))
            self.mask_logits = nn.Parameter(torch.zeros(1, n_head, n_agent, n_agent))

        self.att_bp = None

    def forward(self, hidden_state, graph=None):
        # key, value, query: [batch_size, num_agents, embedding_size]
        B, L, D = hidden_state.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(hidden_state).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(hidden_state).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(hidden_state).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.causal_masked:
            if not self.fixed_masked:
                mask = torch.sigmoid(self.mask_logits)  # Ensure the dimensions match
                att = att * mask
                att = att + torch.log(mask + 1e-8)
                # att = att.masked_fill(binary_mask, float('-inf'))  # Element-wise multiplication
            else:
                att = att * self.causal_mask  # .masked_fill(self.causal_mask[:, :, :L, :L] == 0, -1e9)

        if graph is not None:
            # graph: [batch_size, num_agents, num_agents]
            graph = graph.unsqueeze(1).expand(B, self.n_head, L, L)
            # soft_mask = torch.sigmoid(graph * 10)
            # hard_mask = (graph == 1).float()
            # soft_mask = torch.sigmoid(graph)
            # att = att * hard_mask + att.detach() * (soft_mask - hard_mask)
            att = att * graph # .masked_fill(graph[:, :, :, :] == 0, -1e9)
        att = F.softmax(att, dim=-1)  # attention weight
        # att = F.layer_norm(att, att.shape[-1:])
        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y, att, self.mask_logits


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent, causal_masked=False, fixed_masked=True, mask_threshold=0.5):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        if causal_masked:
            self.attn = SelfAttention(n_embd, n_head, n_agent, causal_masked=True, fixed_masked=fixed_masked, mask_threshold=mask_threshold)
        else:
            self.attn = SelfAttention(n_embd, n_head, n_agent, causal_masked=False, fixed_masked=fixed_masked, mask_threshold=mask_threshold)

        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, graph=None):
        if graph is not None:
            y, att, mask = self.attn(x, graph)
            x = self.ln1(x + y)
        else:
            x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.mlp(x))
        return x, att, mask
