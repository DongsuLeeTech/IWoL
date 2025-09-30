import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


def discrete_autoregreesive_act(decoder, obs_rep, obs, relation_embed, relations, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False, dec_agent=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log


def discrete_parallel_act(decoder, obs_rep, obs, action, relation_embed, relations, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None, dec_agent=False):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logit = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy


def continuous_autoregreesive_act(decoder, obs_rep, obs, relation_embed, relations, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False, dec_agent=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)[:, i, :]
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log


def continuous_parallel_act(decoder, obs_rep, obs, action, relation_embed, relations, batch_size, n_agent, action_dim, tpdv, dec_agent=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs, relation_embed, attn_mask=relations, dec_agent=dec_agent)
    action_std = torch.sigmoid(decoder.log_std) * 0.5

    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy


import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F
from bidexhands .algorithms.utils.util import init


class TransformerACTLayer(nn.Module):
    """
    Transformer-based action layer that handles both discrete and continuous action spaces.
    """
    def __init__(self, action_space, hidden_size, gain=0.01, use_orthogonal=True):
        super(TransformerACTLayer, self).__init__()
        self.multidiscrete_action = False
        self.continuous_action = False
        self.mixed_action = False
        
        # Initialize weights
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = init_(nn.Linear(hidden_size, action_dim))
        elif action_space.__class__.__name__ == "Box":
            self.continuous_action = True
            action_dim = action_space.shape[0]
            self.action_out = init_(nn.Linear(hidden_size, action_dim * 2))
        else:
            raise NotImplementedError

        self.hidden_size = hidden_size
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))

    def forward(self, hidden_states, available_actions=None, deterministic=False):
        """
        Forward pass of the action layer.
        
        Args:
            hidden_states: Output from previous layer
            available_actions: Mask for available actions
            deterministic: If True, return most likely action
            
        Returns:
            actions: Selected actions
            action_log_probs: Log probabilities of selected actions
        """
        if self.continuous_action:
            action_out = self.action_out(hidden_states)
            mean, std = torch.chunk(action_out, 2, dim=-1)
            std = torch.sigmoid(std) + 1e-5

            if deterministic:
                action = mean
            else:
                normal = Normal(mean, std)
                action = normal.rsample()

            action_log_probs = normal.log_prob(action).sum(-1, keepdim=True)
            
        else:  # discrete action
            logits = self.action_out(hidden_states)
            
            if available_actions is not None:
                logits[available_actions == 0] = -1e10

            dist = Categorical(logits=logits)
            
            if deterministic:
                action = dist.probs.argmax(dim=-1, keepdim=True)
            else:
                action = dist.sample().view(-1, 1)

            action_log_probs = dist.log_prob(action.squeeze(-1)).view(action.size(0), -1)

        return action, action_log_probs

    def evaluate_actions(self, hidden_states, actions, available_actions=None, active_masks=None):
        """
        Evaluate actions for training.
        
        Args:
            hidden_states: Output from previous layer
            actions: Actions to evaluate
            available_actions: Mask for available actions
            active_masks: Mask for active agents
            
        Returns:
            action_log_probs: Log probabilities of actions
            dist_entropy: Action distribution entropy
        """
        if self.continuous_action:
            action_out = self.action_out(hidden_states)
            mean, std = torch.chunk(action_out, 2, dim=-1)
            std = torch.sigmoid(std) + 1e-5
            normal = Normal(mean, std)
            
            action_log_probs = normal.log_prob(actions).sum(-1, keepdim=True)
            if active_masks is not None:
                action_log_probs = action_log_probs * active_masks
                
            dist_entropy = normal.entropy().mean()
            
        else:  # discrete action
            logits = self.action_out(hidden_states)
            
            if available_actions is not None:
                logits[available_actions == 0] = -1e10

            dist = Categorical(logits=logits)
            action_log_probs = dist.log_prob(actions.squeeze(-1)).view(actions.size(0), -1)
            
            if active_masks is not None:
                action_log_probs = action_log_probs * active_masks
                
            dist_entropy = dist.entropy().mean()

        return action_log_probs, dist_entropy
