import torch
import torch.nn as nn
import torch.nn.functional as F

from bidexhands.algorithms.utils.util import init, check
from bidexhands.algorithms.utils.cnn import CNNBase
from bidexhands.algorithms.utils.mlp import MLPBase
from bidexhands.algorithms.utils.rnn import RNNLayer
from bidexhands.algorithms.utils.act import ACTLayer
from bidexhands.algorithms.utils.scheduler import Scheduler
from bidexhands.algorithms.utils.transformer_comm import Transformer_Comm

from bidexhands.utils.util import get_shape_from_obs_space
from bidexhands.algorithms.marl.utils.popart import PopArt


class Decoder(nn.Module):
    """Decoder module for reconstructing state from latent space."""

    def __init__(self, output_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return F.tanh(self.fc3(h2))

class ExIWoL_Actor(nn.Module):
    def __init__(self, config, obs_space, cent_obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.config=config
        self._gain = config["gain"]
        self._use_orthogonal = config["use_orthogonal"]
        self._use_policy_active_masks = config["use_policy_active_masks"]
        self._use_naive_recurrent_policy = config["use_naive_recurrent_policy"]
        self._use_recurrent_policy = config["use_recurrent_policy"]
        self._recurrent_N = config["recurrent_N"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.num_agents = 2
        self.use_scheduler = config['use_scheduler']
        self.sch_slope = config['negative_slope']
        self.sch_head = config['scheduler_head']
        self.use_pos_embed = config['pos_embed']
        self.obs_pos_embed_end = config['obs_pos_embed_end']
        self.obs_pos_embed_start = config['obs_pos_embed_start']
        self.skip_connect_final = config['skip_connect_final']
        self.mask_threshold = config['mask_threshold']
        self.obs_enc_type = config['obs_enc_type']
        self.use_vq_vae = config['use_vq_vae']
        self.code_size = config['code_size']
        self.obs_info_scheduler = config['obs_info_scheduler']

        obs_shape = get_shape_from_obs_space(obs_space)
        cent_obs_space = get_shape_from_obs_space(cent_obs_space)

        # Initialize network components
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(self.config, obs_shape)
        if self.obs_info_scheduler == 'obs_enc':
            enc_sample = self.base(torch.zeros(obs_shape)).squeeze(0)
            sch_shape = enc_sample.shape
        elif self.obs_info_scheduler == 'rnn_enc':
            sch_shape = self.hidden_size
        elif self.obs_info_scheduler == 'obs':
            sch_shape = obs_shape

        self.scheduler = Scheduler(config, self.hidden_size, self.hidden_size, self.sch_head, self.sch_slope)
        self.comm = Transformer_Comm(
            self.hidden_size, self.hidden_size, config['comm_hidden_size'],
            config['num_comm_hops'], config['comm_num_heads'], self.num_agents,
            config['causal_masked'], config['fixed_masked'], self.mask_threshold, device
        )

        if self.use_pos_embed:
            self.pos_encoder = nn.Linear(self.obs_pos_embed_end - self.obs_pos_embed_start, self.hidden_size)

        if type(sch_shape) == int:
            self.act = ACTLayer(action_space, self.code_size + sch_shape, self._use_orthogonal, self._gain, self.config)
        else:
            self.act = ACTLayer(action_space, self.code_size + sch_shape[0], self._use_orthogonal, self._gain, self.config)

        # Initialize decoder for world dynamics
        if isinstance(cent_obs_space, tuple):
            output_dim = cent_obs_space[0] if len(cent_obs_space) > 0 else 1
        else:
            output_dim = cent_obs_space
        self.decoder = Decoder(output_dim, self.hidden_size, self.hidden_size)

        self.to(device)

    def forward(self, obs, rnn_states, masks, comm_graphs, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # Position embedding
        pos_embed = None
        if self.use_pos_embed:
            pos_embed = self.pos_encoder(obs[:, self.obs_pos_embed_start - 1:self.obs_pos_embed_end - 1])

        actor_features = self.base(obs)
        if self.obs_info_scheduler == 'obs_enc':
            sch_input = actor_features.clone()
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        elif self.obs_info_scheduler == 'rnn_enc':
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            sch_input = actor_features.clone()
        elif self.obs_info_scheduler == 'obs':
            sch_input = obs.clone()
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        graphs = self.scheduler(sch_input)

        # Communication
        actor_features, att, att_mask = self.comm(
            actor_features.view(-1, self.num_agents, self.hidden_size),
            graphs,
            pos_embed.view(-1, self.num_agents, self.hidden_size) if pos_embed is not None else None
        )
        actor_features = actor_features.view(-1, self.hidden_size)

        # Action generation
        if self.skip_connect_final:
            actor_features = torch.cat((actor_features, sch_input), 1)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, comm_graphs, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # Position embedding
        pos_embed = None
        if self.use_pos_embed:
            pos_embed = self.pos_encoder(obs[:, self.obs_pos_embed_start - 1:self.obs_pos_embed_end - 1])

        actor_features = self.base(obs)
        if self.obs_info_scheduler == 'obs_enc':
            sch_input = actor_features.clone()
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        elif self.obs_info_scheduler == 'rnn_enc':
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            sch_input = actor_features.clone()
        elif self.obs_info_scheduler == 'obs':
            sch_input = obs.clone()
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        graphs = self.scheduler(sch_input)

        # Communication
        actor_features, att, att_mask = self.comm(
            actor_features.view(-1, self.num_agents, self.hidden_size),
            graphs,
            pos_embed.view(-1, self.num_agents, self.hidden_size) if pos_embed is not None else None
        )
        actor_features = actor_features.view(-1, self.hidden_size)

        # Action generation
        if self.skip_connect_final:
            actor_features = torch.cat((actor_features, sch_input), 1)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                action, available_actions,
                                                                active_masks)

        return action_log_probs, dist_entropy

    def latent_world(self, state, obs, rnn_states, masks, comm_graphs, available_actions=None, active_masks=None):
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # Position embedding
        pos_embed = None
        if self.use_pos_embed:
            pos_embed = self.pos_encoder(obs[:, self.obs_pos_embed_start - 1:self.obs_pos_embed_end - 1])

        # Feature extraction
        actor_features = self.base(obs)
        if self.obs_info_scheduler == 'obs_enc':
            sch_input = actor_features.clone()
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        elif self.obs_info_scheduler == 'rnn_enc':
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
            sch_input = actor_features.clone()
        elif self.obs_info_scheduler == 'obs':
            sch_input = obs.clone()
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Communication scheduling
        graphs = self.scheduler(sch_input)

        # Communication
        actor_features, att, att_mask = self.comm(
            actor_features.view(-1, self.num_agents, self.hidden_size),
            graphs,
            pos_embed.view(-1, self.num_agents, self.hidden_size) if pos_embed is not None else None
        )
        actor_features = actor_features.view(-1, self.hidden_size)

        # Decode the communication latent vector to predict the world state
        pred_state = self.decoder(actor_features)

        return pred_state, state


class ExIWoL_Critic(nn.Module):
    def __init__(self, config, cent_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self._use_orthogonal = config["use_orthogonal"]
        self._use_naive_recurrent_policy = config["use_naive_recurrent_policy"]
        self._use_recurrent_policy = config["use_recurrent_policy"]
        self._recurrent_N = config["recurrent_N"]
        self._use_popart = config['use_popart']
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.config = config

        self.num_agents = 2
        self.use_scheduler = config['use_scheduler']
        self.sch_slope = config['negative_slope']
        self.sch_head = config['scheduler_head']
        self.use_pos_embed = config['pos_embed']
        self.obs_pos_embed_end = config['obs_pos_embed_end']
        self.obs_pos_embed_start = config['obs_pos_embed_start']
        self.skip_connect_final = config['skip_connect_final']
        self.mask_threshold = config['mask_threshold']
        self.obs_enc_type = config['obs_enc_type']
        self.use_vq_vae = config['use_vq_vae']
        self.code_size = config['code_size']
        self.obs_info_scheduler = config['obs_info_scheduler']

        # Initialize network components
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(self.config, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        if self.obs_info_scheduler == 'obs_enc':
            enc_sample = self.base(torch.zeros(cent_obs_shape))
            cent_obs_shape = enc_sample.shape
        elif self.obs_info_scheduler == 'rnn_enc':
            cent_obs_shape = self.hidden_size

        if self.use_scheduler:
            self.scheduler = Scheduler(self.config, cent_obs_shape, self.hidden_size, self.sch_head, self.sch_slope)

        self.comm = Transformer_Comm(
            self.hidden_size, self.hidden_size, config['comm_hidden_size'],
            config['num_comm_hops'], config['comm_num_heads'], self.num_agents,
            config['causal_masked'], config['fixed_masked'], self.mask_threshold, device
        )

        if self.use_pos_embed:
            self.pos_encoder = nn.Linear(self.obs_pos_embed_end - self.obs_pos_embed_start, self.hidden_size)

        # Value head
        init_method = nn.init.orthogonal_ if self._use_orthogonal else nn.init.xavier_uniform_
        def init_(m): return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        v_input = self.hidden_size
        if self.skip_connect_final:
            if isinstance(cent_obs_shape, int):
                v_input += cent_obs_shape
            else:
                v_input += cent_obs_shape[0]

        self.v_out = init_(PopArt(v_input, 1, device=device) if self._use_popart else nn.Linear(v_input, 1))
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, comm_graphs):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # Position embedding
        pos_embed = None
        if self.use_pos_embed:
            pos_embed = self.pos_encoder(cent_obs[:, self.obs_pos_embed_start - 1:self.obs_pos_embed_end - 1])

        critic_features = self.base(cent_obs)
        if self.obs_info_scheduler == 'obs_enc':
            sch_input = critic_features.clone()
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        elif self.obs_info_scheduler == 'rnn_enc':
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
            sch_input = critic_features.clone()
        elif self.obs_info_scheduler == 'obs':
            sch_input = cent_obs.clone()
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        graphs = self.scheduler(sch_input)

        # Communication
        critic_features, att, att_mask = self.comm(
            critic_features.view(-1, self.num_agents, self.hidden_size),
            graphs,
            pos_embed.view(-1, self.num_agents, self.hidden_size) if pos_embed is not None else None
        )
        critic_features = critic_features.view(-1, self.hidden_size)

        if self.skip_connect_final:
            critic_features = torch.cat((critic_features, sch_input), 1)

        values = self.v_out(critic_features)

        return values, rnn_states
