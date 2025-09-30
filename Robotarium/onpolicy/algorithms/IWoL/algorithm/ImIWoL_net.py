import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.r_mappo.algorithm.transformer_comm import Transformer_Comm
from onpolicy.algorithms.r_mappo.algorithm.scheduler import Scheduler
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space


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

class Encoder(nn.Module):
    """Encoder module for compressing features into latent space."""
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

class ImIWoL_Actor(nn.Module):
    """Actor network for VoCAL architecture with communication capabilities."""

    def __init__(self, args, obs_space, cent_obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = args.num_agents

        # Communication and scheduling parameters
        self.use_scheduler = args.use_scheduler
        self.sch_slope = args.negative_slope
        self.sch_head = args.scheduler_head
        self.comm_type = args.comm_type
        self.use_pos_embed = args.pos_embed
        self.obs_pos_embed_end = args.obs_pos_embed_end
        self.obs_pos_embed_start = args.obs_pos_embed_start
        self.obs_info_scheduler = args.obs_info_scheduler
        self.skip_connect_final = args.skip_connect_final
        self.mask_threshold = args.mask_threshold
        self.obs_enc_type = args.obs_enc_type
        self.latent_size = args.latent_size

        obs_shape = get_shape_from_obs_space(obs_space)
        cent_obs_space = get_shape_from_obs_space(cent_obs_space)

        # Initialize network components
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)
        if self.obs_info_scheduler == 'obs_enc':
            enc_sample = self.base(torch.zeros(obs_shape)).squeeze(0)
            sch_shape = enc_sample.shape
        elif self.obs_info_scheduler == 'rnn_enc':
            sch_shape = self.hidden_size
        elif self.obs_info_scheduler == 'obs':
            sch_shape = obs_shape

        self.encoder = Encoder(sch_shape, self.hidden_size, self.latent_size)

        if type(sch_shape) == int:
            self.act = ACTLayer(action_space, self.latent_size + sch_shape, self._use_orthogonal, self._gain)
        else:
            self.act = ACTLayer(action_space, self.latent_size + sch_shape[0], self._use_orthogonal, self._gain)

        # Initialize decoder for world dynamics
        if isinstance(cent_obs_space, tuple):
            output_dim = cent_obs_space[0] if len(cent_obs_space) > 0 else 1
        else:
            output_dim = cent_obs_space

        self.world_decoder = Decoder(output_dim, self.hidden_size, self.latent_size)
        self.interactive_decoder = Decoder(self.hidden_size, self.hidden_size, self.latent_size)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """Forward pass of the actor network."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

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
        actor_features = self.encoder(sch_input)

        # Action generation
        if self.skip_connect_final:
            actor_features = torch.cat((actor_features, sch_input), 1)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None,
                         active_masks=None):
        """Evaluate actions for training."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

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
        actor_features = self.encoder(sch_input)

        if self.skip_connect_final:
            actor_features = torch.cat((actor_features, sch_input), 1)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions,
                                                                   active_masks)

        return action_log_probs, dist_entropy

    def latent_world(self, state, obs, rnn_states, masks):
        """
        Predict the world state from the communication latent vectors.
        
        Args:
            state: The actual world state (shared observation)
            obs: Local observations
            rnn_states: RNN states
            masks: Masks for RNN
            comm_graphs: Communication graphs
            available_actions: Available actions
            active_masks: Active masks
            
        Returns:
            pred_state: Predicted state from the decoder
            state: The actual state
        """
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

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
        latent = self.encoder(sch_input)
        # Decode the communication latent vector to predict the world state
        pred_state = self.world_decoder(latent)
        
        return pred_state, state

    def latent_interactive(self, state, obs, rnn_states, masks):
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

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
        latent = self.encoder(sch_input)

        # Decode the communication latent vector to predict the world state
        pred_message = self.interactive_decoder(latent)

        return pred_message

class ImIWoL_Critic(nn.Module):
    """Critic network for VoCAL architecture."""

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.use_scheduler = args.use_scheduler
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = args.num_agents

        # Communication and scheduling parameters
        self.comm_type = args.comm_type
        self.sch_slope = args.negative_slope
        self.sch_head = args.scheduler_head
        self.use_pos_embed = args.pos_embed
        self.obs_pos_embed_end = args.obs_pos_embed_end
        self.obs_pos_embed_start = args.obs_pos_embed_start
        self.obs_info_scheduler = args.obs_info_scheduler
        self.skip_connect_final = args.skip_connect_final
        self.mask_threshold = args.mask_threshold
        self.obs_enc_type = args.obs_enc_type

        # Initialize network components
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        if self.obs_info_scheduler == 'obs_enc':
            enc_sample = self.base(torch.zeros(cent_obs_shape))
            cent_obs_shape = enc_sample.shape
        elif self.obs_info_scheduler == 'rnn_enc':
            cent_obs_shape = self.hidden_size

        if self.use_scheduler:
            self.scheduler = Scheduler(args, cent_obs_shape, self.hidden_size, self.sch_head, self.sch_slope)

        self.comm = Transformer_Comm(
            self.hidden_size, self.hidden_size, args.comm_hidden_size,
            args.num_comm_hops, args.comm_num_heads, self.num_agents,
            args.causal_masked, args.fixed_masked, self.mask_threshold, device
        )

        if self.use_pos_embed:
            self.pos_encoder = nn.Linear(args.obs_pos_embed_end - args.obs_pos_embed_start, self.hidden_size)

        # Value head
        init_method = nn.init.orthogonal_ if self._use_orthogonal else nn.init.xavier_uniform_

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        v_input = self.hidden_size
        if self.skip_connect_final:
            if isinstance(cent_obs_shape, int):
                v_input += cent_obs_shape
            else:
                v_input += cent_obs_shape[0]

        self.v_out = init_(PopArt(v_input, 1, device=device) if self._use_popart else nn.Linear(v_input, 1))
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, comm_graphs):
        """Forward pass of the critic network."""
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if comm_graphs is not None:
            comm_graphs = check(comm_graphs).to(**self.tpdv)

        # Position embedding
        pos_embed = None
        if self.use_pos_embed:
            pos_embed = self.pos_encoder(cent_obs[:, self.obs_pos_embed_start - 1:self.obs_pos_embed_end - 1])

        # Feature extraction
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

        # Communication scheduling
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

        return values, rnn_states, att, att_mask, graphs

    def message_forward(self, cent_obs, rnn_states, masks, comm_graphs):
        """Forward pass of the critic network."""
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if comm_graphs is not None:
            comm_graphs = check(comm_graphs).to(**self.tpdv)

        # Position embedding
        pos_embed = None
        if self.use_pos_embed:
            pos_embed = self.pos_encoder(cent_obs[:, self.obs_pos_embed_start - 1:self.obs_pos_embed_end - 1])

        # Feature extraction
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

        # Communication scheduling
        graphs = self.scheduler(sch_input)

        # Communication
        critic_features, att, att_mask = self.comm(
            critic_features.view(-1, self.num_agents, self.hidden_size),
            graphs,
            pos_embed.view(-1, self.num_agents, self.hidden_size) if pos_embed is not None else None
        )
        critic_features = critic_features.view(-1, self.hidden_size)

        return critic_features
