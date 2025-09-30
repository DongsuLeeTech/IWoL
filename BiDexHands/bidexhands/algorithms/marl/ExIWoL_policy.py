import torch
from bidexhands.algorithms.marl.ExIWoL_net import ExIWoL_Actor, ExIWoL_Critic
from bidexhands.utils.util import update_linear_schedule

class ExIWoL_Policy:
    def __init__(self, config, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = config["lr"]
        self.critic_lr = config["critic_lr"]
        self.opti_eps = config["opti_eps"]
        self.weight_decay = config["weight_decay"]

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = ExIWoL_Actor(config, self.obs_space, self.share_obs_space, self.act_space, self.device)
        self.critic = ExIWoL_Critic(config, self.obs_space, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                              lr=self.lr, eps=self.opti_eps,
                                              weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                               lr=self.critic_lr,
                                               eps=self.opti_eps,
                                               weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                    available_actions=None, comm_graphs=None, deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                    rnn_states_actor,
                                                                    masks,
                                                                    comm_graphs,
                                                                    available_actions,
                                                                    deterministic)

        values, rnn_states_critic = self.critic(obs, rnn_states_critic, masks, comm_graphs)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, rnn_states_critic, masks, comm_graphs=None):
        values, _ = self.critic(obs, rnn_states_critic, masks, comm_graphs)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                        available_actions=None, comm_graphs=None, active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     comm_graphs,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(obs, rnn_states_critic, masks, comm_graphs)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, comm_graphs=None, deterministic=False):
        actions, action_log_probs, rnn_states_actor = \
            self.actor(obs, rnn_states_actor, masks, comm_graphs, available_actions, deterministic)
        return actions, rnn_states_actor

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()