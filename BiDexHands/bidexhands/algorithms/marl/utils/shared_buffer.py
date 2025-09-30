import torch
import numpy as np
from bidexhands.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _comm_flatten(T, N, M, x):
    return x.reshape(T * N * M, *x.shape[3:])


def _cast(x):
    return x.permute(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _comm_cast(x):
    return x.permute(1, 0, 2, 3).reshape(-1, *x.shape[2:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class SharedReplayBuffer(object):
    def __init__(self, config, num_agents, obs_space, cent_obs_space, act_space, device):
        self.episode_length = config["episode_length"]
        self.n_rollout_threads = config["n_rollout_threads"]
        self.hidden_size = config["hidden_size"]
        self.recurrent_N = config["recurrent_N"]
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self._use_gae = config["use_gae"]
        self._use_popart = config["use_popart"]
        self._use_valuenorm = config["use_valuenorm"]
        self._use_proper_time_limits = config["use_proper_time_limits"]
        self.device = device
        self.num_agents = num_agents

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = torch.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),
                                  dtype=torch.float32, device=device)
        self.obs = torch.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), 
                             dtype=torch.float32, device=device)

        self.rnn_states = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size),
            dtype=torch.float32, device=device)
        self.rnn_states_critic = torch.zeros_like(self.rnn_states)

        self.value_preds = torch.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), 
            dtype=torch.float32, device=device)
        self.returns = torch.zeros_like(self.value_preds)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = torch.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
                                             dtype=torch.float32, device=device)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = torch.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), 
            dtype=torch.float32, device=device)
        self.action_log_probs = torch.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), 
            dtype=torch.float32, device=device)
        self.rewards = torch.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), 
            dtype=torch.float32, device=device)

        self.masks = torch.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), 
                              dtype=torch.float32, device=device)
        self.bad_masks = torch.ones_like(self.masks)
        self.active_masks = torch.ones_like(self.masks)
        self.comm_graphs = torch.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, num_agents),
                                    dtype=torch.float32, device=device)

        self.step = 0

    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None,
               comm_graphs=None):
        self.share_obs[self.step + 1].copy_(share_obs)
        self.obs[self.step + 1].copy_(obs)
        self.rnn_states[self.step + 1].copy_(rnn_states_actor)
        self.rnn_states_critic[self.step + 1].copy_(rnn_states_critic)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        if bad_masks is not None:
            self.bad_masks[self.step + 1].copy_(bad_masks)
        if active_masks is not None:
            self.active_masks[self.step + 1].copy_(active_masks)
        if available_actions is not None:
            self.available_actions[self.step + 1].copy_(available_actions)
        if comm_graphs is not None:
            self.comm_graphs[self.step + 1].copy_(comm_graphs)

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.share_obs[self.step].copy_(share_obs)
        self.obs[self.step].copy_(obs)
        self.rnn_states[self.step + 1].copy_(rnn_states)
        self.rnn_states_critic[self.step + 1].copy_(rnn_states_critic)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        if bad_masks is not None:
            self.bad_masks[self.step + 1].copy_(bad_masks)
        if active_masks is not None:
            self.active_masks[self.step].copy_(active_masks)
        if available_actions is not None:
            self.available_actions[self.step].copy_(available_actions)

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0].copy_(self.share_obs[-1])
        self.obs[0].copy_(self.obs[-1])
        self.rnn_states[0].copy_(self.rnn_states[-1])
        self.rnn_states_critic[0].copy_(self.rnn_states_critic[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.active_masks[0].copy_(self.active_masks[-1])
        if self.available_actions is not None:
            self.available_actions[0].copy_(self.available_actions[-1])
        self.comm_graphs[0].copy_(self.comm_graphs[-1])

    def chooseafter_update(self):
        self.rnn_states[0].copy_(self.rnn_states[-1])
        self.rnn_states_critic[0].copy_(self.rnn_states_critic[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size, device=self.device)
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents, num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size, device=self.device)

        share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(-1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(-1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(-1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(share_obs[:-1, ind])
                obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.episode_length, num_envs_per_batch
            share_obs_batch = torch.stack(share_obs_batch, 1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = torch.stack(available_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            active_masks_batch = torch.stack(active_masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            rnn_states_batch = torch.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = torch.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:-1].permute(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:-1].permute(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        rnn_states = self.rnn_states[:-1].permute(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].permute(1, 2, 0, 3, 4).reshape(-1,
                                                                                         *self.rnn_states_critic.shape[
                                                                                          3:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            share_obs_batch = torch.stack(share_obs_batch, axis=1)
            obs_batch = torch.stack(obs_batch, axis=1)

            actions_batch = torch.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = torch.stack(available_actions_batch, axis=1)
            value_preds_batch = torch.stack(value_preds_batch, axis=1)
            return_batch = torch.stack(return_batch, axis=1)
            masks_batch = torch.stack(masks_batch, axis=1)
            active_masks_batch = torch.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, axis=1)
            adv_targ = torch.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = torch.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = torch.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch

    def recurrent_comm_generator(self, advantages, num_mini_batch, data_chunk_length):
         # e.g., episode length 1000, num_rollout_threads 2, num_agents 30, data_chunk_length 10, num_mini_batch 1
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        # batch_size: 2000
        batch_size = n_rollout_threads * episode_length
        # data_chunks: 200
        data_chunks = batch_size // data_chunk_length
        # mini_batch_size: 200
        mini_batch_size = data_chunks // num_mini_batch
        # generate a random permutation of integers
        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        # make sure that the sampled data chunk is from the same env (thread)
        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:-1].permute(1, 0, 2, 3, 4, 5).reshape(-1, *self.share_obs.shape[2:])
            obs = self.obs[:-1].permute(1, 0, 2, 3, 4, 5).reshape(-1, *self.obs.shape[2:])
        else:
            # [num_rollout_threads*episode_length, num_agents, obs_dim]
            share_obs = _comm_cast(self.share_obs[:-1])
            obs = _comm_cast(self.obs[:-1])
        # [num_rollout_threads*episode_length, num_agents, dim]
        actions = _comm_cast(self.actions)
        action_log_probs = _comm_cast(self.action_log_probs)
        advantages = _comm_cast(advantages)
        value_preds = _comm_cast(self.value_preds[:-1])
        returns = _comm_cast(self.returns[:-1])
        masks = _comm_cast(self.masks[:-1])
        active_masks = _comm_cast(self.active_masks[:-1])
        # comm_graphs: [num_rollout_threads*episode_length, num_agents, num_agents]
        comm_graphs = _comm_cast(self.comm_graphs[:-1])
        # rnn_states: [num_rollout_threads*episode_length, num_agents, recurrent_N, hidden_size]
        rnn_states = self.rnn_states[:-1].permute(1, 0, 2, 3, 4).reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].permute(1, 0, 2, 3, 4).reshape(-1,
                                                                                         *self.rnn_states_critic.shape[
                                                                                          2:])

        if self.available_actions is not None:
            available_actions = _comm_cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            comm_graphs_batch = []

            for index in indices:
                ind = index * data_chunk_length
                # [data_chunk_length, num_agents, dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                comm_graphs_batch.append(comm_graphs[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # [num_agents, recurrent_N, hidden_size]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N, M = data_chunk_length, mini_batch_size, self.num_agents

            # [L, N, num_agents, dim]
            share_obs_batch = torch.stack(share_obs_batch, axis=1)
            obs_batch = torch.stack(obs_batch, axis=1)

            actions_batch = torch.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = torch.stack(available_actions_batch, axis=1)
            value_preds_batch = torch.stack(value_preds_batch, axis=1)
            return_batch = torch.stack(return_batch, axis=1)
            masks_batch = torch.stack(masks_batch, axis=1)
            active_masks_batch = torch.stack(active_masks_batch, axis=1)
            comm_graphs_batch = torch.stack(comm_graphs_batch, axis=1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, axis=1)
            adv_targ = torch.stack(adv_targ, axis=1)
            # rnn_states: [N, num_agents, recurrent_N, hidden_size]
            rnn_states_batch = torch.stack(rnn_states_batch)
            rnn_states_critic_batch = torch.stack(rnn_states_critic_batch)
            # flatten rnn_states: [N * num_agents, recurrent_N, hidden_size]
            rnn_states_batch = _flatten(N, M, rnn_states_batch)
            rnn_states_critic_batch = _flatten(N, M, rnn_states_critic_batch)

            # Flatten the (L, N, num_agents, ...) from_numpys to (L * N * num_agents, ...)
            share_obs_batch = _comm_flatten(L, N, M, share_obs_batch)
            obs_batch = _comm_flatten(L, N, M, obs_batch)
            actions_batch = _comm_flatten(L, N, M, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _comm_flatten(L, N, M, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _comm_flatten(L, N, M, value_preds_batch)
            return_batch = _comm_flatten(L, N, M, return_batch)
            masks_batch = _comm_flatten(L, N, M, masks_batch)
            active_masks_batch = _comm_flatten(L, N, M, active_masks_batch)
            old_action_log_probs_batch = _comm_flatten(L, N, M, old_action_log_probs_batch)
            adv_targ = _comm_flatten(L, N, M, adv_targ)

            # flatten comm_graphs_batch: (L * N, num_agents, num_agents)
            comm_graphs_batch = _flatten(L, N, comm_graphs_batch)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch, comm_graphs_batch

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        # [episode_length * n_rollouts_threads, num_agents, dim]
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states = rnn_states[rows, cols]
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        rnn_states_critic = rnn_states_critic[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]
        # [episode_length * n_rollouts_threads, num_agents, num_agents]
        comm_graphs = self.comm_graphs[:-1].reshape(-1, *self.comm_graphs.shape[2:])
        comm_graphs = comm_graphs[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(-1, *rnn_states_critic.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])
            comm_graphs_batch = comm_graphs[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch, comm_graphs_batch