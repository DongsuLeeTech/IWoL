import torch
import torch.nn as nn
import torch.nn.functional as F
from bidexhands.utils.util import get_gard_norm, huber_loss, mse_loss
from bidexhands.algorithms.marl.utils.valuenorm import ValueNorm
from bidexhands.algorithms.marl.utils.popart import PopArt
from bidexhands.algorithms.utils.util import check

class ImIWoL_Trainer:
    def __init__(self, config, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = config["clip_param"]
        self.ppo_epoch = config["ppo_epoch"]
        self.num_mini_batch = config["num_mini_batch"]
        self.data_chunk_length = config["data_chunk_length"]
        self.value_loss_coef = config["value_loss_coef"]
        self.entropy_coef = config["entropy_coef"]
        self.max_grad_norm = config["max_grad_norm"]       
        self.huber_delta = config["huber_delta"]
        self.world_dynamics_coef = config['world_dynamics_coef']
        self.interactive_coef = config['interactive_coef']

        self._use_recurrent_policy = config["use_recurrent_policy"]
        self._use_naive_recurrent = config["use_naive_recurrent_policy"]
        self._use_max_grad_norm = config["use_max_grad_norm"]
        self._use_clipped_value_loss = config["use_clipped_value_loss"]
        self._use_huber_loss = config["use_huber_loss"]
        self._use_popart = config["use_popart"]
        self._use_valuenorm = config["use_valuenorm"]
        self._use_value_active_masks = config["use_value_active_masks"]
        self._use_policy_active_masks = config["use_policy_active_masks"]
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, comm_graphs_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy \
            = self.policy.evaluate_actions(share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions=available_actions_batch,
            comm_graphs=comm_graphs_batch,
            active_masks=active_masks_batch,)

        # actor update
        pred_state, state = self.policy.actor.latent_world(share_obs_batch, obs_batch, rnn_states_batch, masks_batch)
        world_dynamics_loss = F.mse_loss(pred_state, state)

        pred_message = self.policy.actor.latent_interactive(share_obs_batch, obs_batch, rnn_states_batch, masks_batch)
        message = self.policy.critic.message_forward(obs_batch, rnn_states_batch, masks_batch, comm_graphs_batch)
        interactive_loss = F.mse_loss(pred_message, message)

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                           dim=-1,
                                           keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef + world_dynamics_loss * self.world_dynamics_coef + interactive_loss * self.interactive_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        value_loss.backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, world_dynamics_loss, interactive_loss, \
               dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.clone()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = torch.nan
        mean_advantages = torch.mean(advantages_copy)
        std_advantages = torch.std(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['world_dynamics_loss'] = 0
        train_info['interactive_loss'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_comm_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, world_dynamics_loss, interactive_loss, \
                dist_entropy, actor_grad_norm, imp_weights = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['world_dynamics_loss'] += world_dynamics_loss.item()
                train_info['interactive_loss'] += interactive_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()