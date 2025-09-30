import time
import torch
import wandb
import os
import numpy as np
from mqe.algorithms.marl.utils.shared_buffer import SharedReplayBuffer
from mqe.algorithms.marl.ImIWoL_policy import ImIWoL_Policy
from mqe.algorithms.marl.ImIWoL_trainer import ImIWoL_Trainer

def _t2n(x):
    return x.detach().cpu().numpy()

class ImIWoL_Runner:
    def __init__(self, vec_env, config, model_dir=""):
        self.envs = vec_env
        self.eval_envs = vec_env
        
        # parameters
        self.env_name = config['task']
        self.algorithm_name = config["algorithm_name"]
        self.experiment_name = config["experiment_name"]
        self.use_centralized_V = config["use_centralized_V"]
        self.use_obs_instead_of_state = config["use_obs_instead_of_state"]
        self.num_env_steps = config["num_env_steps"]
        self.episode_length = config["episode_length"]
        self.n_rollout_threads = config["n_rollout_threads"]
        self.n_eval_rollout_threads = config["n_rollout_threads"]
        self.use_linear_lr_decay = config["use_linear_lr_decay"]
        self.hidden_size = config["hidden_size"]
        self.use_render = config["use_render"]
        self.recurrent_N = config["recurrent_N"]
        self.latent_size = config["latent_size"]
        
        # interval
        self.save_interval = config["save_interval"]
        self.use_eval = config["use_eval"]
        self.eval_interval = config.get("eval_interval", 100)
        self.eval_episodes = config["eval_episodes"]
        self.log_interval = config["log_interval"]

        self.seed = config["seed"]
        self.model_dir = model_dir

        self.num_agents = self.envs.num_agents
        self.device = config["device"]
        print(f'Using device in magic_runner.py: {self.device}')

        torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir + '/' + self.env_name + '/' + self.algorithm_name +'/logs_seed{}'.format(self.seed))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Initialize wandb
        wandb.init(
            project=f"quadruped",
            name=f"{self.algorithm_name}_seed{self.seed}",
            config={
                "algorithm": self.algorithm_name,
                "env_name": self.env_name,
                "num_agents": self.num_agents,
                "num_env_steps": self.num_env_steps,
                "episode_length": self.episode_length,
                "n_rollout_threads": self.n_rollout_threads,
                "seed": self.seed,
                "use_centralized_V": self.use_centralized_V,
                "latent_size": self.latent_size,
            }
        )

        self.save_dir = str(self.run_dir + '/' + self.env_name + '/' + self.algorithm_name + '/models_seed{}'.format(self.seed))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # policy network
        self.policy = ImIWoL_Policy(config,
                                   self.envs.observation_space,
                                   self.envs.share_observation_space,
                                   self.envs.action_space,
                                   device=self.device)

        if self.model_dir != "":
            self.restore()

        # algorithm
        self.trainer = ImIWoL_Trainer(config, self.policy, device=self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(config,
                                       self.num_agents,
                                       self.envs.observation_space,
                                       self.envs.share_observation_space,
                                       self.envs.action_space,
                                       self.device)

    def run(self):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if hasattr(self.eval_envs, 'reward_buffer'):
                self.eval_envs.reward_buffer["success reward"] = 0
                train_total_episodes = 0
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Observe reward and next obs
                obs, share_obs, rewards, dones, infos = self.envs.step(actions, use_privileged_obs=True)

                data = obs, share_obs, rewards, dones, infos, \
                       values.detach(), actions, action_log_probs.detach(), \
                       rnn_states.detach(), rnn_states_critic.detach()

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # Update overall success tracking
            train_total_episodes += self.n_rollout_threads
            train_success_count = self.envs.reward_buffer["success reward"] / 10
            train_success_rate = train_success_count / max(1, train_total_episodes)

            train_infos["train_success_rate"] = train_success_rate
            train_infos["train_success_count"] = train_success_count
            train_infos["train_total_episodes"] = train_total_episodes

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\nAlgo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                aver_episode_rewards = torch.mean(self.buffer.rewards).item() * self.episode_length
                print("some episodes done, average rewards: ", aver_episode_rewards)
                wandb.log({"train_episode_rewards": aver_episode_rewards}, step=total_num_steps)
                print("train_success_rate is {}.".format(train_success_rate))

            # eval
            if episode % self.eval_interval == 0 or episode == episodes - 1:
                self.eval(total_num_steps)


    def warmup(self):
        # reset env
        obs, share_obs = self.envs.reset(use_privileged_obs=True)
        # share_obs = obs
        if isinstance(share_obs, np.ndarray):
            share_obs = torch.from_numpy(share_obs).to(self.device)
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)

        # replay buffer
        self.buffer.share_obs[0].copy_(share_obs)
        self.buffer.obs[0].copy_(obs)

    def collect(self, step):
        """Collect rollouts for training."""
        # Get actions from policy
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            self.buffer.share_obs[step].reshape(self.n_rollout_threads * self.num_agents, -1),
            self.buffer.obs[step].reshape(self.n_rollout_threads * self.num_agents, -1),
            self.buffer.rnn_states[step].reshape(self.n_rollout_threads * self.num_agents, self.recurrent_N, -1),
            self.buffer.rnn_states_critic[step].reshape(self.n_rollout_threads * self.num_agents, self.recurrent_N, -1),
            self.buffer.masks[step].reshape(self.n_rollout_threads * self.num_agents, -1),
        )

        # Ensure actions has the correct shape [n_rollout_threads, num_agents, action_dim]
        if actions.dim() == 1:
            actions = actions.reshape(self.n_rollout_threads, self.num_agents, -1)
        elif actions.dim() == 2:
            actions = actions.reshape(self.n_rollout_threads, self.num_agents, -1)

        values = values.reshape(self.n_rollout_threads, self.num_agents, -1)
        action_log_probs = action_log_probs.reshape(self.n_rollout_threads, self.num_agents, -1)
        rnn_states = rnn_states.reshape(self.n_rollout_threads, self.num_agents, self.recurrent_N, -1)
        rnn_states_critic = rnn_states_critic.reshape(self.n_rollout_threads, self.num_agents, self.recurrent_N, -1)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # Convert dones to tensor if it's numpy array
        if isinstance(dones, np.ndarray):
            dones = torch.from_numpy(dones).to(self.device).reshape(self.n_rollout_threads, self.num_agents, -1)

        # Convert obs and share_obs to tensors if they are numpy arrays
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)
        if isinstance(share_obs, np.ndarray):
            share_obs = torch.from_numpy(share_obs).to(self.device)

        # Convert rewards to tensor if it's numpy array
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards).to(self.device)

        mask = dones.squeeze(-1).bool()  # (threads, agents) 2‑D
        rnn_states[mask] = 0
        rnn_states_critic[mask] = 0

        masks = torch.ones(self.n_rollout_threads, self.num_agents, 1, device=self.device)
        masks[dones == True] = 0

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                         actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(self.buffer.obs[-1].reshape(self.n_rollout_threads * self.num_agents,  -1),
                                                   self.buffer.rnn_states_critic[-1].reshape(self.n_rollout_threads * self.num_agents, self.recurrent_N, -1),
                                                   self.buffer.masks[-1].reshape(self.n_rollout_threads * self.num_agents, -1))
        next_values = next_values.reshape(self.n_rollout_threads, self.num_agents, -1).detach()
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm.pt")

    def restore(self):
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm.pt')
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            wandb.log({k: v}, step=total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if isinstance(v, torch.Tensor):
                wandb.log({k: torch.mean(v).item()}, step=total_num_steps)
            else:
                wandb.log({k: v}, step=total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        desired_episodes = self.n_eval_rollout_threads
        eval_total_episodes = 0
        episode_rewards = torch.zeros(
            self.n_eval_rollout_threads, self.num_agents, device=self.device
        )
        eval_episode_rewards = []

        obs = self.eval_envs.reset()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)

        rnn_states = torch.zeros(
            self.n_eval_rollout_threads, self.num_agents,
            self.recurrent_N, self.hidden_size, device=self.device
        )
        masks = torch.ones(self.n_eval_rollout_threads, self.num_agents, 1, device=self.device)

        while eval_total_episodes < desired_episodes:
            self.trainer.prep_rollout()
            actions, rnn_states = self.trainer.policy.act(obs.reshape(self.n_rollout_threads * self.num_agents, -1),
                                                 rnn_states.reshape(self.n_rollout_threads * self.num_agents, self.recurrent_N, -1),
                                                 masks.reshape(self.n_rollout_threads * self.num_agents, -1),
                                                 deterministic=True)
            if actions.dim() == 1:
                actions = actions.reshape(self.n_rollout_threads, self.num_agents, -1)
            elif actions.dim() == 2:
                actions = actions.reshape(self.n_rollout_threads, self.num_agents, -1)

            obs, rewards, dones, infos = self.eval_envs.step(actions.to(self.device))

            # Convert numpy arrays to tensors
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).to(self.device)
            if isinstance(rewards, np.ndarray):
                rewards = torch.from_numpy(rewards).to(self.device)
            if isinstance(dones, np.ndarray):
                dones = torch.from_numpy(dones).to(self.device)

            episode_rewards += rewards.reshape(self.n_eval_rollout_threads, self.num_agents)
            done_mask = torch.all(dones.squeeze(-1), dim=1)

            if done_mask.any():
                finished_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)

                eval_episode_rewards.extend(
                    episode_rewards[finished_ids].clone().cpu()
                )
                eval_total_episodes += len(finished_ids)

                rnn_states[finished_ids].zero_()
                episode_rewards[finished_ids].zero_()

        eval_episode_rewards = torch.stack(eval_episode_rewards)  # (Episodes, n_agents)
        mean_r = eval_episode_rewards.mean()
        max_r = eval_episode_rewards.max()

        success_count = self.eval_envs.reward_buffer.get("success count", 0)
        success_rate = success_count / max(1, eval_total_episodes)

        info = dict(
            eval_average_episode_rewards=mean_r,
            eval_max_episode_rewards=max_r,
            eval_success_rate=success_rate,
            eval_success_count=success_count,
            eval_total_episodes=eval_total_episodes,
        )
        print(f"[Eval] {info}")
        self.log_env(info, total_num_steps)