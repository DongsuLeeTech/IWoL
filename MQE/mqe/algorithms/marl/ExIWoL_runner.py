import time
import torch
import wandb
import os
import numpy as np
from mqe.algorithms.marl.utils.shared_buffer import SharedReplayBuffer
from mqe.algorithms.marl.ExIWoL_policy import ExIWoL_Policy
from mqe.algorithms.marl.ExIWoL_trainer import ExIWoL_Trainer

def _t2n(x):
    return x.detach().cpu().numpy()

class ExIWoL_Runner:
    def __init__(self, vec_env, config, model_dir=""):
        self.envs = vec_env
        self.eval_envs = vec_env
        
        # parameters
        self.project_name = config["project_name"]
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
        # video logging controls
        self.log_eval_video = config.get("log_eval_video", False)
        self.eval_video_fps = int(config.get("eval_video_fps", 15))
        self.eval_video_skip = int(config.get("eval_video_skip", 1))
        self.eval_camera_mode = config.get("eval_camera_mode", "follow")  # 'follow' | 'topdown' | 'fixed'
        self.eval_camera_topdown_height = float(config.get("eval_camera_topdown_height", 8.0))
        self.eval_camera_horizontal_fov = float(config.get("eval_camera_horizontal_fov", 90.0))
        # optional resolution override for camera creation
        self.eval_video_width_px = config.get("eval_video_width_px", None)
        self.eval_video_height_px = config.get("eval_video_height_px", None)
        self.eval_camera_env_index = int(config.get("eval_camera_env_index", 0))
        # fixed camera pose (optional)
        self.eval_camera_fixed_pos = config.get("eval_camera_fixed_pos", None)
        self.eval_camera_fixed_lookat = config.get("eval_camera_fixed_lookat", None)
        # debug flag
        self.eval_video_debug = bool(config.get("eval_video_debug", False))

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
            project=f"{self.project_name}",
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
            }
        )

        self.save_dir = str(self.run_dir + '/' + self.env_name + '/' + self.algorithm_name + '/models_seed{}'.format(self.seed))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # policy network
        self.policy = ExIWoL_Policy(config,
                                   self.envs.observation_space,
                                   self.envs.share_observation_space,
                                   self.envs.action_space,
                                   device=self.device)

        if self.model_dir != "":
            self.restore()

        # algorithm
        self.trainer = ExIWoL_Trainer(config, self.policy, device=self.device)
        
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

        print(obs.shape)
        print(share_obs.shape)

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
        # Prepare video capture via MQE env camera (first env only)
        env_for_video = getattr(self.eval_envs, 'env', None)
        video_ready = False
        collected_frames = []
        camera_pose_initialized = False

        if self.log_eval_video and env_for_video is not None:
            # Warn if simulation camera is disabled at config level
            try:
                if getattr(env_for_video.cfg.sim, 'no_camera', False):
                    print('Warning: cfg.sim.no_camera is True; camera sensors may render black frames.')
                    try:
                        env_for_video.cfg.sim.no_camera = False
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                # Ensure a floating camera exists to capture frames headlessly
                if not hasattr(env_for_video, 'rendering_camera'):
                    # Set desired FOV before creating the camera (helper will honor it)
                    try:
                        setattr(env_for_video.cfg.env, 'recording_horizontal_fov', self.eval_camera_horizontal_fov)
                        if self.eval_video_width_px is not None:
                            setattr(env_for_video.cfg.env, 'recording_width_px', int(self.eval_video_width_px))
                        if self.eval_video_height_px is not None:
                            setattr(env_for_video.cfg.env, 'recording_height_px', int(self.eval_video_height_px))
                    except Exception:
                        pass
                    from mqe.utils.helpers import FloatingCameraSensor
                    env_for_video.rendering_camera = FloatingCameraSensor(env_for_video)
                video_ready = True
            except Exception:
                video_ready = False

        # Camera ready for video capture if available

        desired_episodes = int(self.eval_episodes)
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

        step_index = 0
        video_episode_done = False
        blank_frame_count = 0
        fallback_to_follow = False
        debug_frame_logged = False
        camera_pose_initialized = False
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

            # Capture a frame for video if camera is available
            if video_ready and (self.eval_video_skip <= 1 or (step_index % self.eval_video_skip == 0)) and (not video_episode_done):
                try:
                    # Position camera (fixed/topdown/follow). Allow fallback to follow on blanks.
                    cam_mode = 'follow' if fallback_to_follow else self.eval_camera_mode
                    if cam_mode == 'fixed':
                        if not camera_pose_initialized:
                            try:
                                # prefer explicit pos/lookat; otherwise use viewer defaults
                                pos = self.eval_camera_fixed_pos
                                lookat = self.eval_camera_fixed_lookat
                                if pos is None:
                                    pos = getattr(getattr(env_for_video, 'cfg', None), 'viewer', None)
                                    pos = getattr(pos, 'pos', None) if pos is not None else None
                                if lookat is None:
                                    v = getattr(getattr(env_for_video, 'cfg', None), 'viewer', None)
                                    lookat = getattr(v, 'lookat', None) if v is not None else None
                                if pos is None:
                                    pos = [2.0, 2.0, 2.0]
                                if lookat is None:
                                    lookat = [0.0, 0.0, 0.0]
                                env_for_video.rendering_camera.set_position(pos, lookat)
                            except Exception:
                                pass
                            camera_pose_initialized = True
                    elif cam_mode == 'topdown':
                        try:
                            env_for_video.rendering_camera.set_topdown_position(
                                center=None,
                                height=self.eval_camera_topdown_height,
                                env_index=self.eval_camera_env_index,
                            )
                        except Exception:
                            env_for_video.rendering_camera.set_position()
                    else:
                        env_for_video.rendering_camera.set_position()
                    frame = env_for_video.rendering_camera.get_observation()
                    frame = np.asarray(frame)
                    # Remove alpha if present and ensure uint8
                    if frame.shape[-1] > 3:
                        frame = frame[..., :3]
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    # One-time debug dump
                    if self.eval_video_debug and (not debug_frame_logged):
                        try:
                            stats = {
                                'shape': tuple(frame.shape),
                                'dtype': str(frame.dtype),
                                'min': int(frame.min()) if frame.size else -1,
                                'max': int(frame.max()) if frame.size else -1,
                                'mean': float(frame.mean()) if frame.size else 0.0,
                                'nonzero_ratio': float((frame>0).sum())/float(frame.size) if frame.size else 0.0,
                                'mode': cam_mode,
                                'env_index': int(self.eval_camera_env_index),
                                'height': float(self.eval_camera_topdown_height),
                                'fov': float(self.eval_camera_horizontal_fov),
                            }
                            wandb.log({'eval/debug_frame': wandb.Image(frame), 'eval/debug_stats': stats}, step=total_num_steps)
                            debug_frame_logged = True
                        except Exception:
                            pass

                    # Blank/near‑blank detection
                    if frame.max() == 0 or (frame.mean() < 1.0):
                        blank_frame_count += 1
                        # After a few blank frames in topdown mode, fallback to follow
                        if (cam_mode == 'topdown') and blank_frame_count >= 3:
                            fallback_to_follow = True
                    collected_frames.append(frame)
                except Exception:
                    pass
            step_index += 1

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

                # stop collecting video frames after the first episode of env 0 completes
                if video_ready and 0 in finished_ids.tolist() and (not video_episode_done):
                    video_episode_done = True

        eval_episode_rewards = torch.stack(eval_episode_rewards)  # (Episodes, n_agents)
        mean_r = eval_episode_rewards.mean()
        max_r = eval_episode_rewards.max()

        success_count = self.eval_envs.reward_buffer.get("success count", 0)
        success_rate = success_count / max(1, eval_total_episodes)

        # Package collected frames into a WandB video
        if video_ready and len(collected_frames) > 0:
            try:
                frames_np = np.stack([np.asarray(f) for f in collected_frames], axis=0)  # (T, H, W, C)
                # If frames look blank, warn to check graphics/camera init
                if frames_np.max() <= 1 and frames_np.dtype != np.uint8:
                    pass
                elif frames_np.max() == 0:
                    print("Warning: Captured frames are all zeros. Check graphics context and camera setup.")
                video_tensor = frames_np.transpose(0, 3, 1, 2)  # (T, C, H, W)
                video = wandb.Video(video_tensor, fps=self.eval_video_fps, format='mp4')
            except Exception:
                video = None
        else:
            video = None

        info = dict(
            eval_average_episode_rewards=mean_r,
            eval_max_episode_rewards=max_r,
            eval_success_rate=success_rate,
            eval_success_count=success_count,
            eval_total_episodes=eval_total_episodes,
        )
        if video is not None:
            info['eval/rendering'] = video
        print(f"[Eval] {info}")
        self.log_env(info, total_num_steps)