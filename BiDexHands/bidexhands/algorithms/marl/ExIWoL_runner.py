import time
import torch
import wandb
import os
import numpy as np
try:
    from isaacgym import gymapi, gymtorch  # for eval video fallback camera
except Exception:
    gymapi = None
    gymtorch = None
from bidexhands.algorithms.marl.utils.shared_buffer import SharedReplayBuffer
from bidexhands.algorithms.marl.ExIWoL_policy import ExIWoL_Policy
from bidexhands.algorithms.marl.ExIWoL_trainer import ExIWoL_Trainer


def _t2n(x):
    return x.detach().cpu().numpy()


class ExIWoL_Runner:
    def __init__(self, vec_env, config, model_dir=""):
        self.envs = vec_env
        self.eval_envs = vec_env

        # parameters
        self.project_name = config["project_name"]
        config['episode_length'] = vec_env.task.cfg["env"]["episodeLength"]
        self.env_name = vec_env.task.cfg["env"]["env_name"]
        self.algorithm_name = config["algorithm_name"]
        self.experiment_name = config["experiment_name"]
        self.use_centralized_V = config["use_centralized_V"]
        self.use_obs_instead_of_state = config["use_obs_instead_of_state"]
        self.num_env_steps = config["num_env_steps"]
        self.episode_length = vec_env.task.cfg["env"]["episodeLength"]
        self.n_rollout_threads = config["n_rollout_threads"]
        self.n_eval_rollout_threads = config["n_rollout_threads"]
        self.use_linear_lr_decay = config["use_linear_lr_decay"]
        self.hidden_size = config["hidden_size"]
        self.use_render = config["use_render"]
        self.recurrent_N = config["recurrent_N"]

        # video logging options (optional; default off)
        self.log_eval_video = bool(config.get("log_eval_video", False))
        self.eval_video_fps = int(config.get("eval_video_fps", 15))
        self.eval_video_skip = int(config.get("eval_video_skip", 1))
        self.eval_video_key = config.get("eval_video_key", "eval/rendering")
        # camera mode and pose helpers (aligned with ImIWoL)
        self.eval_camera_mode = str(config.get("eval_camera_mode", "fixed")).lower()  # fixed | topdown | follow
        self.eval_camera_topdown_height = float(config.get("eval_camera_topdown_height", 8.0))
        # which env index to follow for video, and minimum frames to record
        self.eval_camera_env_index = int(config.get("eval_camera_env_index", 0))
        self.eval_video_min_frames = int(config.get("eval_video_min_frames", 16))
        # camera intrinsics / resolution
        self.eval_camera_horizontal_fov = float(config.get("eval_camera_horizontal_fov", 90.0))
        self.eval_video_width_px = int(config.get("eval_video_width_px", 1280))
        self.eval_video_height_px = int(config.get("eval_video_height_px", 720))
        # prefer CPU image path to avoid GPU mapping errors unless explicitly enabled
        self.eval_camera_use_gpu_tensors = bool(config.get("eval_camera_use_gpu_tensors", False))
        # optional fixed camera pose
        self.eval_camera_fixed_pos = config.get("eval_camera_fixed_pos", None)
        self.eval_camera_fixed_lookat = config.get("eval_camera_fixed_lookat", None)
        # optional debug toggles
        self.eval_video_debug = bool(config.get("eval_video_debug", False))

        # interval
        self.save_interval = config["save_interval"]
        self.use_eval = config["use_eval"]
        self.eval_interval = config.get("eval_interval", 100)
        self.eval_episodes = config["eval_episodes"]
        self.log_interval = config["log_interval"]

        self.seed = self.envs.task.cfg["seed"]
        self.model_dir = model_dir

        self.num_agents = self.envs.num_agents
        self.device = f"cuda:{self.envs.task.device_id}" if "cuda" in str(self.envs.task.device) else "cpu"
        print(f'Using device in runner.py: {self.device}')

        torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.run_dir = config["run_dir"]
        self.log_dir = str(
            self.run_dir + '/' + self.env_name + '/' + self.algorithm_name + '/logs_seed{}'.format(self.seed))
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

        self.save_dir = str(
            self.run_dir + '/' + self.env_name + '/' + self.algorithm_name + '/models_seed{}'.format(self.seed))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # policy network
        self.policy = ExIWoL_Policy(config,
                                   self.envs.obs_space[0],
                                   self.envs.share_observation_space[0],
                                   self.envs.act_space[0],
                                   device=self.device)

        if self.model_dir != "":
            self.restore()

        # algorithm
        self.trainer = ExIWoL_Trainer(config, self.policy, device=self.device)

        # buffer
        self.buffer = SharedReplayBuffer(config,
                                         self.num_agents,
                                         self.envs.obs_space[0],
                                         self.envs.share_observation_space[0],
                                         self.envs.act_space[0],
                                         self.device)

    def run(self):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # Observe reward and next obs
                obs, share_obs, rewards, dones, infos, _ = self.envs.step(actions_env)

                data = obs, share_obs, rewards, dones, infos, \
                       values.detach(), actions, action_log_probs.detach(), \
                       rnn_states.detach(), rnn_states_critic.detach()

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

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

            # eval
            if episode % self.eval_interval == 0 or episode == episodes - 1:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()
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

        actions_env = actions.reshape(self.n_rollout_threads, self.num_agents * actions.shape[-1])

        # Ensure actions_env is contiguous in memory
        actions_env = actions_env.contiguous()

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

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
        next_values = self.trainer.policy.get_values(
            self.buffer.obs[-1].reshape(self.n_rollout_threads * self.num_agents, -1),
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

        # Prepare eval video capture buffers
        capture_video = bool(self.log_eval_video)
        captured_frames = []
        step_index = 0
        video_episode_done = False
        # debug capture for frame sizes
        debug_first_raw_shape = None
        debug_first_fmt_shape = None
        eval_success_eps = 0
        eval_episode_steps = []
        env_step_counters = torch.zeros(self.n_eval_rollout_threads, dtype=torch.long)

        obs, _, _ = self.eval_envs.reset()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)

        rnn_states = torch.zeros(
            self.n_eval_rollout_threads, self.num_agents,
            self.recurrent_N, self.hidden_size, device=self.device
        )
        masks = torch.ones(self.n_eval_rollout_threads, self.num_agents, 1, device=self.device)

        # Frame post-processing helper: normalize to (H, W, 3) uint8
        def _format_frame(arr, width, height):
            try:
                import numpy as _np
                a = _np.asarray(arr)
                if a.ndim == 1:
                    for ch in (4, 3, 1):
                        if a.size == height * width * ch:
                            a = a.reshape(height, width, ch)
                            break
                elif a.ndim == 2:
                    h, w = a.shape
                    if w == width * 4 and h == height:
                        a = a.reshape(height, width, 4)
                    elif h == height * 4 and w == width:
                        a = a.reshape(height, 4, width).transpose(0, 2, 1)
                elif a.ndim == 3:
                    if a.shape[0] == width and a.shape[1] == height:
                        a = a.transpose(1, 0, 2)
                if a.ndim == 2:
                    a = _np.repeat(a[:, :, None], 3, axis=2)
                if a.ndim == 3:
                    if a.shape[-1] == 1:
                        a = _np.repeat(a, 3, axis=2)
                    elif a.shape[-1] > 3:
                        a = a[..., :3]
                if a.ndim == 3 and (a.shape[0] != height or a.shape[1] != width):
                    try:
                        a = a.reshape(height, width, a.shape[-1])
                    except Exception:
                        pass
                if a.dtype != _np.uint8:
                    a = a.astype(_np.uint8)
                return a
            except Exception:
                return None

        while eval_total_episodes < desired_episodes:
            self.trainer.prep_rollout()
            actions, rnn_states = self.trainer.policy.act(obs.reshape(self.n_rollout_threads * self.num_agents, -1),
                                                          rnn_states.reshape(self.n_rollout_threads * self.num_agents,
                                                                             self.recurrent_N, -1),
                                                          masks.reshape(self.n_rollout_threads * self.num_agents, -1),
                                                          deterministic=True)

            actions_env = actions.reshape(self.n_rollout_threads, self.num_agents * actions.shape[-1])
            obs, _, rewards, dones, infos, _ = self.eval_envs.step(actions_env)

            # Capture a frame for video if supported by task
            if capture_video and (not video_episode_done):
                try:
                    # update camera sensors if available
                    task = getattr(self.eval_envs, 'task', None)
                    gym_api = getattr(task, 'gym', None)
                    sim = getattr(task, 'sim', None)
                    frame_np = None
                    if gym_api is not None and sim is not None and gymapi is not None:
                        try:
                            try:
                                gym_api.fetch_results(sim, True)
                            except Exception:
                                pass
                            try:
                                env_idx = int(self.eval_camera_env_index)
                            except Exception:
                                env_idx = 0
                            if env_idx < 0 or env_idx >= len(task.envs):
                                env_idx = 0
                            if not hasattr(task, '_logging_camera_handle'):
                                cam_props = gymapi.CameraProperties()
                                cam_props.width = int(self.eval_video_width_px)
                                cam_props.height = int(self.eval_video_height_px)
                                cam_props.enable_tensors = False
                                try:
                                    cam_props.horizontal_fov = float(self.eval_camera_horizontal_fov)
                                except Exception:
                                    pass
                                env_ptr = task.envs[env_idx]
                                task._logging_camera_env = env_ptr
                                task._logging_camera_handle = gym_api.create_camera_sensor(env_ptr, cam_props)
                                # set camera pose
                                try:
                                    env_name = getattr(self, 'env_name', str(getattr(getattr(task, 'cfg', {}), 'get', lambda k, d=None: d)('env_name')))
                                except Exception:
                                    env_name = None
                                cam_pos = gymapi.Vec3(0.25, 0.0, 1.0)
                                cam_look = gymapi.Vec3(-0.25, 0.0, 0.0)
                                try:
                                    mode = (self.eval_camera_mode or 'fixed').lower()
                                    if mode == 'topdown':
                                        cam_pos = gymapi.Vec3(0.0, 0.0, float(self.eval_camera_topdown_height))
                                        cam_look = gymapi.Vec3(0.0, 0.0, 0.0)
                                    else:
                                        name = str(env_name or '').lower()
                                        if 'two_catch_underarm' in name:
                                            cam_pos = gymapi.Vec3(0.25, -0.57, 0.75)
                                            cam_look = gymapi.Vec3(-0.24, -0.57, 0.0)
                                        elif 'block_stack' in name:
                                            cam_pos = gymapi.Vec3(0.25, 0.0, 1.0)
                                            cam_look = gymapi.Vec3(-0.25, 0.0, -0.5)
                                        elif 'catch_underarm' in name:
                                            cam_pos = gymapi.Vec3(0.25, -0.57, 0.85)
                                            cam_look = gymapi.Vec3(-0.24, -0.57, 0.0)
                                except Exception:
                                    pass
                                try:
                                    if isinstance(self.eval_camera_fixed_pos, (list, tuple)) and len(self.eval_camera_fixed_pos) == 3:
                                        cam_pos = gymapi.Vec3(float(self.eval_camera_fixed_pos[0]),
                                                              float(self.eval_camera_fixed_pos[1]),
                                                              float(self.eval_camera_fixed_pos[2]))
                                    if isinstance(self.eval_camera_fixed_lookat, (list, tuple)) and len(self.eval_camera_fixed_lookat) == 3:
                                        cam_look = gymapi.Vec3(float(self.eval_camera_fixed_lookat[0]),
                                                               float(self.eval_camera_fixed_lookat[1]),
                                                               float(self.eval_camera_fixed_lookat[2]))
                                except Exception:
                                    pass
                                gym_api.set_camera_location(task._logging_camera_handle, env_ptr, cam_pos, cam_look)
                                if self.eval_video_debug:
                                    print(f"[Eval][Debug] Created camera with props: width={cam_props.width}, height={cam_props.height}, fov={getattr(cam_props, 'horizontal_fov', None)}")
                            env0 = getattr(task, '_logging_camera_env', task.envs[env_idx])
                            cam_h = task._logging_camera_handle
                            try:
                                gym_api.step_graphics(sim)
                            except Exception:
                                pass
                            gym_api.render_all_camera_sensors(sim)
                            frame_np = None
                            if frame_np is None:
                                try:
                                    gym_api.step_graphics(sim)
                                except Exception:
                                    pass
                                img = gym_api.get_camera_image(sim, env0, cam_h, gymapi.IMAGE_COLOR)
                                if self.eval_video_debug and debug_first_raw_shape is None:
                                    try:
                                        shp = tuple(getattr(img, 'shape', ()))
                                    except Exception:
                                        shp = ()
                                    debug_first_raw_shape = shp
                                frame_np = _format_frame(img, self.eval_video_width_px, self.eval_video_height_px)
                                if self.eval_video_debug and debug_first_fmt_shape is None and frame_np is not None:
                                    debug_first_fmt_shape = tuple(frame_np.shape)
                        except Exception:
                            frame_np = None
                    if frame_np is not None:
                        if frame_np.dtype != np.uint8:
                            frame_np = frame_np.astype(np.uint8)
                        if (self.eval_video_skip <= 1) or (step_index % self.eval_video_skip == 0):
                            captured_frames.append(frame_np)
                    done_mask_cap = torch.all(dones.squeeze(-1), dim=1)
                    try:
                        env_idx = int(self.eval_camera_env_index)
                        if 0 <= env_idx < done_mask_cap.shape[0]:
                            done_this_env = bool(done_mask_cap[env_idx].item())
                        else:
                            done_this_env = bool(done_mask_cap.any().item())
                    except Exception:
                        done_this_env = bool(done_mask_cap.any())
                    if done_this_env and len(captured_frames) >= self.eval_video_min_frames:
                        video_episode_done = True
                except Exception:
                    capture_video = False

            # Convert numpy arrays to tensors
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).to(self.device)
            if isinstance(rewards, np.ndarray):
                rewards = torch.from_numpy(rewards).to(self.device)
            if isinstance(dones, np.ndarray):
                dones = torch.from_numpy(dones).to(self.device)

            episode_rewards += rewards.reshape(self.n_eval_rollout_threads, self.num_agents)
            done_mask = torch.all(dones.squeeze(-1), dim=1)
            env_step_counters += 1

            if done_mask.any():
                finished_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)

                eval_episode_rewards.extend(
                    episode_rewards[finished_ids].clone().cpu()
                )
                eval_total_episodes += len(finished_ids)
                try:
                    task = getattr(self.eval_envs, 'task', None)
                    if task is not None and hasattr(task, 'successes'):
                        succ = task.successes[finished_ids]
                        if isinstance(succ, torch.Tensor):
                            succ = succ.detach().cpu()
                        eval_success_eps += int((succ > 0).sum())
                except Exception:
                    pass
                try:
                    steps = env_step_counters[finished_ids].clone().cpu().tolist()
                    eval_episode_steps.extend(steps)
                    env_step_counters[finished_ids] = 0
                except Exception:
                    pass

                rnn_states[finished_ids].zero_()
                episode_rewards[finished_ids].zero_()

            step_index += 1

        eval_episode_rewards = torch.stack(eval_episode_rewards)
        mean_r = eval_episode_rewards.mean()
        max_r = eval_episode_rewards.max()

        info = dict(
            eval_average_episode_rewards=mean_r,
            eval_max_episode_rewards=max_r,
            eval_total_episodes=eval_total_episodes,
        )
        print(f"[Eval] {info}")

        # If we captured frames, package as wandb.Video and log
        if capture_video and len(captured_frames) > 0:
            print(f"[Eval] Captured frames: {len(captured_frames)} @ {self.eval_video_width_px}x{self.eval_video_height_px}")
            try:
                # Ensure each frame is (H, W, 3)
                norm_frames = []
                for f in captured_frames:
                    nf = f
                    if nf.ndim == 2:
                        nf = np.repeat(nf[:, :, None], 3, axis=2)
                    elif nf.ndim == 3 and nf.shape[-1] > 3:
                        nf = nf[..., :3]
                    if nf.shape[0] != self.eval_video_height_px or nf.shape[1] != self.eval_video_width_px:
                        try:
                            nf = nf.reshape(self.eval_video_height_px, self.eval_video_width_px, 3)
                        except Exception:
                            pass
                    norm_frames.append(nf.astype(np.uint8))
                frames_hwxc = np.stack(norm_frames, axis=0)  # (T, H, W, C)
                if self.eval_video_debug:
                    print(f"[Eval][Debug] first raw shape: {debug_first_raw_shape}, first formatted shape: {debug_first_fmt_shape}")
                    try:
                        rh, rw, rc = (-1, -1, -1)
                        if isinstance(debug_first_raw_shape, tuple) and len(debug_first_raw_shape) >= 2:
                            if len(debug_first_raw_shape) == 3:
                                rh, rw, rc = debug_first_raw_shape
                            elif len(debug_first_raw_shape) == 2:
                                rh, rw = debug_first_raw_shape
                        fh, fw, fc = (-1, -1, -1)
                        if isinstance(debug_first_fmt_shape, tuple) and len(debug_first_fmt_shape) == 3:
                            fh, fw, fc = debug_first_fmt_shape
                        wandb.log({
                            "eval/debug_raw_h": rh,
                            "eval/debug_raw_w": rw,
                            "eval/debug_raw_c": rc,
                            "eval/debug_fmt_h": fh,
                            "eval/debug_fmt_w": fw,
                            "eval/debug_fmt_c": fc,
                            "eval/debug_cfg_width": int(self.eval_video_width_px),
                            "eval/debug_cfg_height": int(self.eval_video_height_px),
                        }, step=total_num_steps)
                    except Exception:
                        pass
                # Convert to (T, C, H, W) for wandb.Video
                frames_tchw = frames_hwxc.transpose(0, 3, 1, 2)
                if frames_tchw.dtype != np.uint8:
                    frames_tchw = frames_tchw.astype(np.uint8)
                video = wandb.Video(frames_tchw, fps=self.eval_video_fps, format='mp4')
                wandb.log({self.eval_video_key: video}, step=total_num_steps)
            except Exception as e:
                print(f"[Eval] Video packaging failed: {e}")
        elif capture_video:
            print("[Eval] No frames captured for eval video; skipping upload.")
            try:
                task = getattr(self.eval_envs, 'task', None)
                if task is not None:
                    print(f"[Eval] Debug: num_envs={len(getattr(task,'envs',[]))}, camera_handle={'Y' if hasattr(task,'_logging_camera_handle') else 'N'}, env_idx={getattr(self,'eval_camera_env_index',None)}")
            except Exception:
                pass

        # Additional eval stats logging
        if eval_total_episodes > 0:
            success_rate = float(eval_success_eps) / float(max(1, eval_total_episodes))
            info.update({
                "eval/success_rate": success_rate,
                "eval/success_count": eval_success_eps,
                "eval/finished_episodes": eval_total_episodes,
            })
        if len(eval_episode_steps) > 0:
            import numpy as _np
            steps_arr = _np.array(eval_episode_steps)
            info.update({
                "eval/episode_step_mean": float(steps_arr.mean()),
                "eval/episode_step_max": int(steps_arr.max()),
                "eval/episode_step_min": int(steps_arr.min()),
            })

        self.log_env(info, total_num_steps)
