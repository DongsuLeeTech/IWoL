import gym
from gym import spaces
import numpy
import torch
from copy import copy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

class Go1SeesawWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(12 + self.num_agents,), dtype=float)
        self.share_observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(60,),
                                            dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[2, 0.5, 0.5],],], device=self.env.device).repeat(self.num_envs, self.num_agents, 1)
        self._succeeded = torch.zeros(self.num_envs, self.num_agents,
                                      dtype=torch.bool, device=self.device)
        # for hard setting of reward scales (not recommended)
        
        # self.height_reward_scale = 0
        # self.success_reward_scale = 0

        self.reward_buffer = {
            "height reward": 0,
            "contact punishment": 0,
            "x movement reward": 0,
            "y punishment": 0,
            "agent distance punishment": 0,
            "success reward": 0,
            "fall punishment": 0,
            "step count": 0,
            "success count": 0
        }

    def _init_extras(self, obs):
        return
    
    def reset(self, use_privileged_obs=True):
        obs_buf = self.env.reset()
        self._succeeded.zero_()
        self.reward_buffer["success count"] = 0

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)

        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1])], dim=2)

        if use_privileged_obs:
            # (a) base_quat, lin_vel, ang_vel
            quat = obs_buf.base_quat
            lin_vel = obs_buf.lin_vel
            ang_vel = obs_buf.ang_vel

            # (b) joint pos/vel
            dof_pos = obs_buf.dof_pos
            dof_vel = obs_buf.dof_vel
            last_act = obs_buf.last_action

            # concat all privileged features
            privileged_obs_buf = torch.cat([quat, lin_vel, ang_vel, dof_pos, dof_vel, last_act], dim=1).reshape(
                [self.env.num_envs, self.env.num_agents, -1])
            privileged_obs = torch.cat([obs, privileged_obs_buf], dim=2)
            return obs, privileged_obs
        else:
            return obs

    def step(self, action, use_privileged_obs=True):
        action = torch.clip(action, -1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        if getattr(self, "gate_pos", None) is None:
            self._init_extras(obs_buf)
        
        base_pos = obs_buf.base_pos
        base_rpy = obs_buf.base_rpy
        base_info = torch.cat([base_pos, base_rpy], dim=1).reshape([self.env.num_envs, self.env.num_agents, -1])
        obs = torch.cat([self.obs_ids, base_info, torch.flip(base_info, [1])], dim=2)

        if use_privileged_obs:
            # (a) base_quat, lin_vel, ang_vel
            quat = obs_buf.base_quat
            lin_vel = obs_buf.lin_vel
            ang_vel = obs_buf.ang_vel

            # (b) joint pos/vel
            dof_pos = obs_buf.dof_pos
            dof_vel = obs_buf.dof_vel
            last_act = obs_buf.last_action

            # concat all privileged features
            privileged_obs_buf = torch.cat([quat, lin_vel, ang_vel, dof_pos, dof_vel, last_act], dim=1).reshape(
                [self.env.num_envs, self.env.num_agents, -1])
            privileged_obs = torch.cat([obs, privileged_obs_buf], dim=2)

        self.reward_buffer["step count"] += 1
        reward = torch.zeros([self.env.num_envs, 1], device=self.env.device)

        # x movement reward
        if self.x_movement_reward_scale != 0:
            x_pos = base_pos[:, 0].reshape(self.num_envs, -1)

            if not hasattr(self, "last_x_pos"):
                self.last_x_pos = copy(x_pos)
            
            x_reward = (x_pos - self.last_x_pos).sum(dim=1, keepdim=True)
            x_reward[self.env.reset_ids] = 0

            x_reward *= self.x_movement_reward_scale
            reward += x_reward

            self.last_x_pos = copy(x_pos)

            self.reward_buffer["x movement reward"] += torch.sum(x_reward).cpu()

        # height reward
        if self.height_reward_scale != 0:
            height_reward = self.height_reward_scale * (base_pos[:, 2].reshape(self.num_envs, -1).sum(dim=1) - 0.56)
            reward[:, 0] += height_reward
            self.reward_buffer["height reward"] += torch.sum(height_reward).cpu()

        # y punishment
        if self.y_punishment_scale != 0:
            y_punishment = self.y_punishment_scale * ((base_pos[:, 1].reshape(self.num_envs, -1) ** 2).sum(dim=1) - 0.5)
            reward[:, 0] += y_punishment
            self.reward_buffer["y punishment"] += torch.sum(y_punishment).cpu()

        # contact punishment
        if self.contact_punishment_scale != 0:
            collide_reward = self.contact_punishment_scale * self.env.collide_buf
            reward += collide_reward.unsqueeze(1)
            self.reward_buffer["contact punishment"] += torch.sum(collide_reward).cpu()

        # agent distance punishment
        if self.agent_distance_punishment_scale != 0:
            agent_dis = (base_pos[:, :2] - torch.flip(base_pos[:, :2].reshape(self.num_envs, self.num_agents, 2), dims=[1,]).reshape(-1, 2)) ** 2
            agent_dis = agent_dis.sum(dim=1).reshape(self.num_envs, -1)[:, :1]
            agent_distance_punishment = self.agent_distance_punishment_scale  / agent_dis[agent_dis < 0.25]
            reward[agent_dis < 0.25] += agent_distance_punishment
            self.reward_buffer["agent distance punishment"] += torch.sum(agent_distance_punishment).cpu()

        # success reward
        if self.success_reward_scale != 0:
            success = (base_pos[:, 0] > 7.7) & (base_pos[:, 2] > 1.3)
            success_mask = success.reshape(self.num_envs, self.num_agents)
            new_success = success_mask & (~self._succeeded)

            self._succeeded |= success_mask
            success_reward = self.success_reward_scale * success_mask.sum(dim=1)
            reward[:, 0] += success_reward
            self.reward_buffer["success reward"] += torch.sum(success_reward).cpu()

            self.reward_buffer["success count"] += new_success.sum().item()

            # success = (base_pos[:, 0] > 7.7) * (base_pos[:, 2] > 1.3)
            # success_reward = self.success_reward_scale * success.reshape(self.num_envs, -1).sum(dim=1)
            # reward[:, 0] += success_reward
            # self.reward_buffer["success reward"] += torch.sum(success_reward).cpu()

        if self.fall_punishment_scale != 0:
            fall = self.env.r_term_buff | self.env.p_term_buff
            reward[fall, 0] += self.fall_punishment_scale
            self.reward_buffer["fall punishment"] += self.fall_punishment_scale * torch.sum(fall).cpu()
        
        reward = reward.repeat(1, self.num_agents)

        if use_privileged_obs: return obs, privileged_obs, reward, termination, info
        else: return obs, reward, termination, info