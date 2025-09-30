import gym
from gym import spaces
import numpy as np
import torch
from copy import copy, deepcopy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

from isaacgym.torch_utils import *


# tensor type
def rotation_matrix_2D(theta):
    theta = theta.float()
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    rotation_matrices = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=1),
        torch.stack([sin_theta, cos_theta], dim=1)
    ], dim=1)

    return rotation_matrices


def euler_to_quaternion_tensor(euler_angles):
    roll = euler_angles[:, 0]
    pitch = euler_angles[:, 1]
    yaw = euler_angles[:, 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = torch.stack([qx, qy, qz, qw], dim=1)
    return quaternion


def normalize_rpy(box_rpy):
    box_rpy = box_rpy % (2 * torch.pi)

    return box_rpy


class Go1PushTWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        if getattr(self.cfg.goal, "general_dist", False):
            self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3 + 3 * self.num_agents,),
                                                dtype=float)
            self.share_observation_space = spaces.Box(low=-float('inf'), high=float('inf'),
                                                      shape=(62 + 2 * self.cfg.env.num_npcs,),
                                                      dtype=float)
            pass
        else:
            self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(2 + 3 * self.num_agents,),
                                                dtype=float)
            self.share_observation_space = spaces.Box(low=-float('inf'), high=float('inf'),
                                                      shape=(62 + 2 * self.cfg.env.num_npcs,),
                                                      dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[0.5, 0.5, 0.5], ], ], device="cuda").repeat(self.num_envs, self.num_agents,
                                                                                        1)

        # for hard setting of reward scales (not recommended)

        self.approach_reward_scale = self.cfg.rewards.scales.approach_reward_scale
        self.target_reward_scale = self.cfg.rewards.scales.target_reward_scale
        self.reach_target_reward_scale = self.cfg.rewards.scales.reach_target_reward_scale
        self.collision_punishment_scale = self.cfg.rewards.scales.collision_punishment_scale
        self.push_reward_scale = self.cfg.rewards.scales.push_reward_scale
        self.ocb_reward_scale = self.cfg.rewards.scales.ocb_reward_scale
        self.exception_punishment_scale = self.cfg.rewards.scales.exception_punishment_scale

        self.reward_buffer = {
            "distance_to_target_reward": 0,
            "exception_punishment": 0,
            "approach_to_box_reward": 0,
            "collision_punishment": 0,
            "reach_target_reward": 0,
            "push_reward": 0,
            "ocb_reward": 0,
            "step_count": 0,
            "success count": 0
        }
        self._env_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _init_extras(self, obs):
        return
        # self.gate_pos = obs.env_info["gate_deviation"]
        # self.gate_pos[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
        # self.gate_pos = self.gate_pos.unsqueeze(1).repeat(1, self.num_agents, 1)
        # self.gate_distance = self.gate_pos.reshape(-1, 2)[:, 0]

    def calc_normal_vector_for_obc_reward(self, vertex_list, pos_tensor):
        pos_tensor = pos_tensor.to(self.device)
        vertices = torch.tensor(vertex_list, device=self.device).float()
        num_vertices = vertices.shape[0]

        edges = torch.roll(vertices, -1, dims=0) - vertices
        vp = pos_tensor[:, None, :] - vertices[None, :, :]

        edges_expanded = edges[None, :, :].repeat(pos_tensor.shape[0], 1, 1)
        edge_lengths = torch.norm(edges_expanded, dim=2, keepdim=True)
        edge_unit = edges_expanded / edge_lengths
        edge_normals = torch.stack([-edge_unit[:, :, 1], edge_unit[:, :, 0]], dim=2)

        cross_prod = torch.abs(vp[:, :, 0] * edge_unit[:, :, 1] - vp[:, :, 1] * edge_unit[:, :, 0])
        dot_product1 = (vp * edges_expanded).sum(dim=2)
        dot_product2 = (torch.roll(vp, -1, dims=1) * edges_expanded).sum(dim=2)

        on_segment = (dot_product1 >= 0) & (dot_product2 <= 0)
        dist_to_line = torch.where(on_segment, cross_prod, torch.tensor(float('inf'), device=self.device))

        dist_to_vertex1 = torch.norm(vp, dim=2)
        dist_to_vertex2 = torch.norm(pos_tensor[:, None, :] - torch.roll(vertices, -1, dims=0)[None, :, :], dim=2)

        min_dist_each_edge, indices = torch.min(torch.stack([dist_to_line, dist_to_vertex1, dist_to_vertex2], dim=-1),
                                                dim=2)
        min_dist, indices = torch.min(min_dist_each_edge, dim=1)
        selected_normals = edge_normals[0][indices]

        return selected_normals

    def reset(self, next_target_pos=None, use_privileged_obs=True):
        if getattr(self.cfg.goal, "received_goal_pos", False):
            if next_target_pos == None:
                pass
                # raise ValueError("next_target_pos is required when received_goal_pos is True")
            self.next_target_pos = next_target_pos

        obs_buf = self.env.reset()
        self.reward_buffer["success count"] = 0
        self._env_succeeded.zero_()

        # get agent state
        base_pos = deepcopy(obs_buf.base_pos)
        base_rpy = deepcopy(obs_buf.base_rpy)
        # get box state and target pos
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:, 0, :] - self.env.env_origins
        target_pos = npc_pos[:, 1, :] - self.env.env_origins
        box_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 3:7]
        box_rpy = torch.stack(get_euler_xyz(box_qyaternion), dim=1)
        target_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1, 3:7]
        target_rpy = torch.stack(get_euler_xyz(target_qyaternion), dim=1)

        # rotate box state and target pos to agent's local state
        box_pos = box_pos.repeat_interleave(self.num_agents, dim=0)
        target_pos = target_pos.repeat_interleave(self.num_agents, dim=0)
        box_rpy = box_rpy.repeat_interleave(self.num_agents, dim=0)
        target_rpy = target_rpy.repeat_interleave(self.num_agents, dim=0)
        rotated_box_pos = torch.stack([(box_pos[:, 0] - base_pos[:, 0]) * torch.cos(-base_rpy[:, 2]) - (
                    box_pos[:, 1] - base_pos[:, 1]) * torch.sin(-base_rpy[:, 2]),
                                       (box_pos[:, 0] - base_pos[:, 0]) * torch.sin(-base_rpy[:, 2]) + (
                                                   box_pos[:, 1] - base_pos[:, 1]) * torch.cos(-base_rpy[:, 2]),
                                       box_pos[:, 2]], dim=1)
        rotated_target_pos = torch.stack([(target_pos[:, 0] - base_pos[:, 0]) * torch.cos(-base_rpy[:, 2]) - (
                    target_pos[:, 1] - base_pos[:, 1]) * torch.sin(-base_rpy[:, 2]),
                                          (target_pos[:, 0] - base_pos[:, 0]) * torch.sin(-base_rpy[:, 2]) + (
                                                      target_pos[:, 1] - base_pos[:, 1]) * torch.cos(-base_rpy[:, 2]),
                                          target_pos[:, 2]], dim=1)
        rotated_box_rpy = deepcopy(box_rpy)
        rotated_box_rpy[:, 2] = box_rpy[:, 2] - base_rpy[:, 2]
        rotated_box_rpy = normalize_rpy(rotated_box_rpy)
        rotated_target_rpy = deepcopy(target_rpy)
        rotated_target_rpy[:, 2] = target_rpy[:, 2] - base_rpy[:, 2]
        rotated_target_rpy = normalize_rpy(rotated_target_rpy)
        rotated_box_pos = rotated_box_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_box_rpy = rotated_box_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_target_pos = rotated_target_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_target_rpy = rotated_target_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])

        # rotate other agents' state to agent's local state
        base_pos = base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_rpy = base_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_info = torch.cat([base_pos, base_rpy], dim=2)
        all_base_info = []
        if self.num_agents != 1:
            for i in range(1, self.env.num_agents):
                other_base_info = deepcopy(torch.roll(base_info, i, dims=1))
                # roate other agents' state to agent's local state
                other_base_pos = torch.stack([(other_base_info[:, :, 0] - base_pos[:, :, 0]) * torch.cos(
                    -base_rpy[:, :, 2]) - (other_base_info[:, :, 1] - base_pos[:, :, 1]) * torch.sin(
                    -base_rpy[:, :, 2]),
                                              (other_base_info[:, :, 0] - base_pos[:, :, 0]) * torch.sin(
                                                  -base_rpy[:, :, 2]) + (
                                                          other_base_info[:, :, 1] - base_pos[:, :, 1]) * torch.cos(
                                                  -base_rpy[:, :, 2]),
                                              other_base_info[:, :, 2]], dim=2)
                other_base_rpy = deepcopy(other_base_info[:, :, 3:6])
                other_base_rpy[:, :, 2] = other_base_info[:, :, 5] - base_rpy[:, :, 2]
                other_base_rpy = normalize_rpy(other_base_rpy)
                other_base_info = torch.cat([other_base_pos[:, :, :2], other_base_rpy[:, :, 2].unsqueeze(2)], dim=2)
                all_base_info.append(other_base_info)
            all_base_info = torch.cat(all_base_info, dim=2)

        if getattr(self.cfg.goal, "general_dist", False):
            obs = torch.cat(
                [rotated_target_pos[:, :, :2], rotated_target_rpy[:, :, 2].unsqueeze(2), rotated_box_pos[:, :, :2],
                 rotated_box_rpy[:, :, 2].unsqueeze(2), all_base_info], dim=2)
        else:
            if all_base_info == []:
                obs = torch.cat(
                    [rotated_target_pos[:, :, :2], rotated_box_pos[:, :, :2], rotated_box_rpy[:, :, 2].unsqueeze(2)],
                    dim=2)
            else:
                obs = torch.cat(
                    [rotated_target_pos[:, :, :2], rotated_box_pos[:, :, :2], rotated_box_rpy[:, :, 2].unsqueeze(2),
                     all_base_info], dim=2)

        self.last_box_state = None

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

    def step(self, action, next_target_pos=None, use_privileged_obs=True):
        if next_target_pos is not None:
            assert next_target_pos.shape == (self.num_envs, 3)
            assert self.cfg.generalize_obsersation.rotate_obs
            assert self.cfg.goal.received_goal_pos

        if getattr(self.cfg.goal, "received_goal_pos", False):
            if next_target_pos is None:
                raise ValueError("next_target_pos is required when received_goal_pos is True")
            self.env.next_target_pos = next_target_pos

        action = torch.clip(action, -1.0, 1.0)
        if getattr(self.cfg.goal, "received_goal_pos", False):
            if torch.any(self.env.stop_buf):
                action[self.env.stop_buf] = torch.tensor([0., 0., 0.], device=self.device).repeat(
                    self.stop_buf.sum().item(), self.num_agents, 1)
        # set static action
        # action = torch.tensor([[1.0, 0.0, 0.0]], device="cuda").repeat(self.num_envs, 1, 1)
        obs_buf, _, termination, info = self.env.step(
            (action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        # get agent state
        base_pos = deepcopy(obs_buf.base_pos)
        base_rpy = deepcopy(obs_buf.base_rpy)
        # get box state and target pos
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:, 0, :] - self.env.env_origins
        target_pos = npc_pos[:, 1, :] - self.env.env_origins
        box_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 3:7]
        box_rpy = torch.stack(get_euler_xyz(box_qyaternion), dim=1)
        target_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1, 3:7]
        target_rpy = torch.stack(get_euler_xyz(target_qyaternion), dim=1)

        # rotate box state and target pos to agent's local state
        box_pos = box_pos.repeat_interleave(self.num_agents, dim=0)
        target_pos = target_pos.repeat_interleave(self.num_agents, dim=0)
        box_rpy = box_rpy.repeat_interleave(self.num_agents, dim=0)
        target_rpy = target_rpy.repeat_interleave(self.num_agents, dim=0)
        rotated_box_pos = torch.stack([(box_pos[:, 0] - base_pos[:, 0]) * torch.cos(-base_rpy[:, 2]) - (
                    box_pos[:, 1] - base_pos[:, 1]) * torch.sin(-base_rpy[:, 2]),
                                       (box_pos[:, 0] - base_pos[:, 0]) * torch.sin(-base_rpy[:, 2]) + (
                                                   box_pos[:, 1] - base_pos[:, 1]) * torch.cos(-base_rpy[:, 2]),
                                       box_pos[:, 2]], dim=1)
        rotated_target_pos = torch.stack([(target_pos[:, 0] - base_pos[:, 0]) * torch.cos(-base_rpy[:, 2]) - (
                    target_pos[:, 1] - base_pos[:, 1]) * torch.sin(-base_rpy[:, 2]),
                                          (target_pos[:, 0] - base_pos[:, 0]) * torch.sin(-base_rpy[:, 2]) + (
                                                      target_pos[:, 1] - base_pos[:, 1]) * torch.cos(-base_rpy[:, 2]),
                                          target_pos[:, 2]], dim=1)
        rotated_box_rpy = deepcopy(box_rpy)
        rotated_box_rpy[:, 2] = box_rpy[:, 2] - base_rpy[:, 2]
        rotated_box_rpy = normalize_rpy(rotated_box_rpy)
        rotated_target_rpy = deepcopy(target_rpy)
        rotated_target_rpy[:, 2] = target_rpy[:, 2] - base_rpy[:, 2]
        rotated_target_rpy = normalize_rpy(rotated_target_rpy)
        rotated_box_pos = rotated_box_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_box_rpy = rotated_box_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_target_pos = rotated_target_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_target_rpy = rotated_target_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])

        # rotate other agents' state to agent's local state
        base_pos = base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_rpy = base_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_info = torch.cat([base_pos, base_rpy], dim=2)
        all_base_info = []
        if self.num_agents != 1:
            for i in range(1, self.env.num_agents):
                other_base_info = deepcopy(torch.roll(base_info, i, dims=1))
                # roate other agents' state to agent's local state
                other_base_pos = torch.stack([(other_base_info[:, :, 0] - base_pos[:, :, 0]) * torch.cos(
                    -base_rpy[:, :, 2]) - (other_base_info[:, :, 1] - base_pos[:, :, 1]) * torch.sin(
                    -base_rpy[:, :, 2]),
                                              (other_base_info[:, :, 0] - base_pos[:, :, 0]) * torch.sin(
                                                  -base_rpy[:, :, 2]) + (
                                                          other_base_info[:, :, 1] - base_pos[:, :, 1]) * torch.cos(
                                                  -base_rpy[:, :, 2]),
                                              other_base_info[:, :, 2]], dim=2)
                other_base_rpy = deepcopy(other_base_info[:, :, 3:6])
                other_base_rpy[:, :, 2] = other_base_info[:, :, 5] - base_rpy[:, :, 2]
                other_base_rpy = normalize_rpy(other_base_rpy)
                other_base_info = torch.cat([other_base_pos[:, :, :2], other_base_rpy[:, :, 2].unsqueeze(2)], dim=2)
                all_base_info.append(other_base_info)
            all_base_info = torch.cat(all_base_info, dim=2)

        if getattr(self.cfg.goal, "general_dist", False):
            obs = torch.cat(
                [rotated_target_pos[:, :, :2], rotated_target_rpy[:, :, 2].unsqueeze(2), rotated_box_pos[:, :, :2],
                 rotated_box_rpy[:, :, 2].unsqueeze(2), all_base_info], dim=2)
        else:
            if all_base_info == []:
                obs = torch.cat(
                    [rotated_target_pos[:, :, :2], rotated_box_pos[:, :, :2], rotated_box_rpy[:, :, 2].unsqueeze(2)],
                    dim=2)
            else:
                obs = torch.cat(
                    [rotated_target_pos[:, :, :2], rotated_box_pos[:, :, :2], rotated_box_rpy[:, :, 2].unsqueeze(2),
                     all_base_info], dim=2)

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

        # get env_id which should be reseted, because of nan or inf in obs and reward
        self.value_exception_buf = torch.isnan(obs).any(dim=2).any(dim=1) \
                                   | torch.isinf(obs).any(dim=2).any(dim=1) \

        # remove nan and inf in obs and reward
        obs[torch.isnan(obs)] = 0
        obs[torch.isinf(obs)] = 0

        # calculate reward
        box_state = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0]
        target_state = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1]
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:, 0, :] - self.env.env_origins
        target_pos = npc_pos[:, 1, :] - self.env.env_origins
        box_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 3:7]
        box_rpy = torch.stack(get_euler_xyz(box_qyaternion), dim=1)
        target_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1, 3:7]
        target_rpy = torch.stack(get_euler_xyz(target_qyaternion), dim=1)

        base_pos = obs_buf.base_pos  # (env_num, agent_num, 3)
        base_vel = obs_buf.lin_vel  # (env_num, agent_num, 3)
        base_rpy = obs_buf.base_rpy  # (env_num, agent_num, 3)
        base_pos = base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_vel = base_vel.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_rpy = base_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])

        # occlude nan or inf
        box_pos[torch.isnan(box_pos)] = 0
        box_pos[torch.isinf(box_pos)] = 0
        target_pos[torch.isnan(target_pos)] = 0
        target_pos[torch.isinf(target_pos)] = 0
        base_pos[torch.isnan(base_pos)] = 0
        base_pos[torch.isinf(base_pos)] = 0
        box_rpy[torch.isnan(box_rpy)] = 0
        box_rpy[torch.isinf(box_rpy)] = 0

        self.reward_buffer["step_count"] += 1
        reward = torch.zeros([self.env.num_envs, self.num_agents], device=self.env.device)


        # calculate reach target reward and set finish task termination
        if self.reach_target_reward_scale != 0:
            reward[self.finished_buf, :] += self.reach_target_reward_scale
            self.reward_buffer["reach_target_reward"] += self.reach_target_reward_scale * self.finished_buf.sum().item()

        new_success = self.finished_buf & (~self._env_succeeded)
        self._env_succeeded |= self.finished_buf
        self.reward_buffer["success count"] += new_success.sum().item()

        # calculate exception punishment
        if self.exception_punishment_scale != 0:
            reward[self.exception_buf, :] += self.exception_punishment_scale
            reward[self.value_exception_buf, :] += self.exception_punishment_scale
            # reward[self.time_out_buf, :] += self.exception_punishment_scale
            self.reward_buffer["exception_punishment"] += self.exception_punishment_scale * \
                                                          (
                                                                      self.exception_buf.sum().item() + self.value_exception_buf.sum().item())

        # calculate distance from current_box_pos to target_box_pos reward
        if self.target_reward_scale != 0:
            if self.last_box_state is None:
                self.last_box_state = copy(box_state)
            past_distance = self.env.dist_calculator.cal_dist(self.last_box_state, target_state)
            distance = self.env.dist_calculator.cal_dist(box_state, target_state)
            distance_reward = self.target_reward_scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)
            reward[:, :] += distance_reward.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["distance_to_target_reward"] += torch.sum(distance_reward).cpu()

        # calculate distance from each robot to box reward
        if self.approach_reward_scale != 0:
            reward_logger = []
            for i in range(self.num_agents):
                distance = torch.norm(box_pos - base_pos[:, i, :], dim=1, keepdim=True)
                distance_reward = (-(distance + 0.5) ** 2) * self.approach_reward_scale
                reward_logger.append(torch.sum(distance_reward).cpu())
                reward[:, i] += distance_reward.squeeze(-1)
            self.reward_buffer["approach_to_box_reward"] += np.sum(np.array(reward_logger))

            # calculate collision punishment
        if self.collision_punishment_scale != 0:
            punishment_logger = []
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    distance = torch.norm(base_pos[:, i, :] - base_pos[:, j, :], dim=1, keepdim=True)
                    collsion_punishment = (1 / (0.02 + distance / 3)) * self.collision_punishment_scale
                    punishment_logger.append(torch.sum(collsion_punishment).cpu())
                    reward[:, i] += collsion_punishment.squeeze(-1)
                    reward[:, j] += collsion_punishment.squeeze(-1)
            self.reward_buffer["collision_punishment"] += np.sum(np.array(punishment_logger))

        # calculate push reward for each agent
        if self.push_reward_scale != 0:
            push_reward = torch.zeros((self.env.num_envs,), device=self.env.device)
            push_reward[torch.norm(self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 7:9],
                                   dim=1) > 0.1] = self.push_reward_scale
            reward[:, :] += push_reward.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["push_reward"] += torch.sum(push_reward).cpu()

        # calculate OCB reward for each agent
        if self.ocb_reward_scale != 0:
            if getattr(self.cfg.rewards, "expanded_ocb_reward", False):
                original_target_direction = (target_pos[:, :2] - box_pos[:, :2]) / (
                    torch.norm((target_pos[:, :2] - box_pos[:, :2] + 0.01), dim=1, keepdim=True))
                delta_yaw = target_rpy[:, 2] - box_rpy[:, 2]
                # delta_yaw -->(-pi, pi)
                delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                # rotate target direction by delta_yaw/2
                target_direction = torch.stack([original_target_direction[:, 0] * torch.cos(
                    -delta_yaw / 2) - original_target_direction[:, 1] * torch.sin(-delta_yaw / 2),
                                                original_target_direction[:, 0] * torch.sin(
                                                    -delta_yaw / 2) + original_target_direction[:, 1] * torch.cos(
                                                    -delta_yaw / 2)], dim=1)
                pass
            else:
                target_direction = (target_pos[:, :2] - box_pos[:, :2]) / (
                    torch.norm((target_pos[:, :2] - box_pos[:, :2]), dim=1, keepdim=True))
            vertex_list = self.cfg.asset.vertex_list
            reward_logger = []
            for i in range(self.num_agents):
                gf_pos = base_pos[:, i, :2] - box_pos[:, :2]
                rotation_matrix = rotation_matrix_2D(- box_rpy[:, 2])
                box_relative_pos = torch.bmm(rotation_matrix, gf_pos.unsqueeze(2)).squeeze(2)
                normal_vector = self.calc_normal_vector_for_obc_reward(vertex_list, box_relative_pos)
                rotation_matrix = rotation_matrix_2D(box_rpy[:, 2])
                normal_vector = torch.bmm(rotation_matrix,
                                          normal_vector.to(rotation_matrix.device).unsqueeze(2)).squeeze(2)
                ocb_reward = torch.sum(target_direction * normal_vector, dim=1) * self.ocb_reward_scale
                reward[:, i] += ocb_reward
                reward_logger.append(torch.sum(ocb_reward).cpu())
            self.reward_buffer["ocb_reward"] += np.sum(np.array(reward_logger))

        self.last_box_state = deepcopy(box_state)

        if use_privileged_obs: return obs, privileged_obs, reward, termination, info
        else: return obs, reward, termination, info
