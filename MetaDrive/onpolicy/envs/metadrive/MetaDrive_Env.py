import random
from gym import spaces
import numpy as np
from collections import OrderedDict
from typing import Union, Dict, AnyStr, Optional, Tuple, Callable

from metadrive.utils.math import clip

from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)
from onpolicy.envs.metadrive.marl_intersection_small import MultiAgentIntersectionSmallEnv
from onpolicy.envs.metadrive.marl_intersection_toy import MultiAgentIntersectionToyEnv
from onpolicy.envs.metadrive.marl_intersection_toy_steer import MultiAgentIntersectionToySteerEnv
from onpolicy.envs.metadrive.marl_intersection_toy1 import MultiAgentIntersectionToy1Env
from onpolicy.envs.metadrive.marl_parking_lot import MyMultiAgentParkingLotEnv

envs = dict(
    roundabout=MultiAgentRoundaboutEnv,
    intersection=MultiAgentIntersectionEnv,
    intersection_small=MultiAgentIntersectionSmallEnv,
    intersection_toy=MultiAgentIntersectionToyEnv,
    intersection_toy_steer=MultiAgentIntersectionToySteerEnv,
    intersection_toy1=MultiAgentIntersectionToy1Env,
    tollgate=MultiAgentTollgateEnv,
    bottleneck=MultiAgentBottleneckEnv,
    parkinglot=MultiAgentParkingLotEnv,
    myparkinglot=MyMultiAgentParkingLotEnv,
    pgma=MultiAgentMetaDrive
)
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

class MetaDriveEnv(object):
    '''Wrapper to make MetaDrive environment compatible'''

    def __init__(self, args):
        self.scenario_name = args.scenario_name
        self.num_agents = args.num_agents
        self.num_all_agents = self.num_agents
        self.global_pos = args.meta_global_pos
        self.global_info_dim = 2
        self.navi_pos = args.meta_navi_pos
        self.navi_pos_dim = 4
        self.lidar_pt_cloud = args.meta_lidar_pt_cloud
        self.lidar_num_lasers = args.meta_lidar_num_lasers
        self.comm_graph = np.zeros((self.num_agents, self.num_agents))
        self.comm_range = args.meta_comm_range
        self.comm_max_num = args.meta_comm_max_num
        self.reward_coefficient = args.meta_reward_coeff
        self.disable_steering = args.meta_disable_steering
        self.env = envs[self.scenario_name](
            {
                "use_render": args.use_render_metadrive,
                "num_agents": args.num_agents,
                "allow_respawn": args.meta_allow_respawn,
                # "offscreen_render": args.offscreen_render,
                # "headless_machine_render": True,
                "vehicle_config": dict(lidar = dict(num_lasers=args.meta_lidar_num_lasers, distance=args.meta_lidar_dist, num_others=args.meta_lidar_num_others))
            }
        )
        self.env.default_config()
        obs = self.env.reset()
        # self.vehicles = self.env.vehicles
        self.vehicle_keys = list(self.env.vehicles.keys())
        self.active_agents = list(self.env.agent_manager.active_agents.keys())
        self.dead_agents = []
        self.dead_ids = []

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        for key in self.vehicle_keys:
            if not self.disable_steering:
                self.action_space.append(self.env.action_space[key])
            else:
                self.action_space.append(spaces.Box(-1.0, 1.0, shape=(1, ), dtype=np.float32))
            k = list(self.env.observation_space)[0]
            if self.lidar_pt_cloud:
                self.observation_dim = self.env.observation_space[k].shape[0]
            else:
                self.observation_dim = self.env.observation_space[k].shape[0] - self.lidar_num_lasers
            if self.global_pos:
                self.observation_dim += self.global_info_dim
            if self.navi_pos:
                self.observation_dim += self.navi_pos_dim
                
            self.observation_space.append(spaces.Box(-0.0, 1.0, shape=(self.observation_dim, ), dtype=np.float32))
            self.share_observation_space.append(spaces.Box(-0.0, 1.0, shape=(self.observation_dim * self.num_agents, ), dtype=np.float32))

        self.action_dim = self.action_space[0].shape
        self.num_arrive_dest = 0
        self.num_crash = 0
        self.num_out_of_road = 0
        self.num_dangers = 0
        self.coop_reward = args.meta_coop_reward
        self.x_min, self.x_max, self.y_min, self.y_max = self.env.current_map.road_network.get_bounding_box()
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        print('x range', self.x_range)
        print('y range', self.y_range)
            
    def reset(self):
        obs = self.env.reset()
        self.vehicle_keys = list(self.env.vehicles.keys())
        self.active_agents = list(self.env.agent_manager.active_agents.keys())
        self.dead_agents = []
        self.dead_ids = []
        self.num_arrive_dest = 0
        self.num_crash = 0
        self.num_out_of_road = 0
        self.num_dangers = 0
        self.num_all_agents = self.num_agents
        obs = self._obs_wrapper(obs)
        self._update_comm_graph()
        return obs    
    
    def step(self, action):
        # action will be inserted in the buffer at (t)
        action = self._process_action(action)
        ### step from t to t+1
        ### env.step() may result in some newly dead agents, while the returned 
        ### obs, reward, done, info include the keys for just terminated agents
        ### afterwards, the keys of the terminated agents will be deleted
        ### please note that if there are newly respawn agents, they will appear in the returns immediately
        # obs is used for t+1 -> will be inserted in the buffer at (t+1)
        # reward is used for t -> will be inserted in the buffer at (t)
        # done is used for agent at t -> will be used to calculate the rnn states for (t+1)
        # info should be consistent with what just happened in the step() function (step reward verified)
        # comm_graph in info will be used for the next time step (t+1)
        obs, reward, done, info = self.env.step(action)
        for k, d in done.items():
            if k != '__all__':
                if d and info[k]['arrive_dest']:
                    self.num_arrive_dest += 1  
                if d and info[k]['crash']:
                    self.num_crash += 1
                if d and info[k]['out_of_road']:
                    self.num_out_of_road += 1   
                if d and (info[k]['crash'] or info[k]['out_of_road']):
                    self.num_dangers += 1
        reward = self._reward_wrapper(reward)
        done = self._done_wrapper(done)
        info = self._info_wrapper(info)         
        self._update_agents()
        self._update_comm_graph()
        info = self._update_info(info)
        obs = self._obs_wrapper(obs)
        return obs, reward, done, info
    
    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)
            
    def _process_action(self, ori_action):
        if isinstance(ori_action, (dict, OrderedDict)):
            for dead_agent in self.dead_agents:
                ori_action.pop(dead_agent)
            action = ori_action
        elif isinstance(ori_action, np.ndarray):
            if not self.disable_steering:
                action = {self.vehicle_keys[i]: ori_action[i] for i in range(len(self.vehicle_keys)) if i not in self.dead_ids}
            else:
                action = {self.vehicle_keys[i]: np.array([0., ori_action[i][0]]) for i in range(len(self.vehicle_keys)) if i not in self.dead_ids}
        else:
            raise NotImplementedError
        return action
            
    def _obs_wrapper(self, ori_obs):
        obs = []
        for key in self.vehicle_keys:
            if key not in self.dead_agents:
                if self.lidar_pt_cloud:
                    v_obs = ori_obs[key]
                else:
                    v_obs = ori_obs[key][:-self.lidar_num_lasers]

                if self.global_pos:
                    v_pos = self.env.vehicles[key].position
                    pos_x = clip((v_pos[0] - self.x_min) / self.x_range, 0.0, 1.0)
                    pos_y = clip((v_pos[1] - self.y_min) / self.y_range, 0.0, 1.0)
                    v_obs = np.concatenate([v_obs, [pos_x], [pos_y]])
                    
                if self.navi_pos:
                    navi = self.env.vehicles[key].navigation
                    current_ref_lane = navi.current_ref_lanes[0]
                    next_ref_lane = navi.next_ref_lanes[0] if navi.next_ref_lanes is not None else navi.current_ref_lanes[0]
                    later_middle = (float(navi.get_current_lane_num()) / 2 - 0.5) * navi.get_current_lane_width()
                    current_ckpt = current_ref_lane.position(current_ref_lane.length, later_middle)
                    next_ckpt = next_ref_lane.position(next_ref_lane.length, later_middle)
                    navi_pos = (current_ckpt[0], current_ckpt[1], next_ckpt[0], next_ckpt[1])

                    current_ckpt_x = clip((navi_pos[0] - self.x_min) / self.x_range, 0.0, 1.0)
                    current_ckpt_y = clip((navi_pos[1] - self.y_min) / self.y_range, 0.0, 1.0)
                    next_ckpt_x = clip((navi_pos[2] - self.x_min) / self.x_range, 0.0, 1.0)
                    next_ckpt_y = clip((navi_pos[3] - self.y_min) / self.y_range, 0.0, 1.0)
                    v_obs = np.concatenate([v_obs, [current_ckpt_x], [current_ckpt_y], [next_ckpt_x], [next_ckpt_y]])
                
                obs.append(v_obs)
            else:
                # pad zeros for the obs of dead agents
                obs.append(np.zeros(self.observation_dim))
        obs = np.stack(obs)
        return obs
    
    def _reward_wrapper(self, ori_reward):
        reward = []
        for key in self.vehicle_keys:
            if key not in self.dead_agents:
                reward.append([ori_reward[key]])
            else:
                reward.append([0.])
        if self.coop_reward:
            active_ids = list(set(np.arange(self.num_agents).tolist()) - set(self.dead_ids))
            mean_reward = np.mean(np.array(reward)[active_ids])
            coop_rewards = np.array([mean_reward] * self.num_agents)
            coop_rewards[self.dead_ids] = 0
            return coop_rewards.reshape(-1, 1)
        else:
            reward_graph = self.comm_graph - np.eye(self.num_agents)
            rewards = np.array(reward).repeat(self.num_agents, axis=1).transpose()
            other_sum = np.sum(reward_graph, axis=-1)
            rewards_other = np.sum(rewards * reward_graph, axis=-1) / (other_sum + (other_sum == 0) * 0.1)
            rewards = self.reward_coefficient * np.array(reward) + (1 - self.reward_coefficient) * rewards_other.reshape(-1, 1)
            return rewards
    
    def _done_wrapper(self, ori_done):
        done = []
        for key in self.vehicle_keys:
            if key not in self.dead_agents:
                done.append(ori_done[key])
            else:
                done.append(True)
        return np.array(done)
    
    def _info_wrapper(self, ori_info):
        info = ori_info
        info['success_rate'] = self.num_arrive_dest / self.num_all_agents
        info['crash_rate'] = self.num_crash / self.num_all_agents
        info['out_of_road_rate'] = self.num_out_of_road / self.num_all_agents
        info['danger_rate'] = self.num_dangers / self.num_all_agents
        # store the communication graph used for time step t (within the current env.step())
        info['prev_comm_graph'] = self.comm_graph
        info['num_all_agents'] = self.num_all_agents
        return info
    
    def _update_info(self, info):
        info['comm_graph'] = self.comm_graph
        return info
    
    def _update_agents(self):
        # make everything compatible with respawn agents
        current_active_agents = list(self.env.agent_manager.active_agents.keys())
        new_active_agents = list(set(current_active_agents) - set(self.active_agents))
        new_dead_agents = list(set(self.active_agents) - set(current_active_agents))
        self.active_agents = current_active_agents
        self.dead_agents += new_dead_agents
        if len(new_dead_agents) > 0:
            self.dead_ids = [self.vehicle_keys.index(agent) for agent in self.dead_agents if agent in self.vehicle_keys]
        if len(new_active_agents) > 0:
            for idx, new_active_agent in enumerate(new_active_agents):
                self.vehicle_keys[self.dead_ids[idx]] = new_active_agent 
            for _ in range(len(new_active_agents)):
                self.dead_ids.pop(0)
            self.num_all_agents += len(new_active_agents)  
    
    def _update_comm_graph(self):
        # update the graph determined be the distances between agents (masking agents far away)

        positions = []
        for key in self.vehicle_keys:
            if key not in self.dead_agents:
                pos = self.env.vehicles[key].position
                positions.append([pos[0], pos[1]])
            else:
                positions.append([0., 0.])
        # positions: [num_agents, 2]
        positions = np.array(positions)
        dist_matrix_t = np.tile(positions, (self.num_agents, 1, 1))
        dist_matrix = np.transpose(dist_matrix_t, (1, 0, 2))
        # dist_matrix: [num_agents, num_agents]
        dist_matrix = np.linalg.norm(dist_matrix - dist_matrix_t, axis=-1)
        range_mask = dist_matrix < self.comm_range
        inactive_mask = np.ones((self.num_agents, self.num_agents))
        inactive_mask[self.dead_ids, :] = 0
        inactive_mask[:, self.dead_ids] = 0

        if self.comm_max_num < self.num_agents:
            dist_matrix = np.ma.array(dist_matrix, 
                                      mask=np.ones((self.num_agents, self.num_agents))-range_mask * inactive_mask).filled(9999)
            index = np.argsort(dist_matrix, axis=-1)[:, :self.comm_max_num]
            self.comm_graph = np.zeros((self.num_agents, self.num_agents))
            for i in range(self.num_agents):
                self.comm_graph[i, index[i]] = 1
            self.comm_graph *= range_mask * inactive_mask
        else:
            self.comm_graph = range_mask * inactive_mask

    def render(self, text: Optional[Union[dict, str]] = None, mode=None, *args, **kwargs):
        # if mode in ["top_down", "topdown", "bev", "birdview"]:
        ret = self._render_topdown(text=text, *args, **kwargs)
        return ret

    def _render_topdown(self, text, *args, **kwargs):
        return self.engine.render_topdown(text, *args, **kwargs)

    @property
    def engine(self):
        from metadrive.engine.engine_utils import get_engine
        return get_engine()

    def close(self):
        self.env.close()
