import numpy as np
from gym import spaces
import copy
import yaml
import os

# This file should stay as is when copied to robotarium_eval but local imports must be changed to work with training!
from onpolicy.envs.robotarium.utilities.roboEnv import roboEnv
from onpolicy.envs.robotarium.utilities.misc import *
from onpolicy.envs.robotarium.scenarios.Simple.visualize import *
from onpolicy.envs.robotarium.base import BaseEnv


class Agent:
    '''
    This is a helper class for Simple
    Keeps track of information for each agent and creates functions needed by each agent.
    '''

    def __init__(self, index, action_id_to_word):
        self.index = index
        self.action_id2w = action_id_to_word

    def get_observation(self, state_space):
        '''
        Returns: [agent_x_pos, agent_y_pos]
        array of dimension [1, OBS_DIM]
        '''
        agent_pose = np.array(state_space['poses'][:, self.index][:2])
        return agent_pose

    def generate_goal(self, goal_pose, action, args):
        '''
        Generates the final position for each time-step for the individual
        agent.
        '''

        if isinstance(action, np.ndarray):
            action = action[0]

        if self.action_id2w[action] == 'left':
            goal_pose[0] = max(goal_pose[0] - args.step_dist, args.LEFT)
            goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'right':
            goal_pose[0] = min(goal_pose[0] + args.step_dist, args.RIGHT)
            goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'up':
            goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
            goal_pose[1] = max(goal_pose[1] - args.step_dist, args.UP)
        elif self.action_id2w[action] == 'down':
            goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
            goal_pose[1] = min(goal_pose[1] + args.step_dist, args.DOWN)
        else:
            goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
            goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]

        return goal_pose


# An extremely simple environment for debugging the policy.
# It consists of multiple agent who already knows the goal's location
# and will get dense rewards.

class Simple(BaseEnv):
    def __init__(self, args):
        # Settings
        self.args = args

        self.agent_poses = None  # robotarium convention poses
        self.goal_loc = None

        self.num_goal = 1  # There is only one goal
        self.num_agent = args.num_agents
        self.num_robots = args.num_agents
        self.terminated = False

        if self.args.seed != -1:
            np.random.seed(self.args.seed)

        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3: 'down', 4: 'no_action'}

        self.visualizer = Visualize(self.args)  # visualizer
        self.env = roboEnv(self, args)  # robotarium Environment

        # Initialize the agents
        self.agents = []
        for agent_id in range(self.num_agent):
            self.agents.append(Agent(agent_id, self.action_id2w))

        # Declaring action and observation space
        actions = []
        observations = []

        for agent in self.agents:
            actions.append(spaces.Discrete(5))
            self.obs_dim = 4 # 2 * (self.num_agent + 1)  # Total agents + goal locations
            observations.append(spaces.Box(low=-1.5, high=3, shape=(self.obs_dim,), dtype=np.float32))

        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))
        self.shared_observation_space = spaces.Tuple(tuple([spaces.Box(low=-1.5, high=3,
                                                                       shape=(self.obs_dim * self.num_agent,),
                                                                       dtype=np.float32)]))

    def _generate_step_goal_positions(self, actions):

        '''
        Applies the actions on each agent.
        '''

        goal = copy.deepcopy(self.agent_poses)

        for i, agent in enumerate(self.agents):
            goal[:, i] = agent.generate_goal(goal[:, i], actions[i], self.args)

        return goal

    def _generate_state_space(self):
        '''
        Generates a dictionary describing the state space of the robotarium
         - poses: Poses of all the robots
         - goal: Location of the goal
        '''
        state_space = {}
        state_space['poses'] = self.agent_poses
        state_space['goal'] = []
        state_space['goal'].append(np.array(self.goal_loc).reshape((2, 1)))
        return state_space

    def reset(self, show_figure=False):
        '''
        Resets the environment before running the episode
        '''
        self.episode_steps = 0

        # Specify the area is which agent will be spawne
        width = self.args.ROBOT_INIT_RIGHT_THRESH - self.args.LEFT
        height = self.args.DOWN - self.args.UP
        # Agent pose
        self.agent_poses = generate_initial_locations(self.num_robots, width, height, \
                                                      self.args.ROBOT_INIT_RIGHT_THRESH,
                                                      start_dist=self.args.start_dist)

        # Goal location generation
        width = self.args.RIGHT - self.args.PREY_INIT_LEFT_THRESH
        self.goal_loc = generate_initial_locations(1, width, height, \
                                                   self.args.ROBOT_INIT_RIGHT_THRESH, start_dist=self.args.step_dist,
                                                   spawn_left=False)
        self.goal_loc = self.goal_loc[:2].T

        # Reset episode flag
        self.terminated = False
        # Reset state space
        self.state_space = self._generate_state_space()
        self.env.reset(show_figure)
        return [[0] * (self.obs_dim)] * self.num_agent

    def step(self, actions_):
        '''
        Step into the environment
        Returns observation, reward, done, info (empty dictionary for now)
        '''
        self.episode_steps += 1
        info = {}

        # Steps into the environment and applies the action
        # to get an updated state.
        return_msg, dist, frames = self.env.step(actions_)
        updated_state = self._generate_state_space()
        obs = self.get_observations(updated_state)

        rewards = self.get_rewards(updated_state)
        if return_msg == '':
            self.terminated = self.episode_steps > self.args.max_episode_steps
        else:
            print("Ending due to", return_msg)
            # rewards = [-5] * self.num_robots
            self.terminated = True
            info['remaining'] = return_msg

        if self.args.save_gif:
            info['frames'] = frames

        info['dist_travelled'] = dist
        return obs, rewards, [self.terminated] * self.num_agent, info

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_observations(self, state_space):
        '''
        Return's the full observation for the agents.
        '''
        observations = []
        goal_loc = state_space['goal'][0].reshape(-1)
        for agent in self.agents:
            obs = np.concatenate([agent.get_observation(state_space), goal_loc])
            observations.append(obs)

        # full_observations = []
        # for i, agent in enumerate(self.agents):
        #     full_observations.append(observations[i])
        #     nbr_indices = [j for j in range(self.num_agent) if j != agent.index]
        #     for nbr_index in nbr_indices:
        #         full_observations[i] = np.concatenate((full_observations[i], observations[nbr_index]))
        #
        #     full_observations[i] = np.concatenate((full_observations[i], goal_loc))

        return observations

    def get_rewards(self, state_space):
        '''
        Returns dense rewards based on the negative of the distance between the current agent & goal
        '''
        agent_loc = state_space['poses']
        rewards = []
        info = self.env.robotarium._error_infos

        a = 0.7

        individual_rewards = []

        # Calculate individual rewards
        for agent_id, agent in enumerate(self.agents):
            if info[agent_id]:  # Penalty for error states
                individual_rewards.append(-5)
            else:
                reward = -(np.sum(np.square(agent_loc[:2, agent_id].reshape(1, 2)
                                            - self.goal_loc.reshape(1, 2))))
                reward *= self.args.reward_scaler
                individual_rewards.append(reward)

        # Calculate total rewards by combining individual and collective rewards
        for agent_id in range(len(self.agents)):
            # Calculate collective reward (mean of other agents' rewards)
            collective_reward = np.mean([individual_rewards[i]
                                         for i in range(len(self.agents)) if i != agent_id])

            # Total reward: weighted combination
            # a = self.args.reward_weight  # Weight for individual reward (0 <= a <= 1)
            total_reward = a * individual_rewards[agent_id] + (1 - a) * collective_reward
            rewards.append(total_reward)

        self.state_space = state_space
        return rewards

    def get_share_observation_space(self):
        return self.shared_observation_space

