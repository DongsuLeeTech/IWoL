from gym import spaces, Env
from onpolicy.envs.robotarium.scenarios.Warehouse.warehouse import Warehouse
from onpolicy.envs.robotarium.scenarios.MaterialTransport.MaterialTransport import MaterialTransport
from onpolicy.envs.robotarium.scenarios.ArcticTransport.ArcticTransport import ArcticTransport
from onpolicy.envs.robotarium.scenarios.Simple.Simple import Simple
# Add other scenario imports here
from onpolicy.envs.robotarium.utilities.misc import objectview
import os
import yaml
import numpy as np

env_dict = {'Simple': Simple,
            'Warehouse': Warehouse,
            'MaterialTransport': MaterialTransport,
            'ArcticTransport': ArcticTransport}

class Wrapper(Env):
    def __init__(self, env_name, config_path):
        """Creates the Gym Wrappers

        Args:
            env (PredatorCapturePrey): A PredatorCapturePrey object to wrap in a gym env
        """
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        args = objectview(config)
        self.env = env_dict[env_name](args)
        self.observation_space = self.get_observation_space()
        self.share_observation_space = self.get_share_observation_space()
        self.action_space = self.get_action_space()
        self.n_agents = self.env.num_robots

    def reset(self, show_figure=False):
        obs = self.env.reset(show_figure)
        # share_obs = obs  # In simple scenario, all agents share the same observation
        # available_actions = np.ones((self.env.num_agent, 1))
        return obs #, share_obs, available_actions

    def step(self, action_n):
        # Execute the given action in the wrapped environment
        obs_n, reward_n, done_n, info_n = self.env.step(action_n)
        return tuple(obs_n), reward_n, done_n, info_n

    def get_action_space(self):
        return self.env.get_action_space()

    def get_observation_space(self):
        return self.env.get_observation_space()

    def get_share_observation_space(self):
        return self.env.get_share_observation_space()