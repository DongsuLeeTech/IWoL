import numpy as np
from onpolicy.envs.robotarium.robotarium_base import RobotariumBaseEnv

class FormationControlEnv(RobotariumBaseEnv):
    def __init__(self, num_agents=4):
        super(FormationControlEnv, self).__init__(num_agents, continuous_actions=True)
        
        # Define formation shape (square formation)
        self.formation_shape = self._generate_formation_shape()
        
    def _generate_formation_shape(self):
        """Generate target positions for square formation."""
        if self.num_agents == 4:
            return np.array([
                [0.5, 0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5, -0.5]
            ])
        else:
            # Generate circular formation for different number of agents
            angles = np.linspace(0, 2*np.pi, self.num_agents, endpoint=False)
            radius = 0.5
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            return np.vstack((x, y))
    
    def _get_obs_dim(self):
        return super()._get_obs_dim() + 2  # Add target position to observation
        
    def _get_obs(self):
        base_obs = super()._get_obs()
        poses = self.robotarium.get_poses()
        
        # Add target positions to observation
        full_obs = []
        for i in range(self.num_agents):
            target_pos = self.formation_shape[:, i]
            agent_obs = np.concatenate([
                base_obs[i],
                target_pos
            ])
            full_obs.append(agent_obs)
            
        return np.array(full_obs)
    
    def reset(self):
        self.robotarium.reset()
        obs = self._get_obs()
        return obs, obs, np.ones((self.num_agents, 1))
    
    def step(self, actions):
        # Apply actions
        safe_velocities = self._apply_actions(actions)
        self.robotarium.set_velocities(np.arange(self.num_agents), safe_velocities)
        self.robotarium.step()
        
        # Get new observations
        obs = self._get_obs()
        
        # Calculate rewards
        rewards = self._compute_rewards()
        
        # Check if done
        dones = self._check_dones()
        
        info = [{'formation_error': self._get_formation_error()} for _ in range(self.num_agents)]
        
        return obs, obs, rewards, dones, info, np.ones((self.num_agents, 1))
    
    def _compute_rewards(self):
        poses = self.robotarium.get_poses()
        rewards = []
        
        for i in range(self.num_agents):
            # Distance to target position
            current_pos = poses[:2, i]
            target_pos = self.formation_shape[:, i]
            distance = np.linalg.norm(current_pos - target_pos)
            
            # Reward is negative distance (closer is better)
            reward = -distance
            
            # Bonus for being very close to target
            if distance < 0.05:
                reward += 5.0
                
            rewards.append(reward)
            
        return np.array(rewards)
    
    def _check_dones(self):
        formation_error = self._get_formation_error()
        done = formation_error < 0.1
        return [done] * self.num_agents
    
    def _get_formation_error(self):
        poses = self.robotarium.get_poses()
        total_error = 0
        
        for i in range(self.num_agents):
            current_pos = poses[:2, i]
            target_pos = self.formation_shape[:, i]
            error = np.linalg.norm(current_pos - target_pos)
            total_error += error
            
        return total_error / self.num_agents 