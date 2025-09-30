from scipy.interpolate import CubicSpline
import numpy as np
import torch

class TrajectoryPlanner:
    def __init__(self, num_envs, start_pos, end_target, device='cuda'):
        self.end_target = end_target
        self.start_pos = start_pos
        self.trajectory_length = 13
        self.fixed_height = 0.15
        self.num_envs = num_envs
        self.device = device
        self.trajectory = torch.zeros((self.num_envs, self.trajectory_length, 2), device=self.device)
        
        for env_id in range(num_envs):
            start = self.start_pos[env_id].cpu().numpy()
            end = self.end_target[env_id].cpu().numpy()
            self.trajectory[env_id, :, :] = self.generate_trajectory(start, end)
        
        self.next_planning_position = self.trajectory[:, 1]
        self.next_planning_position_idx = torch.ones(num_envs, dtype=torch.int64, device=self.device)
    
    def generate_trajectory(self, start, end):
        # Generate midpoints with random adjustments
        mid_point = (start[:2] + end[:2]) / 2 + np.random.uniform(-0.5, 0.5, size=2)
        quarter_point_1 = (start[:2] + mid_point) / 2 + np.random.uniform(-0.5, 0.5, size=2)
        quarter_point_2 = (mid_point + end[:2]) / 2 + np.random.uniform(-0.5, 0.5, size=2)

        # Cubic spline interpolation on x and y coordinates
        t = np.array([0, 0.25, 0.5, 0.75, 1])
        points_x = np.array([start[0], quarter_point_1[0], mid_point[0], quarter_point_2[0], end[0]])
        points_y = np.array([start[1], quarter_point_1[1], mid_point[1], quarter_point_2[1], end[1]])
        cs_x = CubicSpline(t, points_x, bc_type='natural')
        cs_y = CubicSpline(t, points_y, bc_type='natural')
        t_new = np.linspace(0, 1, num=self.trajectory_length)
        left_x = cs_x(t_new)
        left_y = cs_y(t_new)

        trajectory = np.vstack((left_x, left_y)).T

        return torch.tensor(trajectory, device=self.device)

    def reset_trajectory(self, env_ids, end_target):
        self.end_target = end_target
        for env_id in env_ids:
            start = self.start_pos[env_id].cpu().numpy()
            end = self.end_target[env_id].cpu().numpy()
            # Update the trajectory for each specified environment
            self.trajectory[env_id, :, :] = self.generate_trajectory(start, end)
            self.next_planning_position[env_ids] = self.trajectory[env_ids, 1]
            self.next_planning_position_idx[env_ids] = 1

    def get_trajectory(self, episode_length_buf=None):
        return self.trajectory.reshape(self.num_envs, self.trajectory_length * 2)
    
    def update_next_planning_position(self, box_pos, trajectory):
        new_trajectory = trajectory.view(self.num_envs, self.trajectory_length, 2)
        box_pos_2d = box_pos[:, :2]
        distances = torch.norm(self.next_planning_position - box_pos_2d, dim=1)

        condition = (distances < 1.3)
        self.next_planning_position_idx[condition] += 1
        self.next_planning_position_idx = torch.clamp(self.next_planning_position_idx, max=self.trajectory_length - 1)
        self.next_planning_position = new_trajectory[torch.arange(self.num_envs), self.next_planning_position_idx]

        return self.next_planning_position