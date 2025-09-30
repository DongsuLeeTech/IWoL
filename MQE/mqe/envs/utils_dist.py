import torch
from typing import List
from isaacgym.torch_utils import get_euler_xyz

class dist_calculator:
    def __init__(self,vertex_list:list, general_dist:bool=False, yaw_active:bool=False, lambda_yaw:float=1.0):
        self.vertex_list = vertex_list
        self.general_dist = general_dist
        self.yaw_active = yaw_active
        self.lambda_yaw = lambda_yaw

    def rotate_point(self, x:torch.Tensor, y:torch.Tensor, yaw:torch.Tensor):
        """
        Rotate the point (x, y) by yaw
        """
        x_new = x * torch.cos(yaw) - y * torch.sin(yaw)
        y_new = x * torch.sin(yaw) + y * torch.cos(yaw)
        vertex_new = torch.stack([x_new, y_new], dim=1)
        return vertex_new
    
    def cal_dist(self, current_box_state:torch.Tensor, target_box_state:torch.Tensor):
        if self.general_dist:
            if self.yaw_active:
                return self.cal_general_dist_with_yaw(current_box_state, target_box_state)
            else:
                return self.cal_general_dist(current_box_state, target_box_state)
        else:
            return torch.norm((current_box_state[:, 0:2] - target_box_state[:, 0:2]).float(), dim=1)

    def cal_general_dist_with_yaw(self, current_box_state:torch.Tensor, target_box_state:torch.Tensor):
        current_x = current_box_state[:,0]
        current_y = current_box_state[:,1]
        current_yaw = torch.stack(get_euler_xyz(current_box_state[:,3:7]),dim=1)[:,2]
        target_x = target_box_state[:,0]
        target_y = target_box_state[:,1]
        target_yaw = torch.stack(get_euler_xyz(target_box_state[:,3:7]),dim=1)[:,2]

        yaw_diff = current_yaw - target_yaw
        yaw_diff = self.lambda_yaw * ((yaw_diff + torch.pi) % (2 * torch.pi) - torch.pi)

        dist = torch.stack([current_x - target_x, current_y - target_y, yaw_diff], dim=1)
        dist = torch.norm(dist, dim=1)
        return dist

    def cal_general_dist(self, current_box_state:torch.Tensor, target_box_state:torch.Tensor):
        """
        Calculate the general distance between current box and target box of all the vertexes
        """
        # box_vertex_list: list of several vertexes of the box in body coordinate (num_vertexes, 2)
        # box_center_point_pos: the center point of the box (num_env, 3)
        # box_yaw: the yaw of the box (num_env, 1)
        # target_center_point_pos: the center point of the target box (num_env, 3)
        # target_yaw: the yaw of the target box (num_env, 1)

        current_x = current_box_state[:,0]
        current_y = current_box_state[:,1]
        current_yaw = torch.stack(get_euler_xyz(current_box_state[:,3:7]),dim=1)[:,2]
        target_x = target_box_state[:,0]
        target_y = target_box_state[:,1]
        target_yaw = torch.stack(get_euler_xyz(target_box_state[:,3:7]),dim=1)[:,2]
        
        box_vertex_list = self.vertex_list
        box_vertex_list = torch.tensor(box_vertex_list).to(current_x.device)

        # calculate current box vertexes in the world coordinate
        current_vertex_list = []
        for vertex in box_vertex_list:
            vertex = self.rotate_point(vertex[0], vertex[1], current_yaw)
            vertex[:,0] += current_x
            vertex[:,1] += current_y
            current_vertex_list.append(vertex)
        current_vertex_list = torch.stack(current_vertex_list, dim=1)

        # calculate the target box vertexes in the world coordinate
        target_vertex_list = []
        for vertex in box_vertex_list:
            vertex = self.rotate_point(vertex[0], vertex[1], target_yaw)
            vertex[:,0] += target_x
            vertex[:,1] += target_y
            target_vertex_list.append(vertex)
        target_vertex_list = torch.stack(target_vertex_list, dim=1) 

        # calculate the distance between the current box vertexes and the target box vertexes
        return torch.sum(torch.norm(current_vertex_list - target_vertex_list, dim=2), dim=1)