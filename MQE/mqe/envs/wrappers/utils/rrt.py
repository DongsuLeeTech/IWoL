import torch
from time import perf_counter

import treelib

class KinodynamicRRT:
    def __init__(self,x_lim, y_lim):
        self.x_lim = x_lim
        self.y_lim = y_lim
        pass
    
    def nearest_neighbor(self, states, x_rand):
        if not torch.is_tensor(states):
            states = torch.stack(states)
        d = torch.norm(states - x_rand, dim=1)
        x_near_idx = d.argmin()
        return states[x_near_idx], x_near_idx.item()
    
    def plan(self, start, goal, obstacle_states, action_scale=0.3, goal_sample_prob=0.2, REACH_THERESHOLD=0.1, COLLISION_THERESHOLD=0.1, 
              visualizer = None, timeout: float = 1.0):
        states = [start]
        tree = treelib.Tree()
        tree.create_node(identifier=0)

        dt = 0
        t0 = perf_counter()
        while dt < timeout:
            # sample a state to extend towards, sometimes the goal if given
            if goal is not None and torch.rand(1) < goal_sample_prob:
                x_target = goal
            else:
                x_target = torch.tensor([ torch.rand(1)*(self.x_lim[1] - self.x_lim[0]) + self.x_lim[0], 
                                          torch.rand(1)*(self.y_lim[1] - self.y_lim[0]) + self.y_lim[0],], device="cuda") 

            # nearest state in the tree
            x_near, x_near_idx = self.nearest_neighbor(states, x_target)

            best_action = (x_target - x_near)/(torch.norm(x_target - x_near)+0.01) * action_scale
            x_next = x_near + best_action

            if torch.norm(obstacle_states - x_next, dim=1).min() > COLLISION_THERESHOLD + 0.8:
            # if self.satisfies_constraints is None or self.satisfies_constraints(x_near, x_next):
                if visualizer is not None:
                    visualizer.draw_connect(x_near, x_next)

                # add to the tree
                new_idx = len(states)
                tree.create_node(identifier=new_idx, parent=x_near_idx)
                states.append(x_next)

                if torch.norm(x_next - goal) < REACH_THERESHOLD:
                    dt = perf_counter() - t0
                    trajectory = self.walk_up_tree(states, tree, x_near_idx)
                    trajectory.append(x_next)

                    if not torch.allclose(trajectory[-1], goal):
                        trajectory.append(goal)

                    print("--------------------------------------")
                    print("             REACH GOAL!              ")
                    print("--------------------------------------")
                    return trajectory
            dt = perf_counter() - t0

        print("--------------------------------------")
        print("           NOT FIND PATH              ")
        print("--------------------------------------")
        return

    def walk_up_tree(self, all_states, tree, idx):
        traj_states = []
        while True:
            x = all_states[idx]
            traj_states.append(x)

            # go up the tree
            parent = tree.parent(idx)
            if parent is None:
                break
            else:
                idx = parent.identifier

        traj_states.reverse()
        return traj_states
    
    
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TwoDVisualizer:
    def __init__(self):
        plt.ion()
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.ax.set_aspect('equal')

    def draw_state(self, x, color='k', s=4, alpha=0.2):
        x = x.cpu().numpy()
        self.ax.scatter(x[0], x[1], color=color, s=s, alpha=alpha)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_connect(self, x_start, x_next):
        self.draw_state(x_next)
        self.ax.plot([x_start[0].cpu().numpy(), x_next[0].cpu().numpy()],
                     [x_start[1].cpu().numpy(), x_next[1].cpu().numpy()], color='gray', linewidth=1, alpha=0.2)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_obstacle(self, obs_state, s=4, alpha=0.2):
        obs_state = obs_state.cpu().numpy()
        for i in range(obs_state.shape[0]):
            square = patches.Rectangle((obs_state[i,0]-0.5, obs_state[i,1]-0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
            self.ax.add_patch(square)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def set_bounds(self, x_lim, y_lim):
        x_min, x_max = x_lim
        y_min, y_max = y_lim
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def clear(self):
        self.ax.cla()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()