import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.utils.video_util import *
from collections import defaultdict
import wandb
import imageio
import os
import pygame
import pickle
import uuid
import psutil
os.environ["SDL_VIDEODRIVER"] = "dummy"

def _t2n(x):
    return x.detach().cpu().numpy()

class MetaDriveRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MetaDrive. See parent class for details."""
    def __init__(self, config):
        super(MetaDriveRunner, self).__init__(config)
        self.env_infos = defaultdict(list)
        self.my_uuid = uuid.uuid4()

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # get reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                num_all_agents = np.array([infos[i]['num_all_agents'] for i in range(self.n_rollout_threads)])
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                train_infos["average_batch_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["normalized_average_batch_episode_rewards"] = np.mean(np.sum(self.buffer.rewards, axis=-2).squeeze() / num_all_agents) * self.episode_length
                print("average batch episode rewards is {}".format(train_infos["average_batch_episode_rewards"]))
                print("normalized average batch episode rewards is {}".format(train_infos["normalized_average_batch_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.env_infos = defaultdict(list)

            # eval
            if episode % self.eval_interval == 0 or episode == episodes - 1:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic, _, _, _, _\
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            comm_graphs=self.buffer.comm_graphs[step])

        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        elif self.envs.action_space[0].__class__.__name__ == 'Box':
            actions_env = actions
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        
        # masks are used for masking rnn states and computing returns
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        dones_env = np.all(dones, axis=1)
        # active masks are only used for computing policy and critic loss
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # no masking between different episodes, this is the same as mpe, smac, and grf
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        comm_graphs = [infos[i]["comm_graph"] for i in range(self.n_rollout_threads)]
        comm_graphs = np.stack(comm_graphs)
        
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, comm_graphs=comm_graphs)

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]),
                                                comm_graphs=self.buffer.comm_graphs[-1])

        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []

        all_frames = []
        comm_list = []
        att_list = []
        att_masks = []
        graphs = []

        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_comm_graphs = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.num_agents), dtype=np.float32)
        eval_success_rates = []
        eval_crash_rates = []
        eval_danger_rates = []
        eval_out_of_road_rates = []
        eval_episode_steps = []
        eval_num_all_agents = []

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            image = self.eval_envs.render(mode="rgb_array")
            image = pygame.surfarray.array3d(image[0]).astype(np.uint8)
            image = np.transpose(image, (1, 0, 2))
            all_frames.append(image)

            eval_action, eval_rnn_states, comm_info, att, att_mask, graph = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                comm_graphs=eval_comm_graphs,
                                                deterministic=True)

            comm_list.append(comm_info.detach().cpu().numpy())
            att_list.append(att.detach().cpu().numpy())
            att_masks.append(att_mask.detach().cpu().numpy())
            graphs.append(graph.detach().cpu().numpy())
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            elif self.envs.action_space[0].__class__.__name__ == 'Box':
                eval_actions_env = eval_actions
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_dones_env = np.all(eval_dones, axis=-1)

            if not self.all_args.meta_allow_respawn:
                for i in range(eval_dones_env.shape[0]):
                    if eval_dones_env[i]:
                        eval_success_rates.append(eval_infos[i]['success_rate'])
                        eval_crash_rates.append(eval_infos[i]['crash_rate'])
                        eval_out_of_road_rates.append(eval_infos[i]['out_of_road_rate'])
                        eval_danger_rates.append(eval_infos[i]['danger_rate'])
                        eval_num_all_agents.append(eval_infos[i]['num_all_agents'])
                        for k, v in eval_infos[i].items():
                            if 'agent' in k:
                                eval_episode_steps.append(eval_infos[i][k]['episode_length'])
                                break
            else:
                if eval_step == self.episode_length - 1:
                    for i in range(eval_dones_env.shape[0]):
                        eval_success_rates.append(eval_infos[i]['success_rate'])
                        eval_crash_rates.append(eval_infos[i]['crash_rate'])
                        eval_out_of_road_rates.append(eval_infos[i]['out_of_road_rate'])
                        eval_danger_rates.append(eval_infos[i]['danger_rate'])
                        eval_num_all_agents.append(eval_infos[i]['num_all_agents'])
                        eval_episode_steps.append(self.episode_length)

            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            eval_comm_graphs = [eval_infos[i]["comm_graph"] for i in range(self.n_eval_rollout_threads)]
            eval_comm_graphs = np.stack(eval_comm_graphs)

        comm_list = np.array(comm_list)
        att_list = np.array(att_list)
        att_masks = np.array(att_masks)
        graphs = np.array(graphs)

        att_viz = save_gif_with_attention(att_list, slicin_idx=100, filename=f'./tmp_data/attention_{self.my_uuid}.gif')
        mask_viz = save_gif_with_attention(att_masks, slicin_idx=100, filename=f'./tmp_data/mask_{self.my_uuid}.gif')
        graph_viz = save_gif_with_graph(graphs, slicin_idx=100, filename=f'./tmp_data/graph_{self.my_uuid}.gif')
        comm_viz = plot_and_log_communication_block(comm_list, filename=f'./tmp_data/message_{self.my_uuid}.png')
        video = record_video('Video', renders=all_frames)

        num_all_agents = np.array([eval_infos[i]['num_all_agents'] for i in range(self.n_eval_rollout_threads)])
        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_success_rates = np.array(eval_success_rates)
        eval_crash_rates = np.array(eval_crash_rates)
        eval_out_of_road_rates = np.array(eval_out_of_road_rates)
        eval_danger_rates = np.array(eval_danger_rates)
        eval_episode_steps = np.array(eval_episode_steps)
        eval_num_all_agents = np.array(eval_num_all_agents)
        
        eval_safety_rates = 1 - eval_danger_rates
        eval_efficiency = (eval_success_rates - eval_danger_rates) * eval_num_all_agents / eval_episode_steps

        eval_env_infos = {}
        eval_env_infos['rew/eval_average_batch_episode_rewards'] = [np.mean(np.sum(np.array(eval_episode_rewards), axis=0))]
        print("eval average batch episode rewards of agent: " + str(eval_env_infos['rew/eval_average_batch_episode_rewards'][0]))
        eval_env_infos['rew/eval_normalized_average_batch_episode_rewards'] = [np.mean(np.sum(np.sum(np.array(eval_episode_rewards), axis=0).squeeze(), axis=-1) / num_all_agents)]
        print("eval normalized average batch episode rewards of agent: " + str(eval_env_infos['rew/eval_normalized_average_batch_episode_rewards'][0]))
        eval_env_infos['avg/eval_success_rate_mean'] = [np.mean(eval_success_rates)]
        eval_env_infos['eval_success_rate_max'] = [np.max(eval_success_rates)]
        eval_env_infos['eval_success_rate_min'] = [np.min(eval_success_rates)]
        print("eval average success rate: " + str(eval_env_infos['avg/eval_success_rate_mean'][0]))
        eval_env_infos['avg/eval_crash_rate_mean'] = [np.mean(eval_crash_rates)]
        eval_env_infos['eval_crash_rate_max'] = [np.max(eval_crash_rates)]
        eval_env_infos['eval_crash_rate_min'] = [np.min(eval_crash_rates)]
        print("eval average crash rate: " + str(eval_env_infos['avg/eval_crash_rate_mean'][0]))
        eval_env_infos['avg/eval_out_of_road_rate_mean'] = [np.mean(eval_out_of_road_rates)]
        eval_env_infos['eval_out_of_road_rate_max'] = [np.max(eval_out_of_road_rates)]
        eval_env_infos['eval_out_of_road_rate_min'] = [np.min(eval_out_of_road_rates)]
        print("eval average out-of-road rate: " + str(eval_env_infos['avg/eval_out_of_road_rate_mean'][0]))
        eval_env_infos['avg/eval_safety_rate_mean'] = [np.mean(eval_safety_rates)]
        eval_env_infos['eval_safety_rate_max'] = [np.max(eval_safety_rates)]
        eval_env_infos['eval_safety_rate_min'] = [np.min(eval_safety_rates)]
        print("eval average safety rate: " + str(eval_env_infos['avg/eval_safety_rate_mean'][0]))
        eval_env_infos['avg/eval_efficiency_mean'] = [np.mean(eval_efficiency)]
        eval_env_infos['eval_efficiency_max'] = [np.max(eval_efficiency)]
        eval_env_infos['eval_efficiency_min'] = [np.min(eval_efficiency)]
        print("eval average efficiency: " + str(eval_env_infos['avg/eval_efficiency_mean'][0]))
        eval_env_infos['avg/eval_episode_step_mean'] = [np.mean(eval_episode_steps)]
        eval_env_infos['eval_episode_step_max'] = [np.max(eval_episode_steps)]
        eval_env_infos['eval_episode_step_min'] = [np.min(eval_episode_steps)]
        print("eval average episode step: " + str(eval_env_infos['avg/eval_episode_step_mean'][0]))
        eval_env_infos['avg/eval_num_all_agents_mean'] = [np.mean(eval_num_all_agents)]
        eval_env_infos['eval_num_all_agents_max'] = [np.max(eval_num_all_agents)]
        eval_env_infos['eval_num_all_agents_min'] = [np.min(eval_num_all_agents)]
        print("eval average number of appeared agents : " + str(eval_env_infos['avg/eval_num_all_agents_mean'][0]))

        eval_env_infos['eval/rendering'] = video
        eval_env_infos['eval/comm_message'] = comm_viz
        eval_env_infos['eval/attention'] = att_viz
        eval_env_infos['eval/att_mask'] = mask_viz
        eval_env_infos['eval/graph'] = graph_viz

        self.log_env(eval_env_infos, total_num_steps)

        mem_percent = 0
        for proc in psutil.process_iter():
            if proc.name() == self.all_args.process_name:
                mem_percent += proc.memory_percent()
        print('cpu memory percent of the current process', mem_percent)
        eval_mem_info = {'cpu memory usage': [mem_percent]}
        self.log_env(eval_mem_info, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        all_rewards = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.envs[0].env.render(mode="top_down", track_target_vehicle=False)
                image = pygame.surfarray.array3d(image).astype(np.uint8)
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            comm_graphs = np.zeros((self.n_rollout_threads, self.num_agents, self.num_agents), dtype=np.float32)
            
            episode_rewards = []
            success_rates = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                print(obs)
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    comm_graphs = comm_graphs,
                                                    deterministic=True)
           
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                actions_env = actions

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
                
                comm_graphs = [infos[i]["comm_graph"] for i in range(self.n_rollout_threads)]
                comm_graphs = np.stack(comm_graphs)

                if self.all_args.save_gifs:
                    image = self.envs.envs[0].env.render(mode="top_down", track_target_vehicle=False)
                    image = pygame.surfarray.array3d(image).astype(np.uint8)
                    all_frames.append(image)
                else:
                    envs.render('human')
                
                if not self.all_args.meta_allow_respawn:
                    if np.all(dones[0]):
                        success_rates.append(infos[0]['success_rate'])
                        break
                    elif step == self.episode_length - 1:
                        success_rates.append(infos[0]['success_rate'])
                else:
                    if step == self.episode_length - 1:
                        success_rates.append(infos[0]['success_rate'])
            
            all_rewards.append(np.array(episode_rewards))

            print("rendering average (real) episode reward is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
            print("rendering average success is: " + str(np.mean(np.array(success_rates))))
            
        if self.all_args.save_gifs:
            self.gif_dir = './'
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

        return all_frames
