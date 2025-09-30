#!/usr/bin/env python
import sys
import os
import wandb
from datetime import datetime
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.metadrive.MetaDrive_Env import MetaDriveEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


"""Train script for MetaDrive."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MetaDrive":
                env = MetaDriveEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MetaDrive":
                env = MetaDriveEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    # parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=40, help="number of players")
    # metadrive parameters
    parser.add_argument("--use_render_metadrive", action='store_true', default=False, help="if use render for metadrive")
    # parser.add_argument("--offscreen_render", action='store_true', default=False, help="off screen render for metadrive")
    parser.add_argument("--meta_coop_reward", action='store_true', default=False, help="use fully cooperative reward for metadrive")
    parser.add_argument("--meta_reward_coeff", type=float, default=1.0, help="the coefficient for individual reward vs. local reward (default: individual)")
    parser.add_argument("--meta_global_pos", action='store_true', default=False, help="if add the global position in the observation")
    parser.add_argument("--meta_lidar_num_lasers", type=int, default=72, help='lidar: number of lasers')
    parser.add_argument("--meta_lidar_dist", type=float, default=40, help='lidar distance for metadrive vehicles')
    parser.add_argument("--meta_lidar_num_others", type=int, default=0, help="the number of surrounding agents' info to acquire")
    parser.add_argument("--meta_lidar_pt_cloud", action='store_true', default=False, help="include the lidar point cloud in the observation")
    parser.add_argument("--meta_allow_respawn", action='store_true', default=False, help="allow respawn in metadrive")
    parser.add_argument("--meta_navi_pos", action='store_true', default=False, help="if add the positions of navigation checkpoints to observation")
    # metadrive parameters (communication related)
    parser.add_argument('--meta_comm_range', type=float, default=20, help="the range (radius of a circle) of the communication")
    parser.add_argument('--meta_comm_max_num', type=int, default=4, help="the maximum number of agents that one agent can communicate with (including itself)")
    parser.add_argument('--meta_disable_steering', action='store_true', default=False, help='if disable the vehicle steering in the signal environment')
    
    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    print("u are choosing to use VoCAL, we set use_recurrent_policy to be True")
    all_args.algorithm_name == "IICWL"
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    g_start_time = int(datetime.now().timestamp())
    exp_name = ''
    exp_name += f'sd{all_args.seed:03d}_n{all_args.num_agents}_'
    exp_name += f'{g_start_time}_'
    exp_name += f'{all_args.scenario_name}'

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.wandb_project_name,
                         name=exp_name,
                         group=all_args.wandb_group,
                         dir=str(run_dir),
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    all_args.process_name = str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.scenario_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name)
    setproctitle.setproctitle(all_args.process_name)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    from onpolicy.runner.shared.ImIWoL_runner import MetaDriveRunner
    runner = MetaDriveRunner(config)
    try:
        runner.run()
    finally:
    # post process
        envs.close()
        if all_args.use_eval and eval_envs is not envs:
            eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
