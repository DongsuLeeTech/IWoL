#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

from onpolicy.config import get_config
from onpolicy.envs.robo_wrapper import Wrapper
from onpolicy.envs.env_wrappers import *


def make_train_env(all_args):
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, f"../../envs/robotarium/scenarios/{all_args.scenario_name}/config.yaml")

    def get_env_fn(rank):
        def init_env():
            all_args.seed = all_args.seed + rank * 1000
            env = Wrapper(all_args.scenario_name, config_path)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, f"../../envs/robotarium/scenarios/{all_args.scenario_name}/config.yaml")

    def get_env_fn(rank):
        def init_env():
            all_args.seed = all_args.seed * 50000 + rank * 10000
            env = Wrapper(all_args.scenario_name, config_path)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='Simple', help="Which scenario to run on")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.scenario_name == "Warehouse":
        all_args.num_agents = 6
    elif all_args.scenario_name == "Simple":
        all_args.num_agents = 4
    else:
        all_args.num_agents = 4

    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False


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

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    from datetime import datetime
    g_start_time = int(datetime.now().timestamp())
    exp_name = ''
    exp_name += f'sd{all_args.seed:03d}_n{all_args.num_agents}_'
    exp_name += f'{g_start_time}_'
    exp_name += f'{all_args.scenario_name}'

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.wandb_project_name,
                         name=exp_name,
                         dir=str(run_dir),
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
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
    from onpolicy.runner.shared.robotarium_ImIWoL_runner import RobotariumRunner as Runner
    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not None:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])