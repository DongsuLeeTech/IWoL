import os
import sys
import time
import numpy as np
import yaml
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import isaacgym modules first
import isaacgym
from mqe.util import get_args, make_env
from mqe.envs.utils import custom_cfg
from mqe.algorithms.marl.runner import Runner

# Then import PyTorch and other dependencies
import torch

def main():
    # Get arguments using MQE's argument parser
    args = get_args()

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg/mappo/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update config with command line arguments
    if args.num_envs is not None:
        config["n_rollout_threads"] = args.num_envs
    if args.max_iterations is not None:
        config["num_env_steps"] = args.max_iterations * config["episode_length"] * config["n_rollout_threads"]

    # Set device
    device = torch.device(args.rl_device)
    config["device"] = device
    config['task'] = args.task
    config['seed'] = args.seed
    print(f'task config: {config["task"]}')
    
    # Create output directory
    run_dir = Path(config["run_dir"]) / config["experiment_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create environment using utils.make_env
    env, env_cfg = make_env(args, custom_cfg(args), False)

    # Initialize runner
    runner = Runner(
        vec_env=env,
        config=config,
        model_dir=""
    )

    # Run training
    runner.run()

    env.close()

if __name__ == "__main__":
    main() 