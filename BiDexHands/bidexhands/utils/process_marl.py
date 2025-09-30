# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


def get_AgentIndex(config):
    agent_index = []
    # right hand
    agent_index.append(eval(config["env"]["handAgentIndex"]))
    # left hand
    agent_index.append(eval(config["env"]["handAgentIndex"]))

    return agent_index
    
def process_MultiAgentRL(args, env, config, model_dir=""):

    config["n_rollout_threads"] = env.num_envs
    config["n_eval_rollout_threads"] = env.num_envs

    if args.algo in ["imiwol"]:
        from bidexhands.algorithms.marl.ImIWoL_runner import ImIWoL_Runner
        marl = ImIWoL_Runner(vec_env=env,
                    config=config,
                    model_dir=model_dir
                    )
    elif args.algo in ["exiwol"]:
        from bidexhands.algorithms.marl.ExIWoL_runner import ExIWoL_Runner
        marl = ExIWoL_Runner(vec_env=env,
                    config=config,
                    model_dir=model_dir
                    )
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    return marl
