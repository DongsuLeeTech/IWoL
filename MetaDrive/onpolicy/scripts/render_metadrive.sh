#!/bin/sh
env="MetaDrive"
scenario="intersection_small" 
num_agents=8
algo="rmappo"
exp="render"
seed_max=1
episode_length=999

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_metadrive.py --save_gifs --env_name ${env} --algorithm_name ${algo} \
    --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --episode_length ${episode_length} \
    --n_training_threads 1 --n_rollout_threads 1 --render_episodes 1 \
    --meta_lidar_num_lasers 2 --meta_lidar_dist 10 --meta_lidar_num_others 4 \
    --hidden_size 64 \
    --meta_comm_range 100 --meta_comm_max_num 4 \
    --model_dir "results/MetaDrive/intersection_toy/rmappo_comm/nter_toyr4v_nospawn_nosteer_nolidar2_5_others0_global_pos_em_t1comm_hop1_hid64_range100_4_coeff05/wandb/latest-run/files" --use_wandb
done
