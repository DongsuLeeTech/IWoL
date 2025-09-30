#!/bin/sh
env="robotarium"
scenario="Simple"
algo="ImIWoL" #"mappo" "ippo" "rmappo_comm"
exp="robo_simple"
seed_max=5
episode_length=50

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=3 python ../train/train_robotarium_ImIWoL.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --seed ${seed} --episode_length ${episode_length} \
    --n_training_threads 1 --n_rollout_threads 32 --num_mini_batch 16  --num_env_steps 200000 \
    --ppo_epoch 5 --log_interval 1 --use_eval --eval_interval 10 --n_eval_rollout_threads 1 \
    --lr 5e-4 --critic_lr 5e-4 --hidden_size 256 \
    --comm_type 1 --num_comm_hops 4 --comm_hidden_size 256 --comm_num_heads 4 \
    --pos_embed --obs_pos_embed_start 1 --obs_pos_embed_end 2 \
    --obs_info_scheduler rnn_enc --skip_connect_final --fixed_masked --mask_threshold 0.5 --use_scheduler \
    --n_obs_head 1 --obs_enc_type rnn --use_vq_vae --code_size 256 --vq_temp 1.0 --use_centralized_V \
    --wandb_project_name robotarium --user_name user_naem
done

