#!/usr/bin/env bash
set -euxo pipefail

ulimit -n 65535

#################### 参数区（只需在这里修改参数） ####################
EXP_LOG_NAME="ReMemR1_7B"

N_NODE=4 # set to 1 for single node training
N_GPU=8

# Training Setting
MAXLEN=8192
MAX_NEW_TOKEN=2048
LR=1.0e-6
ROLLOUT_N=16
ROLLOUT_VAL_N=4
ALPHA=0.8

TRAIN_BS=128
PPO_MINI_BS=8
METRIC_NAME=em


PROJ_ROOT=`pwd`
STORAGE_ROOT="$PROJ_ROOT"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
DATASET_ROOT="${STORAGE_ROOT}/data/train"

export EXP=memory_agent/$EXP_LOG_NAME
export PROJ_DIR="$STORAGE_ROOT/results/${EXP}"
export LOG_PATH="$PROJ_ROOT/log/$EXP_LOG_NAME.log"
export ROLLOUT_DIR="${PROJ_DIR}/log/rollout_trajectory/${EXP_LOG_NAME}/"
export VAL_PATH="${DATASET_ROOT}/hotpotqa_dev.parquet"
export TRAIN_PATH="${DATASET_ROOT}/hotpotqa_train_32k.parquet"

##########################################

PYTHONPATH=$PROJ_ROOT
export WANDB_API_KEY="YOURE_WANDB_TOKEN"
export WANDB_PROJECT="your_wandb_project"

mkdir -p $PROJ_DIR
mkdir -p $ROLLOUT_DIR

export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1

export TIMESTAMP=$(date '+%Y%m%d')


python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=5000 \
    recurrent.memory.path="recurrent/impls/memory_revisit.py" \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    algorithm.alpha=$ALPHA \
    algorithm.action_reweight=false \
    trainer.save_freq=20 \
    trainer.save_best_val=true \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.val_kwargs.n=$ROLLOUT_VAL_N \
    trainer.logger=['console','wandb'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=$TRAIN_BS \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='thread' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BS \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAXLEN + 8192)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=0.999 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAXLEN + 8192)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXP_LOG_NAME \
    trainer.val_before_train=true \
    trainer.n_gpus_per_node=$N_GPU \
    trainer.nnodes=$N_NODE \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR \
    trainer.total_epochs=30 \
    2>&1 | tee $LOG_PATH