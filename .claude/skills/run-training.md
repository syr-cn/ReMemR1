# Run Training - ReMemR1

Guide for training ReMemR1 models (3B and 7B) using GRPO with revisitable memory.

## Prerequisites

### Environment Setup

```bash
conda create -n rememr1 python=3.11
conda activate rememr1
pip install httpx==0.23.1 aiohttp -U ray[serve,default] vllm
pip install nltk pyyaml beautifulsoup4 html2text wonderwords tenacity fire
pip install vllm==0.9 --index-url https://download.pytorch.org/whl/cu126
pip install "sglang==0.4.6"
pip install hydra-core accelerate tensordict torchdata wandb "tensordict<=0.6.2"
```

### Hardware Requirements

| Model | Recommended Nodes | GPUs per Node | Total GPUs |
|-------|------------------|---------------|------------|
| 3B    | 2                | 8             | 16         |
| 7B    | 4                | 8             | 32         |

Single-node training is possible by setting `N_NODE=1` (reduce `TRAIN_BS` accordingly).

### Data Preparation

1. Download training data from [HuggingFace](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main)
2. Place files under `data/train/`:
   - `hotpotqa_train_32k.parquet` (training set)
   - `hotpotqa_dev.parquet` (validation set)
3. For evaluation data, run: `bash scripts/0_run_data_process.sh`

### WandB Configuration

Set these environment variables before training:
```bash
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_PROJECT="your_project_name"
```

## Training Scripts

| Script | Model | Default Nodes |
|--------|-------|---------------|
| `scripts/1_run_train_ReMemR1_7B.sh` | Qwen/Qwen2.5-7B-Instruct | 4 |
| `scripts/1_run_train_ReMemR1_3B.sh` | Qwen/Qwen2.5-3B-Instruct | 2 |

## Single-Node Training

1. Edit the training script and set `N_NODE=1`
2. Adjust `TRAIN_BS` to fit your GPU memory (e.g., `TRAIN_BS=32` for 8 GPUs)
3. Launch directly:

```bash
bash scripts/1_run_train_ReMemR1_7B.sh
```

## Multi-Node Training

### Step 1: Start Ray Cluster

On the **head node**:
```bash
ray start --head --dashboard-host=0.0.0.0
```

On each **worker node**:
```bash
ray start --address=<head_node_ip>:6379
```

Verify all nodes are connected:
```bash
ray status
```

### Step 2: Launch Training

From the head node (project root):
```bash
bash scripts/1_run_train_ReMemR1_7B.sh
```

Logs are saved to `log/<EXP_LOG_NAME>.log` and also streamed to stdout via `tee`.


## Key Parameters (7B defaults)

Edit the parameter section at the top of the training script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_NODE` | 4 | Number of nodes (set to 1 for single-node) |
| `N_GPU` | 8 | GPUs per node |
| `MAXLEN` | 8192 | Max prompt/context length |
| `MAX_NEW_TOKEN` | 2048 | Max generation length |
| `LR` | 1e-6 | Actor learning rate |
| `ROLLOUT_N` | 16 | Number of rollout samples per prompt |
| `ROLLOUT_VAL_N` | 4 | Number of validation rollout samples |
| `ALPHA` | 0.8 | GRPO alpha for multi-level reward aggregation |
| `TRAIN_BS` | 128 | Training batch size (global) |
| `PPO_MINI_BS` | 8 | PPO mini-batch size per GPU |
| `METRIC_NAME` | em | Evaluation metric (exact match) |

## Architecture & Algorithm Details

### Entry Point

```bash
python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=5000 \
    ...  # Hydra config overrides
```

### Core Algorithm Config

- **Advantage estimator:** GRPO (`algorithm.adv_estimator=grpo`)
- **Alpha:** Controls multi-level reward aggregation (`algorithm.alpha=0.8`)
- **GRPO use advantage:** Disabled (`algorithm.grpo_use_adv=False`)
- **Action reweight:** Disabled (`algorithm.action_reweight=false`)

### Memory (Revisitable)

- **Enabled via:** `recurrent.enable=memory`
- **Chunk size:** 5000 tokens (`recurrent.memory.config.chunk_size=5000`)
- **Implementation:** `recurrent/impls/memory_revisit.py`

### Rollout Engine

- **Backend:** SGLang (`actor_rollout_ref.rollout.name=sglang`)
- **Temperature:** 1.0 (training), 1.0 (validation)
- **Top-p:** 0.999 (training), 0.7 (validation)
- **GPU memory utilization:** 0.6
- **Tensor parallel size:** 1

### FSDP & Offloading

- **FSDP size:** 8
- **Parameter offload:** Enabled (actor + ref)
- **Optimizer offload:** Enabled (actor)
- **Gradient checkpointing:** Enabled
- **Remove padding:** Enabled for efficiency

### KL Regularization

- **KL loss type:** `low_var_kl`
- **KL loss coefficient:** 0.001 (`actor_rollout_ref.actor.kl_loss_coef`)
- **KL control coefficient:** 0.001 (`algorithm.kl_ctrl.kl_coef`)

### Training Schedule

- **Total epochs:** 30
- **LR warmup steps:** 20
- **Save frequency:** Every 20 steps
- **Test frequency:** Every 10 steps
- **Validate before training:** Yes
- **Save best validation:** Yes
- **Critic warmup:** 0 steps


## Config Tuning Guide

### Scaling Down (fewer GPUs)

1. **Reduce `N_NODE`** to match your cluster (minimum 1)
2. **Reduce `TRAIN_BS`** proportionally: `TRAIN_BS = N_NODE * N_GPU * PPO_MINI_BS / desired_grad_accum`
3. **Reduce `ROLLOUT_N`** if OOM during rollout (e.g., 8 instead of 16)
4. **Increase `gpu_memory_utilization`** from 0.6 to 0.7-0.8 if GPU memory allows

### Memory & Speed Tradeoffs

| Change | Effect |
|--------|--------|
| Decrease `MAXLEN` | Less memory, but shorter context window |
| Decrease `MAX_NEW_TOKEN` | Faster rollout, less memory |
| Increase `PPO_MINI_BS` | More GPU memory needed, fewer gradient accumulation steps |
| Disable `param_offload` | Faster but needs more GPU memory |
| Increase `gpu_memory_utilization` | More VRAM for SGLang, faster rollout |
| Decrease `ROLLOUT_N` | Fewer samples per prompt, less compute per step |

### Reward & Algorithm Tuning

- **`ALPHA` (0.0-1.0):** Controls multi-level reward aggregation weight. Higher = more weight on final answer reward vs. intermediate step rewards. Default 0.8 works well.
- **`kl_loss_coef`:** Increase (e.g., 0.01) if policy diverges too fast from reference. Decrease if training is too conservative.
- **`clip_ratio_high` (0.20):** PPO clipping ratio. Increase for more aggressive updates.
- **`entropy_coeff` (0.0):** Set > 0 to encourage exploration.

### Data Configuration

- **`data.truncation=center`:** Truncates from the middle of long contexts (preserves start and end)
- **`data.filter_overlong_prompts=True`:** Skips prompts exceeding `MAXLEN`
- **`data.shuffle=False`:** Deterministic data ordering (set True for random)

## Output & Checkpoints

### Directory Structure

```
results/memory_agent/<EXP_LOG_NAME>/
├── global_step_20/
│   └── actor/          # Actor checkpoint
├── global_step_40/
│   └── actor/
├── ...
└── log/
    └── rollout_trajectory/<EXP_LOG_NAME>/  # Rollout logs
```

### Checkpoint Merging (for Evaluation)

After training converges, merge FSDP sharded checkpoints:
```bash
bash scripts/merge_ckpt.sh "results/memory_agent/ReMemR1_7B/global_step_200/actor"
```

This creates `hf_ckpt/` inside the actor directory with a standard HuggingFace checkpoint.

### Evaluation

```bash
bash scripts/2_run_eval_ReMemR1.sh
```


## Common Issues

### OOM During Rollout

- Reduce `gpu_memory_utilization` (e.g., 0.5)
- Reduce `ROLLOUT_N` (e.g., 8)
- Reduce `max_num_batched_tokens` in the rollout config
- Ensure `free_cache_engine=False` is set

### OOM During Training

- Enable both `param_offload` and `optimizer_offload` (default: both True)
- Reduce `PPO_MINI_BS`
- Reduce `ppo_max_token_len_per_gpu` (default: `MAXLEN + 8192`)
- Enable gradient checkpointing (default: True)

### Ray Cluster Issues

- Verify all nodes see each other: `ray status` should show expected node count
- Check firewall rules: Ray needs ports 6379 (GCS), 8265 (dashboard), and dynamic worker ports
- If nodes disconnect mid-training, restart Ray cluster and relaunch
- Set `ulimit -n 65535` on all nodes before starting Ray

### SGLang Rollout Failures

- Ensure `sglang==0.4.6` is installed
- Check that `tensor_model_parallel_size` matches available GPUs per rollout worker
- If rollout hangs, check SGLang server logs in the Ray dashboard

### WandB Logging Issues

- Ensure `WANDB_API_KEY` is set (not the placeholder value)
- For offline logging, set `WANDB_MODE=offline` before training
- Logger is configured as `trainer.logger=['console','wandb']`

### Training Not Converging

- Check validation metrics in WandB (exact match on HotpotQA dev)
- Ensure `trainer.val_before_train=true` to get a baseline
- If loss is unstable, try reducing `LR` (e.g., 5e-7) or increasing `kl_loss_coef`
- Check that `ALPHA=0.8` is appropriate for your reward distribution

### Checkpoint Issues

- Always run `merge_ckpt.sh` before evaluation (FSDP shards are not directly loadable)
- If merge fails, ensure the checkpoint directory contains all shard files
- Best validation checkpoint is saved when `trainer.save_best_val=true`

## Quick Reference

```bash
# Single-node 7B training (edit N_NODE=1 in script first)
bash scripts/1_run_train_ReMemR1_7B.sh

# Multi-node 7B training (4 nodes)
# Node 0 (head):
ray start --head --dashboard-host=0.0.0.0
# Nodes 1-3 (workers):
ray start --address=<head_ip>:6379
# Then on head node:
bash scripts/1_run_train_ReMemR1_7B.sh

# 3B training (2 nodes by default)
bash scripts/1_run_train_ReMemR1_3B.sh

# Merge checkpoint for evaluation
bash scripts/merge_ckpt.sh "results/memory_agent/ReMemR1_7B/global_step_200/actor"

# Run evaluation
bash scripts/2_run_eval_ReMemR1.sh
```
