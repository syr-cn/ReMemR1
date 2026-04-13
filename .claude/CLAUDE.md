# ReMemR1 — Revisitable Memory for Long-Context LLM Agents

**Paper:** "Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents" (ICLR 2026)
**Authors:** Yaorui Shi, An Zhang, Xiang Wang et al.

ReMemR1 trains LLM agents to process long documents chunk-by-chunk while maintaining and revisiting an evolving memory. The agent learns through GRPO-based RL with a multi-level reward combining outcome correctness (EM/F1) and step-level state advantages (format + action rewards for memory updates/recalls).

## Architecture Overview

The system has three pillars:
1. **Recurrent Agent** — Chunk-by-chunk document processing with memory update/recall actions
2. **Multi-Level Reward** — Outcome reward + state advantage, mixed by alpha (default 0.8)
3. **Training via verl** — GRPO advantage, Ray distributed training, SGLang rollout, FSDP

## Directory Structure

```
ReMemR1/
  recurrent/                    # Core recurrent agent framework
    interface.py                # Abstract bases: RAgent, RDataset, RConfig, RRegister, AsyncRAgent
    generation_manager.py       # Sync generation loop orchestrator
    async_generation_manager.py # Async generation loop orchestrator
    utils.py                    # TokenTemplate, chat_template(), pad/unpad, final_batch()
    tool.py                     # Tool interface
    async_utils.py              # ChatCompletionProxy for async rollout
    impls/                      # Concrete implementations
      memory_revisit.py         # MemoryAgent, MemoryConfig, MemoryDataset, REGISTER
      tf_idf_retriever.py       # TfidfRetriever (sklearn-based)
      async_memory.py           # Async version of memory agent
    chat_template/              # Chat template utilities
    test/                       # Unit tests and perf benchmarks
  verl/                         # Modified verl RL training framework
    trainer/
      main_ppo.py               # Training entry point (hydra config)
      ppo/
        ray_trainer.py          # Main training loop, reward aggregation, advantage computation
        core_algos.py           # PPO/GRPO core algorithms, KL controllers
        metric_utils.py         # Format/action/data metrics, validation metrics
    workers/                    # Distributed worker implementations
      actor/                    # Actor (policy) workers
      critic/                   # Critic workers
      rollout/                  # Rollout workers (SGLang/vLLM)
      reward_manager/           # Reward computation managers
      reward_model/             # Reward model workers
      sharding_manager/         # FSDP sharding
    utils/
      reward_score/
        hotpotqa.py             # Outcome reward: EM/F1 for QA tasks
        math.py, gsm8k.py, ... # Other reward functions
      dataset/rl_dataset.py     # Base RLHFDataset
      checkpoint/               # FSDP/Megatron checkpoint managers
    models/                     # Model implementations (Qwen2, LLaMA, Megatron)
    protocol.py                 # DataProto — the universal data container
  taskutils/
    memory_eval/                # Evaluation pipeline
      run_eval.py               # Main eval entry point
      utils/
        rememr1.py              # Inference: async_query_llm() for ReMemR1 evaluation
        tf_idf_retriever.py     # TF-IDF retriever (eval copy)
        envs.py                 # Eval env config (URL, API_KEY, chunk sizes)
    data_synthesis/             # Data preprocessing scripts
  scripts/
    0_run_data_process.sh       # Download and preprocess benchmarks
    1_run_train_ReMemR1_7B.sh   # Train ReMemR1-7B (reference config)
    1_run_train_ReMemR1_3B.sh   # Train ReMemR1-3B
    2_run_eval_ReMemR1.sh       # Run evaluation
    merge_ckpt.sh               # Merge FSDP checkpoints
    model_merger.py             # Model merging utility
    converter_hf_to_mcore.py    # HF to Megatron-Core conversion
```

## Key Abstractions

### RAgent / RConfig / RDataset / RRegister (`recurrent/interface.py`)

The plugin system for recurrent agents. All custom implementations register via `RRegister`:

```python
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryAgent)
```

- **RConfig** — Dataclass base for agent config. Subclass adds domain-specific fields.
- **RDataset** — Extends `RLHFDataset`. Override `__getitem__` and `get_bactch_keys` (note: typo in codebase, keep it).
- **RAgent** — Lifecycle: `start()` -> loop of `action()` -> rollout -> `update()` until `done()` -> `end()`. Returns `(final_mask, sample_index)` from `end()`.
- **AsyncRAgent** — Alternative async interface with `rollout()` method for single-sample async generation.
- **RRegister.from_filename()** — Dynamic loading of implementations from file path + object name.

### MemoryAgent (`recurrent/impls/memory_revisit.py`)

The core ReMemR1 agent (402 lines). Key design:

- **Chunk processing:** Iterates over context chunks (sized `config.chunk_size` tokens). Active samples tracked via `active_mask = ctx_length > step * chunk_size`.
- **Memory state:** `self.memory` (current memory per sample), `self.history_memory` (set of all past memories per sample), `self.recall_memories` (last recalled memory).
- **Token-level templates:** Uses `TokenTemplate` for format-then-tokenize (avoids decode/re-encode roundtrip). Templates: `TEMPLATE` (chunk processing) and `TEMPLATE_FINAL_BOXED` (final answer).
- **Actions in LLM output:**
  - `<thinking>...</thinking>` — reasoning trace
  - `<update>...</update>` — updated memory content
  - `<recall>query</recall>` — triggers TF-IDF retrieval over `history_memory`
- **Callback mechanism:** On `<recall>`, `TfidfRetriever.top1_retrieve(query, history_memory[idx])` fetches the most relevant past memory.
- **Final turn:** When all chunks consumed (`active_mask.sum() == 0`), switches to `TEMPLATE_FINAL_BOXED` expecting `\boxed{answer}`.
- **action_type encoding:** 0 = final, 1 = callback, 2 = memory update.

### TfidfRetriever (`recurrent/impls/tf_idf_retriever.py`)

Lightweight retriever using sklearn's `TfidfVectorizer` with the LLM tokenizer as custom tokenizer. `top1_retrieve(query, corpus)` returns the best-matching historical memory string.

### TokenTemplate (`recurrent/utils.py`)

Format strings operating on token IDs directly. Splits template by `{keyword}` placeholders, pre-tokenizes the static parts, then `format(**kwargs)` concatenates token tensors. Avoids decode/encode overhead.

### DataProto (`verl/protocol.py`)

The universal data container throughout verl. Holds `batch` (TensorDict), `non_tensor_batch` (dict of numpy arrays), and `meta_info` (dict). All workers communicate via DataProto.

## Multi-Level Reward System

Defined in `ray_trainer.py` L1288-1340 and `metric_utils.py`. Two levels mixed by `alpha` (default 0.8):

### Outcome Advantage (weight = alpha)
- Applied to the **final answer** only.
- Reward function in `verl/utils/reward_score/hotpotqa.py`: `compute_score()` extracts `\boxed{answer}` and computes EM or F1 against ground truth.
- Advantage computed via `compute_1D_grpo_advantage()` using `reward_batch.non_tensor_batch['uid']` as group index.

### State Advantage (weight = 1 - alpha)
- Applied to **every step** (memory update, recall, final).
- **Format reward** (`compute_format_rewards`): 1.0 if response contains exactly one valid tag for its action type (one `<update>`, one `<recall>`, or one `\boxed{}`), else 0.0.
- **Action reward** (`compute_action_rewards`):
  - For memory steps (action_type=2): word-level recall improvement. Compares ground-truth word overlap before/after memory update, plus recall retrieval benefit.
  - For callback (action_type=1) and final (action_type=0): 0.0.
- State reward = format_reward + action_reward.
- Advantage computed via `compute_1D_grpo_advantage()` using `step_uid` (uid + step_id) as group index — each step at each question gets its own GRPO group.

### Optional Action Reweighting
- `compute_action_reweights()`: reweights advantages by ground-truth word coverage in prompt. Range [0.5, 1.5]. Enabled via `algorithm.action_reweight=true`.

### compute_1D_grpo_advantage (`ray_trainer.py` L249)
- Groups samples by index (uid for outcome, step_uid for state).
- Computes per-group mean and std.
- Returns normalized (score - mean) / std per sample. Single-sample groups get advantage 0 with std 1.

## Training Pipeline

### Entry Point
```bash
python3 -m verl.trainer.main_ppo recurrent.enable=memory ...
```
Uses Hydra for config. Key training parameters (from `1_run_train_ReMemR1_7B.sh`):

| Parameter | Default | Description |
|---|---|---|
| `recurrent.enable` | `memory` | Enable memory agent mode |
| `recurrent.memory.path` | `recurrent/impls/memory_revisit.py` | Agent implementation file |
| `recurrent.memory.config.chunk_size` | 5000 | Tokens per context chunk |
| `algorithm.adv_estimator` | `grpo` | Only GRPO supported for recurrent |
| `algorithm.alpha` | 0.8 | Outcome vs state advantage mixing |
| `algorithm.grpo_use_adv` | False | Whether to normalize by std |
| `algorithm.action_reweight` | false | Enable action reweighting |
| `actor_rollout_ref.rollout.n` | 16 | Rollout samples per prompt |
| `actor_rollout_ref.rollout.name` | `sglang` | Rollout engine |
| `actor_rollout_ref.actor.use_kl_loss` | True | Required for recurrent mode |
| `actor_rollout_ref.actor.kl_loss_coef` | 0.001 | KL penalty coefficient |
| `data.truncation` | `center` | Required: center truncation for long contexts |
| `data.context_key` | `context` | Column name for document context in dataset |
| `trainer.n_gpus_per_node` | 8 | GPUs per node |
| `trainer.nnodes` | 4 | Number of nodes (32 GPUs total for 7B) |

### Training Loop (ray_trainer.py)
1. Sample batch from dataset
2. **Recurrent rollout:** MemoryAgent.start() -> loop action/rollout/update -> end()
3. Compute outcome reward (EM/F1 on final answers)
4. Compute state rewards (format + action) for all steps
5. Compute GRPO advantages (outcome + state, mixed by alpha)
6. Apply advantages to response tokens via `eos_mask`
7. Update actor with PPO loss + KL penalty
8. Periodic validation and checkpointing

### Infrastructure
- **Ray** — Distributed orchestration
- **SGLang** — Rollout engine (inference)
- **FSDP** — Model sharding with param/optimizer offload
- **Gradient checkpointing** — Enabled by default
- **WandB** — Experiment tracking

## Evaluation Pipeline

### Running Eval
```bash
bash scripts/2_run_eval_ReMemR1.sh
```
Runs `taskutils/memory_eval/run_eval.py`. Configure via environment variables in `taskutils/memory_eval/utils/envs.py`:
- `URL` — Model serving endpoint
- `API_KEY` — API key
- `RECURRENT_CHUNK_SIZE` — Chunk size for eval
- `RECURRENT_MAX_NEW` — Max new tokens
- `RECURRENT_MAX_CONTEXT_LEN` — Max context length

### Inference Logic (`taskutils/memory_eval/utils/rememr1.py`)
`async_query_llm()` mirrors the training MemoryAgent:
1. Tokenize context, apply center truncation if needed
2. For each chunk: format with TEMPLATE, call LLM, parse `<update>` and `<recall>`, update memory and history
3. Final turn: format with TEMPLATE_FINAL_BOXED, call LLM, return response
4. Uses TF-IDF retriever for recall (same as training)

### Metrics (`metric_utils.py`)
- `calc_test_metrics()` — EM, sub-EM, F1, precision, recall
- `extract_solution()` / `extract_answer()` — Answer extraction from `\boxed{}`
- `process_validation_metrics()` — Best-of-N, worst-of-N, majority voting with bootstrap

### Data Preparation
```bash
bash scripts/0_run_data_process.sh
```
Downloads HotpotQA and 2WikiMultiHopQA from FlashRAG datasets, processes into parquet.

## Coding Conventions

- **Python 3.10+** with type hints. Uses `typing_extensions` for `@override`.
- **Dataclasses** for configs (`@dataclass` extending `RConfig`).
- **OmegaConf/Hydra** for all training configuration.
- **TensorDict** from `tensordict` for batch data. Note: importing TensorDict initializes CUDA.
- **NumPy object arrays** (`np.empty(n, dtype=object)`) for ragged tensor lists (variable-length memory tokens).
- **Left-padded tensors** by default for generation; context tensors right-padded for 2D indexing.
- **Logging** via Python `logging` module, level set to INFO.
- **Known typo:** `get_bactch_keys` in `RDataset` — keep this spelling for compatibility.
- **Action types:** 0 = final answer, 1 = callback/recall, 2 = memory update. Encoded as integers in `batch['action_type']`.
- **XML-style tags** in LLM output: `<thinking>`, `<update>`, `<recall>`, and `\boxed{}` for final answers.
- **Regex parsing** for all tag extraction (not XML parsers).

## Common Tasks

### Add a New Recurrent Agent
1. Create `recurrent/impls/my_agent.py`
2. Define `MyConfig(RConfig)`, `MyDataset(RDataset)`, `MyAgent(RAgent)`
3. Export `REGISTER = RRegister(config_cls=MyConfig, dataset_cls=MyDataset, agent_cls=MyAgent)`
4. Set `recurrent.memory.path=recurrent/impls/my_agent.py` in training config

### Add a New Reward Function
1. Create `verl/utils/reward_score/my_task.py` with `compute_score(solution_str, ground_truth, reward_metric)`
2. Wire it in the reward manager config

### Modify the Reward Signal
- Format rewards: edit `compute_format_rewards()` in `metric_utils.py`
- Action rewards: edit `compute_action_rewards()` in `metric_utils.py`
- Outcome rewards: edit the task-specific file in `verl/utils/reward_score/`
- Alpha mixing: `algorithm.alpha` in training config

### Add a New Evaluation Benchmark
1. Add data processing in `taskutils/data_synthesis/`
2. Create eval script in `taskutils/memory_eval/utils/`
3. Wire into `run_eval.py`

### Debug Rollout
- `run_memory_debug.sh` — Debug script for local rollout testing
- MemoryAgent logs detailed step-by-step info via `log_step()` (message, response, active count)
- Set `VERL_LOGGING_LEVEL=DEBUG` for verbose output

## Key Gotchas

- **Only GRPO works for recurrent mode.** PPO raises `NotImplementedError`.
- **KL loss is required** (`use_kl_loss=True`). Without it, recurrent mode raises `NotImplementedError`.
- **Center truncation required** for MemoryDataset (`data.truncation='center'`).
- **`sample_index` vs `final_mask`:** `sample_index` maps each rollout step to the original batch index; `final_mask` marks which steps contain the final answer.
- **Memory is a set** (`history_memory`), not a list — duplicate memories are automatically deduplicated.
- **TF-IDF retriever re-fits on every query** (`fit_transform` on corpus each time) — this is intentional for the small per-sample memory corpus.
