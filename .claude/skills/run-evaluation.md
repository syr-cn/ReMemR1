# Running ReMemR1 Evaluation

This guide covers the full evaluation pipeline for ReMemR1: data preparation, checkpoint merging, model serving, running evaluations, and interpreting results.

## Prerequisites

- **Hardware:** 8 GPUs (evaluation uses tensor parallelism across all 8)
- **Software:**
  - Python 3.10+
  - SGLang (for model serving)
  - PyTorch with FSDP support
  - HuggingFace Transformers and Datasets
  - `uvloop` (async event loop)
- **Trained checkpoint:** e.g., `results/memory_agent/ReMemR1_7B/global_step_200/actor`
- **Environment:** All commands run from repo root

## Step-by-Step Evaluation

### Step 0: Prepare Test Data

Download and process evaluation datasets (HotpotQA and 2WikiMultiHopQA):

```bash
bash scripts/0_run_data_process.sh
```

This runs `taskutils/data_synthesis/process_test.py` which:
- Downloads HotpotQA and 2WikiMultiHopQA from HuggingFace (`RUC-NLPIR/FlashRAG_datasets`)
- Processes 128 subsets per dataset
- Saves processed data to `data/test/`

Verify data exists:
```bash
ls data/test/
# Should contain hotpotqa and 2wikimultihopqa subdirectories
```
### Step 1: Merge Checkpoints

Training saves FSDP-sharded checkpoints. Merge them into a single HuggingFace checkpoint:

```bash
bash scripts/merge_ckpt.sh "results/memory_agent/ReMemR1_7B/global_step_200/actor"
```

This script:
1. Runs `scripts/model_merger.py` with `--backend fsdp` to merge sharded weights
2. Reads from `<ckpt_path>/huggingface` (base model config) and `<ckpt_path>/` (trained shards)
3. Writes merged model to `<ckpt_path>/hf_ckpt/`
4. Copies tokenizer configs from `<ckpt_path>/huggingface/` to `<ckpt_path>/hf_ckpt/`

Verify the merged checkpoint:
```bash
ls results/memory_agent/ReMemR1_7B/global_step_200/actor/hf_ckpt/
# Should contain model weights, config.json, tokenizer files
```
### Step 2: Run Evaluation

```bash
bash scripts/2_run_eval_ReMemR1.sh
```

This script:
1. Sets `PROJECT_ROOT` to the current directory and `DATAROOT` to `data/test/`
2. Calls `python3 taskutils/memory_eval/run_eval.py`
3. Logs output to `log/eval/run_eval.log`

#### What happens under the hood

The evaluation pipeline in `run_eval.py` uses a `Config` class with a **serve-then-test** pattern:

1. **Serve:** Launches the model via SGLang with tensor parallelism
   - Command: `python3 -m sglang.launch_server --model-path <ckpt> --tensor-parallel-size <tp> --data-parallel-size <dp> --port <SERVE_PORT>`
   - Data parallelism is computed as `dp = 8 / tp` (uses all 8 GPUs)
   - Waits for server readiness by polling `/v1/models` endpoint

2. **Test:** Runs evaluations across all task combinations:
   - **Datasets:** `hotpotqa`, `2wikimultihopqa`
   - **Document counts:** `[50, 100, 200, 400, 800, 1600, 3200, 6400]`
   - Total: 16 evaluation tasks (2 datasets x 8 doc counts)
   - Each task runs `test_qa.py` with the appropriate API method

3. **Cleanup:** Kills the SGLang server after all tests complete

#### ReMemR1 Config in run_eval.py

The ReMemR1 evaluation configs use:
- `method="rememr1"` (activates the callback/recall inference pipeline)
- `tp=4` (tensor parallelism = 4, so data parallelism = 2)
- `concur=256` (concurrent requests)
- `RECURRENT_CHUNK_SIZE=5000` tokens per chunk
- `RECURRENT_MAX_NEW=2048` max new tokens per generation
- `RECURRENT_MAX_CONTEXT_LEN=100000000000` (effectively unlimited)
### Step 3: Understanding ReMemR1 Inference

The ReMemR1 inference pipeline (`taskutils/memory_eval/utils/rememr1.py`) implements the **revisitable memory** mechanism:

1. **Chunking:** The input context is split into chunks of `RECURRENT_CHUNK_SIZE` tokens
2. **Per-chunk processing:** Each chunk is processed with `TEMPLATE` which instructs the model to:
   - Read the chunk section
   - Think about it in `<thinking>` tags
   - Update memory with new information via `<update>` tags
   - Optionally recall earlier information via `<recall>query</recall>` tags
3. **Memory accumulation:** After each chunk:
   - The `<update>` content becomes the new running memory
   - If a `<recall>` query is issued, a `TfidfRetriever` searches all previous memories and returns relevant recalled content
4. **Final answer:** After all chunks are processed, `TEMPLATE_FINAL_BOXED` prompts the model to answer the question based on accumulated memory, outputting the answer in `\boxed{}`

This mirrors the training-time mechanism, ensuring consistency between training and evaluation.
## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVE_PORT` | `8000` | Port for the SGLang model server |
| `DASH_PORT` | `8265` | Port for the dashboard (used for server shutdown) |
| `RECURRENT_CHUNK_SIZE` | None | Token count per context chunk (set to `5000` for ReMemR1) |
| `RECURRENT_MAX_NEW` | None | Max new tokens per generation step (set to `2048` for ReMemR1) |
| `RECURRENT_MAX_CONTEXT_LEN` | None | Max total context length (set very high for ReMemR1) |
| `REVERSED` | `0` | If `1`, reverses the order of model configs to evaluate |
| `DATAROOT` | `$PROJECT_ROOT/data/test` | Root directory for test data |
| `TOKENIZERS_PARALLELISM` | - | Set to `false` during data processing to avoid warnings |

## Interpreting Results

### Metrics

Evaluation computes these metrics in `test_qa.py` via `calc_metrics()`:

| Metric | Description |
|--------|-------------|
| **EM** (Exact Match) | Whether the predicted answer exactly matches the gold answer |
| **sub_EM** | Whether the gold answer is a substring of the prediction |
| **F1** | Token-level F1 score between prediction and gold answer |
| **Precision** | Token-level precision |
| **Recall** | Token-level recall |

### Results Location

Results are saved per-task in:
```
taskutils/memory_eval/results/eval_<dataset>_<num_docs>/<config_name>.jsonl
```

For example:
```
taskutils/memory_eval/results/eval_hotpotqa_50/ReMemR1-7B.jsonl
taskutils/memory_eval/results/eval_hotpotqa_6400/ReMemR1-7B.jsonl
taskutils/memory_eval/results/eval_2wikimultihopqa_800/ReMemR1-7B.jsonl
```

### What to Look For

- **Scaling behavior:** Performance across increasing `num_docs` (50 -> 6400) shows how well the model handles growing context
- **ReMemR1 advantage:** The recurrent memory mechanism should maintain performance at high doc counts where direct-context models degrade
- **Cross-dataset consistency:** Compare HotpotQA vs 2WikiMultiHopQA patterns
## Custom Evaluation

### Evaluating a Different Checkpoint

Edit `taskutils/memory_eval/run_eval.py` — update the `ckpt` path in the relevant Config:

```python
ReMemR1_7B = Config(
    name="ReMemR1-7B",
    ckpt="results/memory_agent/ReMemR1_7B/global_step_200/actor/hf_ckpt",  # your merged ckpt path
    tp=4,
    method="rememr1",
    concur=256,
    env=ENV(RECURRENT_MAX_CONTEXT_LEN=100000000000, RECURRENT_CHUNK_SIZE=5000, RECURRENT_MAX_NEW=2048),
)
```

Then update the `CONFIGS` list to only include your target config.

### Evaluating a Subset of Tasks

Modify `RULER_TEST_TASKS` or the `run_test_tasks()` function to filter specific datasets or doc counts:

```python
# Only evaluate on small doc counts for quick testing
TEST_NUM_DOCS = [50, 100, 200]

# Or filter specific datasets
RULER_TEST_TASKS = [f"eval_hotpotqa_{n}" for n in TEST_NUM_DOCS]
```

### Comparing Methods

The evaluation supports multiple inference methods via the `method` parameter:
- `"rememr1"` — ReMemR1 with callback/recall (uses `\boxed{}` answer extraction)
- `"recurrent"` — Basic recurrent memory without recall
- `"openai"` — Direct context (standard OpenAI-compatible API, no chunking)
- `"recurrent-rag"` — Recurrent with RAG
- `"boxed"` / `"recurrent-boxed"` — Variants using boxed answer format

### Adjusting Chunk Size

The `RECURRENT_CHUNK_SIZE` controls the granularity of memory updates. Smaller chunks = more updates but finer-grained memory; larger chunks = fewer updates but more context per step. Default is `5000` tokens.

### Running with Different GPU Counts

The `tp` (tensor parallelism) parameter determines GPU usage. Data parallelism is auto-computed as `8/tp`. For fewer GPUs, adjust `tp` accordingly and ensure `8/tp` is an integer, or modify the `serve()` method in `Config`.

## Quick Reference

```bash
# Full pipeline from trained checkpoint to evaluation results:
# 1. Prepare data (one-time)
bash scripts/0_run_data_process.sh

# 2. Merge checkpoint
bash scripts/merge_ckpt.sh "results/memory_agent/ReMemR1_7B/global_step_200/actor"

# 3. Update ckpt path in run_eval.py, then run
bash scripts/2_run_eval_ReMemR1.sh

# 4. Check results
ls taskutils/memory_eval/results/
```
