# Adding a New Task/Dataset to ReMemR1

This skill guides you through adding a new task or dataset to the ReMemR1 framework. The current reference implementation is HotpotQA (multi-hop QA over long documents).

## Architecture Overview

ReMemR1 has these key components for any task:

1. **Dataset class** - loads and tokenizes data (`recurrent/impls/memory_revisit.py`)
2. **Reward function** - scores model outputs (`verl/utils/reward_score/`)
3. **Eval pipeline** - runs evaluation (`taskutils/memory_eval/`)
4. **Prompt templates** - formats input/output (`recurrent/impls/memory_revisit.py`)
5. **Data processing scripts** - prepares test data (`scripts/`)

## Step-by-Step Guide

### Step 1: Prepare Training Data

Training data must be in **Parquet** format. Required columns depend on `MemoryDataset` in `recurrent/impls/memory_revisit.py`.

**Current reference (HotpotQA):**
- Source: `BytedTsinghua-SIA/hotpotqa` on HuggingFace
- Key columns:
  - `prompt` — the question text
  - `context` — the long document/passage
  - `ground_truth` — stored in `reward_model` field for reward computation

**For your new task:**
- Create a Parquet file with at minimum:
  - A column for the long context (name configured via `context_key` param)
  - A column for the prompt/question
  - Ground truth labels accessible to the reward function
- The `context_key` config parameter in `MemoryDataset` specifies which column contains the long context. Set this to match your column name.

### Step 2: Understand the Dataset Class

The dataset class is `MemoryDataset` in `recurrent/impls/memory_revisit.py`. It extends `RDataset`.

**Key behavior:**
- `context_key` config param tells the dataset which Parquet column holds the long context
- `__getitem__` tokenizes the context and stores `prompt_ids` as a list
- The template system (see Step 5) wraps the context and prompt into the final input format

**To support a new task:**
- If your data format matches (long context + prompt), you may not need to modify `MemoryDataset` at all — just set `context_key` appropriately
- If your task has a fundamentally different structure (e.g., multiple contexts, structured inputs), you may need to subclass `MemoryDataset`

### Step 3: Add a Reward Function

Reward functions live in `verl/utils/reward_score/`. The current HotpotQA reward is in `verl/utils/reward_score/hotpotqa.py`.

**HotpotQA reference:**
- `compute_score()` computes Exact Match (EM) and F1 scores
- Called by the reward manager during training

**For your new task:**
1. Create a new file: `verl/utils/reward_score/{your_task}.py`
2. Implement `compute_score()` following the same interface as `hotpotqa.py`
3. Choose appropriate metrics for your task:
   - QA tasks: EM, F1
   - Classification: accuracy
   - Generation: ROUGE, BLEU, or custom metrics
4. Update the `reward_manager` config to point to your new reward function

**If your task uses the same EM/F1 metrics as HotpotQA**, you may be able to reuse the existing reward function directly.

### Step 4: Add Evaluation Config

The eval pipeline lives in `taskutils/memory_eval/`. The entry point is `run_eval.py`.

**How eval configs work:**
- The `Config` class in `run_eval.py` defines test tasks using the naming convention: `eval_{dataset}_{num_docs}`
- Each config specifies the dataset, number of documents, and evaluation parameters

**To add your task:**
1. Open `taskutils/memory_eval/run_eval.py`
2. Add new config entries in the `Config` class following the pattern:
   ```python
   eval_{your_dataset}_{num_docs} = EvalConfig(...)
   ```
3. Configure the test data path, number of documents, and other task-specific parameters

### Step 5: Adjust Prompt Templates (if needed)

Templates are defined in `recurrent/impls/memory_revisit.py`:
- `TEMPLATE` — the main prompt template
- `TEMPLATE_FINAL_BOXED` — asks the model to put its final answer in `\boxed{}` format

**Important:** The current templates are designed for QA tasks and expect a boxed answer format. For non-QA tasks (e.g., summarization, classification), you will likely need to:
1. Define new template strings appropriate for your task
2. Adjust the answer extraction logic accordingly

### Step 6: Update Answer Parsing (if needed)

Metric computation is in `taskutils/memory_eval/metric_utils.py`:
- `calc_test_metrics()` is the main metric calculation function
- It uses `extract_solution()`, `extract_answer()`, and `extract_boxed_answer()` for parsing model outputs

**If your task uses a different output format** (not `\boxed{}`), you need to:
1. Add a new extraction function in `metric_utils.py`
2. Update `calc_test_metrics()` to use it for your task

### Step 7: Prepare Test Data

Test data processing uses `scripts/0_run_data_process.sh`.

**For your new task:**
1. Prepare test data in the format expected by your eval config
2. Either extend the existing processing script or create a new one: `scripts/0_run_data_process_{your_task}.sh`
3. Ensure test data is accessible at the path specified in your eval config

## Quick Checklist

- [ ] Training data in Parquet format with appropriate columns
- [ ] `context_key` config set to your long-context column name
- [ ] Reward function in `verl/utils/reward_score/{task}.py` (or reuse existing)
- [ ] `reward_manager` config updated to use your reward function
- [ ] Eval config added in `taskutils/memory_eval/run_eval.py` (`eval_{dataset}_{num_docs}`)
- [ ] Prompt templates adjusted if task is not standard QA
- [ ] Answer extraction updated in `metric_utils.py` if output format differs
- [ ] Test data prepared and processing script created/updated

## Key Files Reference

| File | Purpose |
|------|---------|
| `recurrent/impls/memory_revisit.py` | Dataset class (`MemoryDataset`), templates (`TEMPLATE`, `TEMPLATE_FINAL_BOXED`) |
| `verl/utils/reward_score/hotpotqa.py` | Reference reward function (`compute_score()`) |
| `taskutils/memory_eval/run_eval.py` | Eval pipeline entry, `Config` class for eval tasks |
| `taskutils/memory_eval/metric_utils.py` | Metric computation, answer extraction functions |
| `scripts/0_run_data_process.sh` | Test data processing script |

## Minimal Example: Adding a Task Similar to HotpotQA

If your new task is also a QA task over long documents (e.g., MuSiQue, 2WikiMQA):

1. **Data:** Upload Parquet to HuggingFace or local path with `prompt`, `context`, and ground truth columns
2. **Reward:** Reuse `hotpotqa.py` if EM/F1 metrics are appropriate
3. **Eval:** Add `eval_{new_dataset}_{num_docs}` entries in `run_eval.py`
4. **Templates:** Reuse existing templates (they work for QA)
5. **No changes needed** to `MemoryDataset` if column structure matches

## Notes

- The `context_key` parameter is the most important config to get right — it tells the dataset which column to treat as the long document
- For training, data must be in Parquet format; other formats need conversion first
- The reward function interface must match what the `reward_manager` expects — follow the `compute_score()` pattern in `hotpotqa.py`
- When adding eval configs, follow the `eval_{dataset}_{num_docs}` naming convention strictly
