# Modify Reward System

Guide for modifying the ReMemR1 reward system. The reward architecture has five layers:
format rewards, action rewards, action reweights, outcome rewards, and multi-level advantage aggregation.

## Architecture Overview

ReMemR1 uses a **multi-level reward + GRPO advantage** system:

```
Final advantage = alpha * outcome_advantage + (1-alpha) * state_advantage
```

- `outcome_advantage`: GRPO-normalized advantage over the final EM/F1 score (grouped by `uid`)
- `state_advantage`: GRPO-normalized advantage over per-step (format + action) rewards (grouped by `step_uid`)
- `alpha`: mixing coefficient (default 0.8, set in `scripts/1_run_train_ReMemR1_7B.sh`)
- When `alpha=1.0`, only the outcome reward is used (no per-step shaping)

## Key Files

| File | What It Controls |
|------|-----------------|
| `verl/trainer/ppo/ray_trainer.py` L1288-1340 | Multi-level advantage aggregation |
| `verl/trainer/ppo/ray_trainer.py` L249-288 | `compute_1D_grpo_advantage()` — per-group normalization |
| `verl/trainer/ppo/metric_utils.py` | `compute_format_rewards()`, `compute_action_rewards()`, `compute_action_reweights()` |
| `verl/utils/reward_score/hotpotqa.py` | `compute_score()` — outcome reward (EM / F1) |
| `scripts/1_run_train_ReMemR1_7B.sh` | Training config including `alpha` and `action_reweight` toggle |

## 1. Format Rewards

**File:** `verl/trainer/ppo/metric_utils.py` → `compute_format_rewards()`

Binary reward (1.0 or 0.0) based on whether the step output matches the expected format:

| `action_type` | Step Type | Format Check |
|--------------|-----------|-------------|
| 0 | Final answer | Contains `\boxed{...}` answer |
| 1 | Callback (recall) | Contains exactly one `<recall>...</recall>` tag |
| 2 | Memory (update) | Contains exactly one `<update>...</update>` tag |

**To modify:** Change the regex patterns or add new action types. To add a new step type,
add a new `action_type` value and corresponding format check branch.

## 2. Action Rewards

**File:** `verl/trainer/ppo/metric_utils.py` → `compute_action_rewards()`

Per-step reward measuring memory quality improvement:

- **Memory steps (action_type=2):** Computes two components:
  1. **Recall improvement:** Word-level overlap improvement of `generated_memory` vs `previous_memory`
     against `ground_truth`. Measures whether the memory update brings content closer to the answer.
  2. **Revisit reward:** Contribution of `recalled_memory` — rewards the agent for retrieving useful
     information from memory.
- **Callback steps (action_type=1):** Returns 0.0 (no action reward)
- **Final steps (action_type=0):** Returns 0.0 (no action reward)

**To modify:** Adjust the word overlap computation, change how recall improvement is measured,
or add action rewards to callback/final steps.

## 3. Action Reweights

**File:** `verl/trainer/ppo/metric_utils.py` → `compute_action_reweights()`

Optional multiplicative reweighting of memory steps based on relevance:

- Computes word overlap between memory content and `ground_truth`
- Maps to range **[0.5, 1.5]** — irrelevant memory steps get downweighted, relevant ones upweighted
- **Enabled via:** `algorithm.action_reweight` config flag
- Only applies to memory steps (action_type=2); other steps get weight 1.0

**To modify:** Adjust the weight range, change the overlap metric, or add reweighting for other step types.

## 4. Outcome Reward

**File:** `verl/utils/reward_score/hotpotqa.py` → `compute_score()`

Final-answer quality measured by:

- **Exact Match (EM):** Binary 1.0/0.0 — does the `\boxed{...}` answer exactly match any entry in `ground_truth` list?
- **F1 Score:** Token-level F1 between predicted answer and best-matching ground truth

The choice of EM vs F1 is configured in the training script.

**To modify:** Add new answer quality metrics, change normalization, or support additional answer formats.

## 5. Multi-Level Advantage Aggregation

**File:** `verl/trainer/ppo/ray_trainer.py` L1288-1340

This is where all rewards are combined into the final training signal:

### Step-by-step computation:

1. **Per-step reward:** `format_reward + action_reward` (optionally multiplied by `action_reweight`)
2. **State advantage:** `compute_1D_grpo_advantage()` over per-step rewards, grouped by `step_uid`
   - `step_uid` groups all rollouts of the same step together
   - GRPO normalizes: `(score - mean) / std` within each group
3. **Outcome advantage:** `compute_1D_grpo_advantage()` over final EM/F1 reward, grouped by `uid`
   - `uid` groups all rollouts of the same problem together
4. **Final advantage:** `alpha * outcome_advantage + (1-alpha) * state_advantage`

### `compute_1D_grpo_advantage()` (L249-288):

- Groups scores by the provided index (uid or step_uid)
- Computes mean and std per group
- Returns normalized score: `(score - group_mean) / group_std`
- Handles edge cases (single sample, zero std)

## Common Modification Recipes

### Recipe A: Add a new reward component to per-step rewards

1. Add your reward function in `verl/trainer/ppo/metric_utils.py`
2. Call it in `ray_trainer.py` alongside `compute_format_rewards()` / `compute_action_rewards()`
3. Add the new reward to the per-step reward sum before `compute_1D_grpo_advantage()`

### Recipe B: Change the outcome reward metric

1. Edit `verl/utils/reward_score/hotpotqa.py` → `compute_score()`
2. Add your metric (e.g., ROUGE, BERTScore) alongside or replacing EM/F1
3. Return the new score (must be a float)

### Recipe C: Adjust the outcome vs step-level tradeoff

1. Modify `alpha` in `scripts/1_run_train_ReMemR1_7B.sh`
2. `alpha=1.0` → pure outcome reward (no step-level shaping)
3. `alpha=0.0` → pure step-level reward (no outcome signal)
4. Default `alpha=0.8` → mostly outcome, some step-level shaping

### Recipe D: Add a new action type

1. Define a new `action_type` integer (e.g., 3 for a new step type)
2. Add format check in `compute_format_rewards()` for the new type
3. Add action reward logic in `compute_action_rewards()` if needed
4. Add reweight logic in `compute_action_reweights()` if needed
5. Ensure the data pipeline assigns the correct `action_type` to new steps

### Recipe E: Disable step-level rewards entirely

1. Set `alpha=1.0` in the training script
2. This makes `final_advantage = outcome_advantage` only
3. Format and action rewards will still be computed (for logging) but not used for training

### Recipe F: Enable/disable action reweighting

1. Toggle `algorithm.action_reweight` in the training config
2. When enabled: memory step rewards are multiplied by relevance weight [0.5, 1.5]
3. When disabled: all steps get uniform weight 1.0

## Reward Flow Diagram

```
For each rollout step:
  format_reward = compute_format_rewards(action_type, output)    # 0.0 or 1.0
  action_reward = compute_action_rewards(action_type, ...)       # float >= 0
  reweight      = compute_action_reweights(action_type, ...)     # [0.5, 1.5] or 1.0
  step_reward   = (format_reward + action_reward) * reweight

For the final step:
  outcome_reward = compute_score(boxed_answer, ground_truth)     # EM or F1

Advantage computation:
  state_advantage   = GRPO_normalize(step_rewards, group_by=step_uid)
  outcome_advantage = GRPO_normalize(outcome_rewards, group_by=uid)
  final_advantage   = alpha * outcome_advantage + (1-alpha) * state_advantage
```

## Debugging Tips

- **Check reward distributions:** Log per-step rewards to verify they are meaningful (not all 0 or all 1)
- **Monitor advantage stats:** After GRPO normalization, advantages should have ~zero mean within each group
- **Verify grouping:** Ensure `uid` and `step_uid` correctly group related rollouts
- **Watch for degenerate cases:** If all rollouts in a group get the same reward, GRPO std=0 and advantage=0 (no gradient signal)
- **Test with alpha=1.0 first:** If step-level rewards cause instability, fall back to outcome-only training
