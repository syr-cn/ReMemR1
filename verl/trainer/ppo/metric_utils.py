# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

import re
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from verl import DataProto
from taskutils.memory_eval.utils import (
    extract_answer,
    extract_boxed_answer,
    extract_solution,
    exact_match_score,
    sub_exact_match_score,
    f1_score,
)


def calc_test_metric_single(pred, gold):
    if pred is None:
        return {'sub_em': 0, 'em': 0, 'f1': 0, 'prec': 0, 'recall': 0, 'valid_num': 0}
    em = exact_match_score(pred, gold)
    subem = sub_exact_match_score(pred, gold)

    f1, prec, recall = f1_score(pred, gold)
    metric_dict = {'sub_em': subem, 'em': float(em), 'f1': f1, 'prec': prec, 'recall': recall, 'valid_num': 1}
    return metric_dict

def calc_test_metrics(responses, goldens):
    assert len(responses) == len(goldens)
    metrics = defaultdict(list)
    for response, gold in zip(responses, goldens):
        assert isinstance(response, str)
        pred, _ = extract_solution(response)
        pred = extract_answer(pred) if pred else extract_answer(response)
        # refer to taskutils/memory_eval/ruler_hqa.py:get_pred for the original code

        if isinstance(gold, list):
            metric_dicts = [calc_test_metric_single(pred, gold_item) for gold_item in gold]
            metric_dict = max(metric_dicts, key=lambda d: d['f1'])
        else:
            metric_dict = calc_test_metric_single(pred, gold)
        for k, v in metric_dict.items():
            metrics[k].append(v)
    return metrics

def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )

def compute_format_rewards(all_responses_str: List[str], batch: DataProto) -> torch.Tensor:
    format_rewards = []
    for response_str, action_type in zip(all_responses_str, batch.batch['action_type']):
        action_type = int(action_type)
        if action_type == 1: # callback, not used in revisit version
            all_recall_queries = re.findall(r'<recall>(.*?)</recall>', response_str)
            all_recall_queries = [recall_query for recall_query in all_recall_queries if recall_query.strip()]
            reward_score = 1.0 if len(all_recall_queries) == 1 else 0.0

        elif action_type == 2: # memory
            all_update_memories = re.findall(r'<update>(.*?)</update>', response_str)
            all_update_memories = [memory for memory in all_update_memories if memory.strip()]
            reward_score = 1.0 if len(all_update_memories) == 1 else 0.0

        elif action_type == 0: # final
            all_final_answers = re.findall(r'\\boxed{(.+)}', response_str)
            all_final_answers = [answer for answer in all_final_answers if answer.strip()]
            reward_score = 1.0 if len(all_final_answers) == 1 else 0.0
        else:
            raise ValueError(f"Invalid action type: {action_type}")
        format_rewards.append(reward_score)
    format_rewards = torch.tensor(format_rewards, dtype=torch.float32, device=batch.batch['responses'].device)
    return format_rewards


def compute_action_reweights(all_prompt_str: List[str], batch: DataProto, reward_batch: DataProto, sample_index: torch.Tensor) -> torch.Tensor:
    action_reweight_scalar = []

    for prompt_str, action_type, reward_batch_item in zip(all_prompt_str, batch.batch['action_type'], reward_batch[sample_index]):
        action_type = int(action_type)
        ground_truth_list = reward_batch_item.non_tensor_batch['reward_model']['ground_truth']
        if action_type == 1 or action_type == 0: # callback or final: directly assign a reweight score of 1
            reweight_score = 1.0
        elif action_type == 2: # memory: word-level recall
            reweight_score = 0
            for ground_truth in ground_truth_list:
                ground_truth_words = list(ground_truth.split())
                if not len(ground_truth_words):
                    continue
                reweight_score += sum([1.0 if word in prompt_str else 0.0 for word in ground_truth_words]) / len(ground_truth_words)
            reweight_score /= len(ground_truth_list) # word-level recall
            reweight_score += 0.5 # range: [0.5, 1.5]
        else:
            raise ValueError(f"Invalid action type: {action_type}")
        action_reweight_scalar.append(reweight_score)
    action_reweight_scalar = torch.tensor(action_reweight_scalar, dtype=torch.float32, device=batch.batch['responses'].device)
    return action_reweight_scalar

def compute_action_rewards(all_prompt_str: List[str], all_responses_str: List[str], all_recalled_memories_str: List[str], batch: DataProto, reward_batch: DataProto, sample_index: torch.Tensor, rewarded_scalar: torch.Tensor) -> torch.Tensor:
    action_reward_scalar = []
    rewarded_mask = rewarded_scalar > 0.0
    assert sample_index.shape == rewarded_mask.shape

    def recall_metric(predict, ground_truth_list):
        reward_score = 0.0
        for ground_truth in ground_truth_list:
            ground_truth_words = list(ground_truth.split())
            if not len(ground_truth_words):
                continue
            reward_score += sum([1.0 if word in predict else 0.0 for word in ground_truth_words]) / len(ground_truth_words)
        reward_score /= len(ground_truth_list)
        return reward_score

    for prompt_str, response_str, recalled_memories_str, action_type, reward_batch_item in zip(all_prompt_str, all_responses_str, all_recalled_memories_str, batch.batch['action_type'], reward_batch[sample_index]):
        action_type = int(action_type)
        ground_truth_list = reward_batch_item.non_tensor_batch['reward_model']['ground_truth']
        if action_type == 1: # callback: min_callback as golden standard, calculate penalty for over-callback
            reward_score = 0.0
        elif action_type == 2: # memory: word-level recall reward
            # reward for memory update
            previous_memory = re.findall(r'<memory>(.*?)</memory>', prompt_str)
            previous_memory = previous_memory[0] if previous_memory else ''
            generated_memory = re.findall(r'<update>(.*?)</update>', response_str)
            generated_memory = generated_memory[0] if generated_memory else ''
            generated_memory_recall = recall_metric(generated_memory, ground_truth_list)
            previous_memory_recall = recall_metric(previous_memory, ground_truth_list)
            update_reward_score = generated_memory_recall - previous_memory_recall

            # reward for memory revisit
            generated_revisit_recall = recall_metric(recalled_memories_str, ground_truth_list)
            prompt_recall = recall_metric(prompt_str, ground_truth_list)
            revisit_reward_score = max(0, generated_revisit_recall - prompt_recall)

            # aggregate the rewards
            reward_score = update_reward_score + revisit_reward_score
        elif action_type == 0: # final: word-level recall reward
            # we omit the final reward for now
            reward_score = 0.0
        else:
            raise ValueError(f"Invalid action type: {action_type}")
        action_reward_scalar.append(reward_score)
    action_reward_scalar = torch.tensor(action_reward_scalar, dtype=torch.float32, device=batch.batch['responses'].device)
    return action_reward_scalar

def calculate_callback_count(prompt: str, response: str):
    x = re.findall(r'<recall>(.*?)</recall>', response)
    if '<recall>' in prompt:
        # Only keep the callback count for the callback decision turns
        return len(x) > 0
    else:
        return None

def parse_documents(prompt: str):
    x = re.findall(r'<section>(.*?)</section>', prompt)
    return x[0] if x else ''

def calculate_update_count(prompt: str, response: str):
    x = re.findall(r'<update>(.*?)</update>', response)
    if '<update>' in prompt:
        return len(x) > 0
    else:
        return None

def calculate_callback_distance(prompt: str, response: str):
    callback_ids = re.findall(r'<recalled_memory> \(step=(\d+)\)', prompt) # callback ID
    callback_ids = [int(i) for i in callback_ids if int(i) >= 1]

    range_ids = re.findall(r'<current_memory> \(step=(\d+)\)', prompt) # range ID
    range_ids = [int(i) for i in range_ids if int(i) >= 1]

    if len(callback_ids) == 0:
        return 0
    
    if len(range_ids) == 0:
        return 0

    distance = abs(callback_ids[0] - range_ids[0]) # We only consider the first callback ID and chunk ID for now
    return distance

def compute_action_metrics(batch: DataProto, tokenizer) -> Dict[str, Any]:
    prompts = []
    responses = []
    for i in range(len(batch)):
        data_item = batch[i]
        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]
        
        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        # NOTE: we assume the prompt is not needed, please make sure that is true!
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses'] 
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        prompts.append(prompt_str)
        responses.append(response_str)
    
    metrics = {}
    for metric_name, metric_fn in [
        ('callback_count', calculate_callback_count),
        ('update_count', calculate_update_count),
        # ('callback_distance', calculate_callback_distance),
    ]:
        metric_results = [metric_fn(prompt, response) for prompt, response in zip(prompts, responses)]
        metric_results = [metric_result for metric_result in metric_results if metric_result is not None]
        metrics[metric_name + '/mean'] = float(np.mean(metric_results)) if metric_results else 0.0
        metrics[metric_name + '/max'] = int(np.max(metric_results)) if metric_results else 0
        metrics[metric_name + '/min'] = int(np.min(metric_results)) if metric_results else 0

    return metrics

def update_validate_metrics(test_batch: DataProto, output_texts) -> Dict[str, Any]:
    metrics = {}
    responses = []
    ground_truths = []
    for data_item, output_text in zip(test_batch, output_texts):
        responses.append(output_text)
        ground_truths.append(data_item.non_tensor_batch['reward_model']['ground_truth'])
    
    answer_metrics = calc_test_metrics(responses, ground_truths)
    metrics = {f'test_{k}': v for k, v in answer_metrics.items()}

    return metrics

def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    ####### HACK: use binay score
    passrate = (sequence_score == 1.0).float()
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # passrate
        'critic/pass@n':
            torch.mean(passrate).detach().item(),
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # Metrics for step-wise rewards
    rewards_format = batch.batch["rewards_format"] if "rewards_format" in batch.batch else None
    rewards_action = batch.batch["rewards_action"] if "rewards_action" in batch.batch else None
    rewards_step = batch.batch["rewards_step"] if "rewards_step" in batch.batch else None
    action_type = batch.batch["action_type"] if "action_type" in batch.batch else None
    advantages_step = batch.batch["advantages_step"] if "advantages_step" in batch.batch else None
    advantages_overall = batch.batch["advantages_overall"] if "advantages_overall" in batch.batch else None
    
    metrics.update({
        "critic/rewards_format/mean": torch.mean(rewards_format).detach().item(),
        "critic/rewards_format/max": torch.max(rewards_format).detach().item(),
        "critic/rewards_format/min": torch.min(rewards_format).detach().item(),
    } if rewards_format is not None else {})
    
    metrics.update({ # compute the mean, max, min of rewards_action
        "critic/rewards_action/mean": torch.mean(rewards_action).detach().item(),
        "critic/rewards_action/max": torch.max(rewards_action).detach().item(),
        "critic/rewards_action/min": torch.min(rewards_action).detach().item(),
    } if rewards_action is not None else {})
    
    metrics.update({ # compute the mean, max, min of rewards_step
        "critic/rewards_step/mean": torch.mean(rewards_step).detach().item(),
        "critic/rewards_step/max": torch.max(rewards_step).detach().item(),
        "critic/rewards_step/min": torch.min(rewards_step).detach().item(),
    } if rewards_step is not None else {})
    
    metrics.update({ # compute the ratio of each action type
        "critic/action_type/0": torch.mean((action_type == 0).float()).detach().item(),
        "critic/action_type/1": torch.mean((action_type == 1).float()).detach().item(),
        "critic/action_type/2": torch.mean((action_type == 2).float()).detach().item(),
    } if action_type is not None else {})
    
    metrics.update({ # compute the mean, max, min of advantages_step
        "critic/advantages_step/mean": torch.mean(advantages_step).detach().item(),
        "critic/advantages_step/max": torch.max(advantages_step).detach().item(),
        "critic/advantages_step/min": torch.min(advantages_step).detach().item(),
    } if advantages_step is not None else {})
    
    metrics.update({ # compute the mean, max, min of advantages_overall
        "critic/advantages_overall/mean": torch.mean(advantages_overall).detach().item(),
        "critic/advantages_overall/max": torch.max(advantages_overall).detach().item(),
        "critic/advantages_overall/min": torch.min(advantages_overall).detach().item(),
    } if advantages_overall is not None else {})

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.

    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample

    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                ##### Add avg_passrate for 0-1 reward
                #metric[f'avg_passrate@{n_resps}'] = np.mean([float(r > 0.0) for r in var_vals])
                metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                n = 2
                while n < n_resps:
                    ns.append(n)
                    n *= 2
                ns.append(n_resps)

                for n in ns:
                    # Best/Worst-of-N
                    [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                    metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                    metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                    # Majority voting
                    if var2vals.get("pred", None) is not None:
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                        [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                            data=vote_data,
                            subset_size=n,
                            reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                            seed=seed,
                        )
                        metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
