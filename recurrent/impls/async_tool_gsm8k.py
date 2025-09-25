import logging
from omegaconf import DictConfig
import torch
from typing_extensions import override
from dataclasses import dataclass

from transformers import PreTrainedTokenizer
from recurrent.async_utils import ChatCompletionProxy


from recurrent.interface import AsyncRAgent, RConfig, RDataset, RRegister, AsyncOutput
from verl.protocol import DataProtoItem
from verl.trainer.ppo.ray_trainer import _timer
from recurrent.utils import msg, log_step
from recurrent.tool import ToolSchema, ToolCall, toolcall_system_prompt, toolcall_extract, merge_system_prompt

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

@dataclass
class RToolGsm8kConfig(RConfig):
    max_turn: int
    max_generation_per_turn: int

class AsyncToolGsm8kAgent(AsyncRAgent):
    """
    This agent works as a simple single-turn agent(standard verl), but its rollout is executed as async recurrent agent.
    """
    TOOLS = [
        ToolSchema(
            name="calc_gsm8k_reward",
            description="A tool for calculating the reward of gsm8k. (1.0 if parsed answer is correct, 0.0 if parsed answer is incorrect or not correctly parsed)",
            parameters={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The model's answer to the GSM8K math problem, must be a digits",
                    },
                },
                "required": ["answer"],
            },
        )
    ]
    def __init__(self, proxy: ChatCompletionProxy, tokenizer: PreTrainedTokenizer, config: RConfig, rollout_config: DictConfig):
        self.system_msg = toolcall_system_prompt(self.TOOLS)
        super().__init__(proxy, tokenizer, config, rollout_config)

    async def tool_message(self, gen_item: DataProtoItem, parsed: ToolCall):
        if parsed.name == "calc_gsm8k_reward":
            answer = parsed.args.get("answer", None)
            if not answer:
                return {"role": "tool", "content": "`answer` not found in parsed args"}
            if not isinstance(answer, str):
                answer = str(answer)
            from verl.utils.reward_score.gsm8k import compute_score
            # keep align with Sglang's demo in verl, 
            # can also use a customized dataset to simplify the code.
            reward = compute_score(answer, gen_item.non_tensor_batch["tools_kwargs"]["calc_gsm8k_reward"]["create_kwargs"]["ground_truth"])
            return {"role": "tool", "content": f"Current parsed {answer=} {reward=}"}
        elif parsed.name == ToolSchema.INVALID_TOOL:
            return {"role": "tool", "content": f"{parsed.args['msg']}"}
        else:
            return {"role": "tool", "content": f"Unknown tool {parsed.name}"}

    @override
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
        timing_raw = {}
        sample_index = gen_item.batch['sample_index'].item()
        kwargs = self.sampling_params(gen_item.meta_info)
        kwargs["max_completion_tokens"] = self.config.max_generation_per_turn
        kwargs["stop"] = ["</tool_call>"]
        if sample_index == 0:
            logger.info(f"generate_sequences sampling params: {kwargs}")

        conversation = merge_system_prompt(list(gen_item.non_tensor_batch["raw_prompt"]), self.system_msg)
        with _timer('mt_async_gen', timing_raw):
            completions, err = await self.proxy.get_chat_completions(
                messages = conversation,
                **kwargs
            )
        if err:
            raise err
        conversation.append(msg(completions.choices[0]))
        if sample_index == 0:
            log_step(logger, 0, conversation)

        step = 0
        while step < self.config.max_turn:
            # should not use choice directly, it does not conain the </tool_call>
            parsed = toolcall_extract(conversation[-1]["content"])
            if not parsed:
                break
            tool_msg = await self.tool_message(gen_item, parsed)
            conversation.append(tool_msg)
            with _timer('mt_async_gen', timing_raw):
                completions, err = await self.proxy.get_chat_completions(
                    messages = conversation,
                    **kwargs
                )
            if err:
                raise err
            choice = completions.choices[0]
            conversation.append(msg(choice))
            step += 1
            if sample_index == 0:
                log_step(logger, step, conversation[-2:])
            
        sample_index = torch.tensor([sample_index], dtype=torch.long)
        final_mask = torch.tensor([True], dtype=torch.bool)
        metric = {"workflow/tool_call": step}
        return AsyncOutput([conversation], sample_index, final_mask, timing_raw, metric)

    
    
# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.async_path / recurrent.async_path
REGISTER = RRegister(config_cls=RToolGsm8kConfig, dataset_cls=RDataset, agent_cls=AsyncToolGsm8kAgent)
