# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import logging
from typing import Dict, List, Tuple, Type
from omegaconf import DictConfig
import torch
import numpy as np
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast
from verl import DataProto

from .interface import AsyncRAgent, RConfig, AsyncOutput
from .async_utils import run_coroutine_in_chat_scheduler_loop, ChatCompletionProxy
from .chat_template.utils import set_chat_template
from verl.workers.rollout.async_server import AsyncLLMServerManager
from verl.trainer.ppo.ray_trainer import _timer
from .utils import create_position_ids, pad_tensor_list_to_length

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

STARTING_MSG = [{"role": "user", "content": "padding"}]
class AsyncLLMGenerationManager:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        async_server: AsyncLLMServerManager,
        config: RConfig,
        rollout_config: DictConfig,
        agent_cls: Type[AsyncRAgent]
    ):
        self.config = config
        self.rollout_config = rollout_config
        set_chat_template(tokenizer)
        self.tokenizer = tokenizer
        if "generation" not in self.tokenizer.chat_template:
            raise ValueError("tokenizer.chat_template must support return_assistant_tokens_mask, see https://huggingface.co/docs/transformers/main/chat_templating")
        self.async_server = async_server
        assert isinstance(self.async_server.chat_scheduler, ChatCompletionProxy), \
            "async_server.chat_scheduler must be a ChatCompletionProxy for Async Recurrent RL"
        self.agent_cls = agent_cls
        self.agent = agent_cls(self.async_server.chat_scheduler, self.tokenizer, self.config, self.rollout_config)


    def run_llm_loop(self, gen_batch: DataProto, timing_raw) -> Tuple[DataProto, torch.BoolTensor, torch.LongTensor]:
        """Run main LLM generation loop.
        genbatch: 'context_ids','context_length','prompt_ids'
        timing_raw: timing dict used in ray_trainer, note that we will accumulate the time cost in this loop, instead of override each time as in ray_trainer.
        see `_timer` implementation at the top of this file for more details.
        """
        with _timer("mt_engine", timing_raw):
            self.async_server.wake_up()
            self.agent.start(gen_batch, timing_raw)
            gen_batch.batch["sample_index"] = torch.arange(len(gen_batch), dtype=torch.long)
        async def rollout_coro():
            async def rollout(b):
                async_output = await self.agent.rollout(b)
                with _timer("mt_output", async_output.timing_raw):
                    batch = self.tokenize_output(async_output)
                    async_output.batch = batch
                return async_output
            return await asyncio.gather(*[rollout(b) for b in gen_batch])
        gen_output_list = run_coroutine_in_chat_scheduler_loop(self.async_server, rollout_coro())
        with _timer("mt_gather", timing_raw):
            gen_output = self.concat_output([o.batch for o in gen_output_list])
            sample_index = torch.cat([o.sample_index for o in gen_output_list])
            final_mask = torch.cat([o.final_mask for o in gen_output_list])
            assert sum(final_mask) == len(gen_batch)
            timing_raw.update(self.agent.reduce_timings([g.timing_raw for g in gen_output_list]))
            gen_output.meta_info["metrics"] = self.agent.reduce_metrics([g.metrics for g in gen_output_list])
        with _timer("mt_engine", timing_raw):
            self.agent.end()
            self.async_server.sleep()
        return gen_output, final_mask, sample_index # pyright: ignore
    
    def concat_output(self, batch_list: list[dict]) -> DataProto:
        starting = self.tokenizer.apply_chat_template(STARTING_MSG, add_generation_prompt=True)
        len_starting = len(starting)
        for i in range(len_starting):
            assert batch_list[0]["responses"][0][i] == starting[i], i
        concated = {
            k: np.concatenate([b[k] for b in batch_list], axis=0)
            for k in batch_list[0].keys()
        }
        prompt_ids, prompt_attn_mask = pad_tensor_list_to_length(
            [torch.from_numpy(arr) for arr in concated['prompts']],
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            return_mask=True
        )
        response_ids, response_attn_mask = pad_tensor_list_to_length(
            [torch.from_numpy(arr) for arr in concated['responses']],
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=False,
            return_mask=True
        )
        response_mask = pad_tensor_list_to_length(
            [torch.from_numpy(arr) for arr in concated['response_mask']],
            pad_token_id=0,
            left_pad=False,
        )
        response_ids = response_ids[:, len_starting:]
        response_attn_mask = response_attn_mask[:, len_starting:]
        response_mask = response_mask[:, len_starting:]

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attn_mask, response_attn_mask], dim=1)
        position_ids = create_position_ids(attention_mask)
        return DataProto(
            batch=TensorDict(
                {
                    "prompts": prompt_ids,
                    "responses": response_ids,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "response_mask": response_mask,
                },
                batch_size=len(prompt_ids),
            )
        )
        
    def tokenize_output(self, gen_output: AsyncOutput) -> dict[str, np.ndarray]:
        def get_prompt_and_response(conv):
            if conv[0]["role"] == "system":
                prompts = conv[:2]
                responses = conv[2:]
            else:
                prompts = conv[:1]
                responses = conv[1:]
            assert len(responses), f"empty response for conv={conv}"
            return prompts,  STARTING_MSG + responses

        p_r = [get_prompt_and_response(conv) for conv in gen_output.conversations]
        encoded_prompt = self.tokenizer.apply_chat_template([p for p, _ in p_r],
            add_generation_prompt=True,
            return_tensors="np",
            padding='do_not_pad',
            return_dict=True,
        )
        encoded_response = self.tokenizer.apply_chat_template([r for _, r in p_r],
            add_generation_prompt=False,
            return_tensors="np",
            padding='do_not_pad',
            return_dict=True,
            return_assistant_tokens_mask=True
        )
        def to1D(arr):
            if len(arr.shape) > 1:
                empty_arr = np.empty(arr.shape[0], dtype=object)
                empty_arr[:] = [a for a in arr]
                return empty_arr
            else:
                return arr
        batch = {
            'prompts': to1D(encoded_prompt['input_ids']),
            'responses': to1D(encoded_response['input_ids']),
            'response_mask': to1D(encoded_response['assistant_masks']),
        }
        return batch
       
        