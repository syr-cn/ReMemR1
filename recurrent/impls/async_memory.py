import logging
from typing import List, Optional, Tuple, Union, Dict
from uuid import uuid4

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override
from recurrent.async_utils import ChatCompletionProxy


from recurrent.interface import AsyncRAgent, RConfig, RDataset, RRegister, AsyncOutput
from recurrent.utils import log_step, msg
from verl.protocol import DataProtoItem
from verl.trainer.ppo.ray_trainer import _timer
import verl.utils.torch_functional as verl_F


logger = logging.getLogger(__file__)
logger.setLevel('INFO')


from recurrent.impls.memory import MemoryConfig, TEMPLATE, TEMPLATE_FINAL_BOXED

class AsyncMemoryDataset(RDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(
        self,
        recurrent_config: MemoryConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if data_config.truncation != 'center':
            raise ValueError('MemoryDataset only support center truncation')
        data_config.max_prompt_length=recurrent_config.max_chunks * recurrent_config.chunk_size
        self.context_key = recurrent_config.context_key
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )


    @override
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)
        context = row_dict.pop(self.context_key)

        model_inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)

        context_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        context_ids, attention_mask = verl_F.postprocess_data(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id, # pyright: ignore
            left_pad=False,
            truncation=self.truncation,
        )

        row_dict["context_ids"] = context_ids[0]
        lengths = attention_mask.sum(dim=-1)
        row_dict["context_length"] = lengths[0]
        row_dict["prompt"] = chat[0]["content"]
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt will not be indexed, so keep it as list.
        return ["context_ids", "context_length"], ["prompt"]


class AsyncMemoryAgent(AsyncRAgent):
    def __init__(self, proxy: ChatCompletionProxy, tokenizer: PreTrainedTokenizer, config: RConfig, rollout_config: DictConfig):
        super().__init__(proxy, tokenizer, config, rollout_config)
    @override
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
        """
        tensor keys in output:
        - standard verl: "prompts", "responses", "input_ids", "attention_mask", "position_ids"
        - recurrent rl: "sample_index", "final_mask"
        > input_ids = torch.cat([prompts, responses], dim=1)
        """
        timing_raw = {}
        sample_index = gen_item.batch['sample_index'].item()
        context_length = gen_item.batch["context_length"].item()
        step = 0
        conversations = []
        memory = None
        while True:
            with _timer('mt_mics', timing_raw):
                if step * self.config.chunk_size >= context_length:
                    break
                assert step < self.config.max_chunks, f"{step=} exceeds {self.config.max_chunks=}, {context_length=}"
                chunk_ids = gen_item.batch["context_ids"]\
                    [step * self.config.chunk_size: (step + 1) * self.config.chunk_size]

                kwargs = self.sampling_params(gen_item.meta_info)
                if sample_index == 0:
                    logger.info(f"generate_sequences sampling params: {kwargs}")
                self.config : MemoryConfig
                kwargs["max_completion_tokens"] = self.config.max_memorization_length

                conversation = [
                    {"role": "user", "content": TEMPLATE.format(
                        prompt=gen_item.non_tensor_batch["prompt"],
                        memory=memory if memory else "No previous memory",
                        chunk=self.tokenizer.decode(chunk_ids, skip_special_tokens=True),
                    )}
                ]
            with _timer('mt_async_gen', timing_raw):
                completions, err = await self.proxy.get_chat_completions(
                    messages=conversation,
                    **kwargs
                )
                if err:
                    raise err
            with _timer('mt_mics', timing_raw):
                choice = completions.choices[0]
                conversation.append(msg(choice))
                conversations.append(conversation)
                memory = conversation[-1]["content"]
                if sample_index == 0:
                    log_step(logger, step, conversation)
            step += 1

        # final turn
        with _timer('mt_mics', timing_raw):
            conversation = [
                {
                    "role": "user",
                    "content": TEMPLATE_FINAL_BOXED.format(
                        prompt=gen_item.non_tensor_batch["prompt"],
                        memory=memory if memory else "No previous memory",
                    ),
                }
            ]
            kwargs["max_completion_tokens"] = self.config.max_final_response_length
        with _timer('mt_async_gen', timing_raw):
            completions, err = await self.proxy.get_chat_completions(
                messages=conversation,
                **kwargs
            )
        with _timer('mt_mics', timing_raw):
            if err:
                raise err
            choice = completions.choices[0]
            conversation.append(msg(choice))
            conversations.append(conversation)
            if sample_index == 0:
                log_step(logger, step, conversation)

            sample_index = torch.full((len(conversations),), sample_index, dtype=torch.long)
            final_mask = torch.full((len(conversations),), False, dtype=torch.bool)
            final_mask[-1] = True
        return AsyncOutput(conversations, sample_index, final_mask, timing_raw)


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=AsyncMemoryDataset, agent_cls=AsyncMemoryAgent)
