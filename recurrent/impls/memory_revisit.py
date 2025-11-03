import logging
from dataclasses import dataclass
import re
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import TokenTemplate, chat_template, now, unpad
from recurrent.impls.tf_idf_retriever import TfidfRetriever
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel('INFO')

@dataclass
class MemoryConfig(RConfig):
    context_key: str
    max_prompt_length: int  #
    chunk_size: int  # size of each context chunk in number of tokens
    max_memorization_length: int  # max number of tokens to memorize
    # max_input_length = max_prompt_length + chunk_size + max_memorization_length + template_length
    max_chunks: int  # max number of chunks to process
    max_final_response_length: int
    # max_output_length = max_final_response_length if final else max_memorization_length

    @property
    def max_raw_input_length(self):
        return self.max_prompt_length + self.chunk_size + self.max_memorization_length

    # use property incase we want to adapt soft punishment to length.
    @property
    def gen_max_tokens_memorization(self):
        return self.max_memorization_length

    @property
    def gen_max_tokens_final_response(self):
        return self.max_final_response_length

    @property
    def gen_pad_to(self):
        return max(self.max_prompt_length, self.max_final_response_length)

class MemoryDataset(RDataset):
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
        row_dict["prompt_ids"] = self.tokenizer.encode(
            chat[0]["content"], add_special_tokens=False
        )
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
         # tensor can use 2-deminsional index for chunking.
         # while prompt_ids will not be indexed, so keep it as list.
        return ["context_ids", "context_length"], ["prompt_ids"]

TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. You should generate response in the following format:
- Output your thinking process in <thinking>your_thinking_process</thinking>.
- Read the provided section carefully and update the memory with the new information that helps to answer the problem in only one <update>the_updated_memory</update> action. Be sure to retain all relevant details from the previous memory while adding any new, useful information.
- If you notice partial key evidence that is not enough to answer the problem, also output only one `<recall>query</recall>` (e.g. `<recall>who's the president of the United States?</recall>`) to retrieve information in previous memories.

<problem> 
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

Your answer:
"""


class MemoryAgent(RAgent):
    def __init__(self, tokenizer:PreTrainedTokenizer, config: MemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        # A trick to get a simple chat_template for any tokenizer
        # the output text looks like:
        # '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n'
        # This is a format string itself, '{message}' will be replaced by the actual message.
        self.chat_template = chat_template(tokenizer)
        self.token_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE), tokenizer)
        self.token_final_message_template = TokenTemplate(self.chat_template.format(message=TEMPLATE_FINAL_BOXED), tokenizer)
        # we assume that final_message template is difinately shorter than message_template
        self.max_input_length = self.config.max_raw_input_length + self.token_message_template.length + self.config.max_raw_input_length
        logger.info(f'\n[RECURRENT] max_input_length: {self.config.max_raw_input_length}(raw) '
              f'+ {self.token_message_template.length}(message_template) = {self.max_input_length}\n')
        self.NO_MEMORY_STRING = "No previous memory"
        self.NO_MEMORY_TOKENS = torch.tensor(tokenizer.encode(self.NO_MEMORY_STRING, add_special_tokens=False), dtype=torch.long)
        self.NO_MEMORY_RECALLED_STRING = "No memory was recalled."
        self.NO_MEMORY_RECALLED_TOKENS = torch.tensor(tokenizer.encode(self.NO_MEMORY_RECALLED_STRING, add_special_tokens=False), dtype=torch.long)

        self.retriever = TfidfRetriever(tokenizer)

    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.step = 0
        self.final_mask_list = [] # only the final turn will be verified, used for reward compute
        self.sample_index_list = [] # map each turn in final to the sample id in the original batch
        
        self.ctx_length = gen_batch.batch['context_length'] # if all context is used, then the sample will no more be active
        self.bsz = len(self.ctx_length)
        self.history_memory = [set() for _ in range(self.bsz)]
        self.memory = np.empty(self.bsz, dtype=object)
        self.recall_memories = np.empty(self.bsz, dtype=object)
        self.is_final = False
    
    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        # suppose 0 is pad_token_id
        # max_chunks = 3, chunk_size = 2
        # pi is token in prompt, ti is token in chat template, 
        # [1,2] [3,4] [5,0] | p0 string
        # [1,2] [3,0] [0,0] | p1,p1 string
        # [1,0] [0,0] [0,0] | p2,p2,p2 string
        # -------- round 1 ---------
        # [1,2]            [t0,p0,t1, m,t2, 1, 2,t3]                           [ 0, 0, 0,t0,p0,t1, m,t2, 1, 2,t3]
        # [1,2]  -format-> [t0,p1,p1,t1, m,t2, 1, 2,t3] -pad2Dlist2Tendors->   [ 0, 0,t0,p1,p1,t1, m,t2, 1, 2,t3]
        # [1,0]            [t0,p2,p2,p3,t1, m,t2, 1,t3]                        [ 0, 0,t0,p2,p2,p3,t1, m,t2, 1,t3]
        # get mask & positionids
        active_mask = self.ctx_length > self.step * self.config.chunk_size
        self.active_mask = active_mask
        gen_batch = self.gen_batch
        # if all context is used, and its not done, then it will be the final turn for this batch
        if active_mask.sum().item() == 0:
            self.is_final = True
            self.messages = [
                self.token_final_message_template.format(
                    prompt=prompt,
                    memory=memory if memory is not None else self.NO_MEMORY_TOKENS,
                    recalled_memory=recalled_memory if recalled_memory is not None else self.NO_MEMORY_RECALLED_TOKENS,
                )
                for prompt, memory, recalled_memory in zip(gen_batch.non_tensor_batch['prompt_ids'], self.memory, self.recall_memories)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.int)
            final_mask = torch.full(sample_index.shape, True, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.max_final_response_length,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'FINAL TURN: MemoryAgent.next() done')
        else:
            # 1. no need to pad prompt
            # 2. context padded for 2D indexing, elegant engineering
            # 3. no need to pad memory
            prompt_i = gen_batch.non_tensor_batch['prompt_ids'][active_mask]
            chunk_i = gen_batch.batch['context_ids'][active_mask, self.config.chunk_size * self.step: self.config.chunk_size * (self.step+1)] # bs * chunk_size
            memory_i = self.memory[active_mask]
            recalled_memory_i = self.recall_memories[active_mask]
            
            # format: we use our token_template to avoid decoding & formatting with str function & encoding back.
            self.messages = [
                self.token_message_template.format(
                        prompt=prompt,
                        memory=memory if memory is not None else self.NO_MEMORY_TOKENS, # use pre-tokenized "No previous memory" for first round
                        recalled_memory=recalled_memory if recalled_memory is not None else self.NO_MEMORY_RECALLED_TOKENS,
                        chunk=chunk[chunk != self.tokenizer.pad_token_id], # unpadding needed here
                )
                for prompt, memory, recalled_memory, chunk in zip(prompt_i, memory_i, recalled_memory_i, chunk_i)
            ]
            sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask] # map active sample to original batch
            final_mask = torch.full(sample_index.shape, False, dtype=torch.bool) # all False
            self.meta_info = {'input_pad_to': self.max_input_length,
                         'pad_to': self.config.gen_pad_to,
                         'generation_kwargs': {
                          'max_tokens': self.config.gen_max_tokens_memorization,
                          'n': 1 # note that we have already repeat n times in ray_trainer
                        }}
            logger.info(f'MemoryAgent.action() done')
        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        return self.messages, self.meta_info

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        all_decoded_responses = self.tokenizer.batch_decode(gen_output.batch['responses'], skip_special_tokens=True) # List[str], length: [recalled_bsz]
        # update recalled memory
        recalled_queries = [self._parse_recall_query(response) for response in all_decoded_responses] # List[str], length: [recalled_bsz]
        if not self.is_final:
            active_indices = self.active_mask.nonzero().squeeze().cpu().numpy()
        else:
            active_indices = torch.arange(len(recalled_queries), dtype=torch.int)
        recalled_memories = [self.retriever.top1_retrieve(query, self.history_memory[idx]) for query, idx in zip(recalled_queries, active_indices)] # List[str], length: [recalled_bsz]
        recalled_memories_values = [
            torch.tensor(self.tokenizer.encode(memory_str, add_special_tokens=False), dtype=torch.long) if memory_str is not None else self.NO_MEMORY_RECALLED_TOKENS
            for memory_str in recalled_memories
        ]
        recalled_memories_arr = np.empty(len(recalled_memories_values), dtype=object)
        recalled_memories_arr[:] = recalled_memories_values
        gen_output.batch['recalled_memories'] = recalled_memories_arr

        if not self.is_final:
            self.memory[self.active_mask] = unpad(self.tokenizer, gen_output.batch['responses'], remove_eos=True)
            all_update_memories = [self._parse_update_memory(response) for response in all_decoded_responses] # List[str], length: [recalled_bsz]

            # update memory
            update_values = [
                torch.tensor(self.tokenizer.encode(memory_str, add_special_tokens=False), dtype=torch.long) if memory_str is not None else self.NO_MEMORY_TOKENS
                for memory_str in all_update_memories
            ]
            updates_arr = np.empty(len(update_values), dtype=object)
            updates_arr[:] = update_values
            self.memory[self.active_mask] = updates_arr # List[torch.Tensor], shape: [recalled_bsz]
            self.recall_memories[self.active_mask] = recalled_memories_arr

            # update history memory
            self.update_memory(all_update_memories, active_indices)

        self.log_step(gen_output)
        self.step += 1
        return gen_output

    def update_memory(self, memory_strings: List[str], active_indices: List[int]):
        # self.history_memory is a set
        assert len(active_indices) == len(memory_strings)
        new_memories = [memory_str if memory_str is not None else self.NO_MEMORY_STRING for memory_str in memory_strings]
        for idx, memory in zip(active_indices, new_memories):
            self.history_memory[int(idx)].add(memory)
    
    def _preprocess_text(self, text: str) -> set[str]:
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Split into words and return a set of unique tokens
        return set(text.split())
    
    def look_up_memory(self, query: str, idx: int) -> str:
        assert False
        """Look up the top-1 memory based on the query."""
        scores = []
        if query is None:
            return None
        query_tokens = self._preprocess_text(query)
        for memory_string in self.history_memory[idx]:
            memory_tokens = self._preprocess_text(memory_string)
            intersection = query_tokens.intersection(memory_tokens)
            union = query_tokens.union(memory_tokens)
            if not union:
                score = 0.0
            else:
                score = len(intersection) / len(union)
            if score > 0:
                scores.append((memory_string, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[0][0] if scores else None

    ## MODIFIED: Helper function to parse the callback ID from the LLM's text output
    def _parse_recall_query(self, text_response: str) -> str:
        """Extracts the chunk ID from a string like '<recall>who's the president of the United States?</recall>'."""
        try:
            match = re.search(r'<recall>(.+)</recall>', text_response)
            if match:
                query = match.group(1) # we pick the first group, which is the query
                return query
        except (ValueError, TypeError):
            pass # Fall through to return None if parsing fails
        return None

    def _parse_update_memory(self, text_response: str) -> str:
        try:
            cleaned = re.sub(r'<recall>.*?</recall>', '', text_response, flags=re.DOTALL)
            return cleaned.strip()
        except (ValueError, TypeError):
            return None

    @override
    def done(self):
        return self.is_final
    
    @override
    def end(self):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.memory
        del self.messages
        del self.history_memory
        del self.recall_memories
        del self.active_mask
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index
        

    def log_step(self, gen_output):
        """Log multi-turn conversation details in a single consolidated function.
        """
        def clip_long_string(string, max_length=3000):
            """Clip long string to a maximum length."""
            if not len(string) > max_length:
                return string
            return string[:max_length//2] + '\n\n...(ignored)\n\n' + string[-max_length//2:]

        # Header with dynamic step number
        step = self.step if not self.is_final else "FINAL"
        active_count = self.active_mask.sum().item() if not self.is_final else self.bsz
        logger.info(f"\n{' '*10}{'='*30}[RECURRENT] STEP{step} [active: {active_count}/{self.bsz}] {'='*30}{' '*10}")

        # Message and Response section
        if self.active_mask[0]:
            decoded_message = self.tokenizer.decode(self.messages[0])
            rsp0 = gen_output.batch['responses'][0]
            decoded_response = self.tokenizer.decode(rsp0[rsp0!=self.tokenizer.pad_token_id])
            logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            logger.info(f"{' '*10}{'-'*20}prompt end{'-'*20}{' '*10}")
            logger.info(f"[RESPONSE] {decoded_response}")
            logger.info(f"{' '*10}{'-'*20}response end{'-'*20}{' '*10}")
        else:
            logger.info("MESSAGE and RESPONSE is empty since it is not active.")


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryAgent)
