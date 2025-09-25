import logging
import torch
from typing_extensions import override


from recurrent.interface import AsyncRAgent, RConfig, RDataset, RRegister, AsyncOutput
from verl.protocol import DataProto, DataProtoItem
from verl.trainer.ppo.ray_trainer import _timer
from recurrent.utils import msg

logger = logging.getLogger(__file__)
logger.setLevel('INFO')



class AsyncTrivalAgent(AsyncRAgent):
    """
    This agent works as a simple single-turn agent(standard verl), but its rollout is executed as async recurrent agent.
    """
    @override
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
        timing_raw = {}
        sample_index = gen_item.batch['sample_index'].item()
        with _timer('mt_misc', timing_raw):
            kwargs = self.sampling_params(gen_item.meta_info)
            # use default max length
            kwargs["max_completion_tokens"] = self.rollout_config.response_length
            if sample_index == 0:
                logger.info(f"generate_sequences sampling params: {kwargs}")
            conversation = list(gen_item.non_tensor_batch["raw_prompt"])
        with _timer('mt_async_gen', timing_raw):
            completions, err = await self.proxy.get_chat_completions(
                messages = conversation,
                **kwargs
            )
        with _timer('mt_misc', timing_raw):
            if err:
                raise err
            choice = completions.choices[0]
            # or `conversation.append(msg(choice))`
            conversation.append({"role": choice.message.role, 
                                 "content": choice.message.content, 
                                 "finished": choice.finish_reason=="stop"
                                 })
            conversations = [conversation]
            sample_index = torch.tensor([sample_index], dtype=torch.long)
            final_mask = torch.tensor([True], dtype=torch.bool)
        return AsyncOutput(conversations, sample_index, final_mask, timing_raw)

    
    
# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.async_path / recurrent.async_path
REGISTER = RRegister(config_cls=RConfig, dataset_cls=RDataset, agent_cls=AsyncTrivalAgent)
