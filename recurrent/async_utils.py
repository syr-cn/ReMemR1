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
from typing import Any, Callable, Dict, List, Optional, Tuple, Coroutine
import heapq
import asyncio
import aiohttp
from omegaconf import DictConfig
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from uuid import uuid4
from verl.workers.rollout.async_server import ChatCompletionScheduler, AsyncLLMServerManager
from httpx import AsyncClient, Timeout, Limits
class ChatCompletionProxy(ChatCompletionScheduler):
    """
    I still want to utilize the LLM executor and chat_scheduler_thread managed by the AsyncLLMServerManager.
    But I'd like to run a coroutine instead of use the `callback chain` pattern in 
    `examples.ppo_trainer.naive_chat_scheduler.NaiveChatCompletionScheduler`.

    So make ChatCompletionScheduler a proxy that returns the ChatCompletion directly, instead of 
    a scheduler that triggers a callback chain, and directly submit the coroutine which starts 
    the rollout of the whole batch to the chat_scheduler_loop.
    """
    def __init__(self, config: DictConfig, model_path: str, server_addresses: List[str], max_cache_size: int = 10000):
        self.addr_client_map = {}
        conn = aiohttp.TCPConnector(limit=len(server_addresses) * 1024, limit_per_host=1024, keepalive_timeout=600, 
                                    loop=asyncio.get_event_loop()) # aiohttp use get_running_loop(), but the loop is not launched yet
        self.session = aiohttp.ClientSession(connector=conn)
        super().__init__(config, model_path, server_addresses, max_cache_size)
    
    def get_client(self, address) -> AsyncClient:
        return self.addr_client_map.get(address, AsyncClient(
                timeout=Timeout(connect=60, read=None, write=None, pool=None),
                limits=Limits(max_connections=8192, max_keepalive_connections=8192, keepalive_expiry=600),
            ))

    async def submit_chat_completions(self, callback: Callable[[ChatCompletion, Dict[str, Any], Exception], None], callback_additional_info: Dict[str, Any], **chat_complete_request):
        raise NotImplementedError("ChatCompletionProxy does not support submit_chat_completions")
    
    async def _chat_completions_openai(self, address: str, **chat_complete_request) -> ChatCompletion:
        client = AsyncOpenAI(
            base_url=f"http://{address}/v1",
            api_key="token-abc123",
            http_client=self.get_client(address)
        )
        return await client.chat.completions.create(**chat_complete_request)

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        extra_headers = chat_complete_request.pop("extra_headers")
        async with self.session.post(
            url=f"http://{address}/v1/chat/completions",
            headers={"Authorization": "Bearer token-abc123", **extra_headers},
            json=chat_complete_request,
        ) as resp:
            data = await resp.json()
            return ChatCompletion(**data)

    
    async def get_chat_completions(
        self,
        model=None,
        **chat_complete_request,
    ) -> Tuple[ChatCompletion, Optional[Exception]]:
        """
        Submit a chat completion request to the server with the least number of requests.

        Args:
            **chat_complete_request: dict, request parameters same as OpenAI AsyncCompletions.create.
                OpenAI API reference: https://platform.openai.com/docs/api-reference/chat/create
        """
        model = model or self.model_name
        if "extra_headers" not in chat_complete_request:
            chat_complete_request["extra_headers"] = {}

        extra_headers = chat_complete_request["extra_headers"]
        request_id = extra_headers.get("x-request-id", None)
        if request_id:
            if request_id.startswith("chatcmpl-"):
                request_id = request_id[len("chatcmpl-") :]
                extra_headers["x-request-id"] = request_id

            address = self.request_id_to_address[request_id]
        else:
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

            request_id = uuid4().hex
            self.request_id_to_address[request_id] = address
            chat_complete_request["extra_headers"]["x-request-id"] = request_id

        completions, exception = None, None
        try:
            # TODO: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp(address, model=model, **chat_complete_request)
        except Exception as e:
            # Let user handle the exception
            exception = e

        return completions, exception


def run_coroutine_in_chat_scheduler_loop(async_server: AsyncLLMServerManager, coro: Coroutine):
    """
    Adapted from AsyncLLMServerManager, a clever way to run a coroutine in a seperate thread.
    Originally designed to run an async method of chat_scheduler, now we use it to start `RAsyncAgent.rollout`
    and gather the results of them.
    """
    assert async_server.chat_scheduler is not None, "chat scheduler is not initialized."
    future = asyncio.run_coroutine_threadsafe(coro, async_server.chat_scheduler_loop)
    return future.result()
