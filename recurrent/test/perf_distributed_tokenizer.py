import torch
import concurrent.futures
from transformers import AutoTokenizer

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
from recurrent.chat_template.utils import set_chat_template
set_chat_template(tokenizer)
tokens = [
    tokenizer.decode(i)
    for i in range(100000)
]

import random
def random_string(a, b):
    return "".join(random.choice(tokens) for _ in range(random.randint(a, b)))

import ray

@ray.remote
class TokenizerActor():
    def __init__(self, name_or_path, eos_token_id, pad_token_id):
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.tokenizer.pad_token_id = pad_token_id
        self.tokenizer.eos_token_id = eos_token_id
        set_chat_template(self.tokenizer)
    
    def call(self, method, *args, **kwargs):
        return getattr(self.tokenizer, method)(*args, **kwargs)


def init_actor_pool(num_actors, tokenizer):
    """初始化actor池"""
    actors = []
    for _ in range(num_actors):
        actor = TokenizerActor.remote(
            name_or_path=tokenizer.name_or_path,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        actors.append(actor)
    return actors


kwargs = dict(
    add_generation_prompt=False,
    return_tensors="np",
    return_dict=True,
    return_assistant_tokens_mask=True,
    tokenize=True
)
print(kwargs)

# 初始化actor池
actors = init_actor_pool(32, tokenizer)
import numpy as np
def parallel_apply_chat_template(conversations, tokenizer):
    """
    并行处理对话列表
    
    参数:
    conversations: 对话列表
    tokenizer: 分词器实例
    batch_size: 每个工作线程处理的对话数量
    max_workers: 最大工作线程数，None表示使用CPU核心数
    
    返回:
    处理后的torch tensor字典
    """
    # 将对话分成多个批次
    batch_size = len(conversations) // len(actors)
    batches = [
        conversations[i:i+batch_size] 
        for i in range(0, len(conversations), batch_size)
    ]

    futures = [
        actor.call.remote("apply_chat_template", batch, **kwargs)
        for actor, batch in zip(actors, batches)
    ]
    results = ray.get(futures)
    
    # 假设结果是一个列表，每个元素是一个torch tensor的dict
    merged_result = {}
    for key in results[0].keys():
        merged_result[key] = np.concatenate([result[key] for result in results], axis=0)
    
    return merged_result

def sequential_apply_chat_template(conversations, tokenizer):
    return tokenizer.apply_chat_template(conversations, **kwargs)
# 测试代码
def test_parallel_chat_template(size):    
    print("test", size)
    # 准备测试对话
    conversations = [
        [{"role": "user", "content": random_string(5000, 6000)}]
        for i in range(size)  # 创建32个测试对话
    ]
    
    # 并行处理
    import time
    start = time.time()
    parallel_result = parallel_apply_chat_template(conversations, tokenizer)
    end = time.time()
    print(end - start)

    # 串行处理
    start = time.time()
    sequential_result = sequential_apply_chat_template(conversations, tokenizer)
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    test_parallel_chat_template(128)
    test_parallel_chat_template(2048)   
    test_parallel_chat_template(8192)