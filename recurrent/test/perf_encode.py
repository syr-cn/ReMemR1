from transformers import AutoTokenizer
from recurrent.chat_template.utils import set_chat_template

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
set_chat_template(tokenizer)
tokens = [
    tokenizer.decode(i)
    for i in range(10000)
]

import random
def random_string(a, b):
    return "".join(random.choice(tokens) for _ in range(random.randint(a, b)))
# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
import timeit
print("start")
time = timeit.timeit(lambda: tokenizer.encode(random_string(8192, 8192)), number=100)
print(time / 100) # 25ms
