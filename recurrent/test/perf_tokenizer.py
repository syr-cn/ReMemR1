from transformers import AutoTokenizer

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

from recurrent.chat_template.utils import set_chat_template
set_chat_template(tokenizer)
tokens = [
    tokenizer.decode(i)
    for i in range(10000)
]

import random
def random_string(a, b):
    return "".join(random.choice(tokens) for _ in range(random.randint(a, b)))
strings = [
    random_string(5000, 6000) for _ in range(2048)
]

import time
start = time.time()
tokenizer(strings, return_tensors="pt", padding="longest", padding_side="left")
end = time.time()
print(end - start)


from recurrent.interface import AsyncOutput
import torch
import random
def random_conv():
    return [
        {"role": "user", "content": random_string(5000, 6000)},
        {"role": "assistant", "content":  random_string(100, 2048)},
    ]
def random_output(idx):
    bsz = random.randint(6, 7)
    convs = [random_conv() for _ in range(bsz)]
    sample_index = torch.tensor([idx] * bsz)
    final_mask = torch.tensor([0 if i < bsz - 1 else 1 for i in range(bsz)])
    return AsyncOutput(convs, sample_index, final_mask, {})
from tqdm import tqdm
output_list = [
    random_output(i)
    for i in tqdm(range(2048))
]

from verl.trainer.ppo.ray_trainer import _timer
from recurrent.utils import create_position_ids
def test1():
    timing = {}
    def concat_output(gen_output_list):
        STARTING_MSG = [{"role": "user", "content": "padding"}]
        def get_prompt_and_response(conv):
            if conv[0]["role"] == "system":
                prompts = conv[:2]
                responses = conv[2:]
            else:
                prompts = conv[:1]
                responses = conv[1:]
            assert len(responses), f"empty response for conv={conv}"
            return prompts,  STARTING_MSG + responses
        with _timer("1", timing):
            p_r = [get_prompt_and_response(conv) for out in gen_output_list for conv in out.conversations]
        with _timer("2", timing):
            encoded_prompt = tokenizer.apply_chat_template([p for p, _ in p_r],
                add_generation_prompt=True,
                return_tensors="pt",
                padding="longest",
                tokenizer_kwargs=dict(padding_side="left"),
                return_dict=False,
                tokenize=True
            )
        with _timer("3", timing):
            encoded_response = tokenizer.apply_chat_template([r for _, r in p_r],
                add_generation_prompt=False,
                return_tensors="pt",
                padding="longest",
                tokenizer_kwargs=dict(padding_side="right"),
                return_dict=False,
                return_assistant_tokens_mask=False,
                tokenize=True
            )

    ret = concat_output(output_list)
    print(timing)

def test2():
    timing = {}
    def concat_output(gen_output_list):
        STARTING_MSG = [{"role": "user", "content": "padding"}]
        def get_prompt_and_response(conv):
            if conv[0]["role"] == "system":
                prompts = conv[:2]
                responses = conv[2:]
            else:
                prompts = conv[:1]
                responses = conv[1:]
            assert len(responses), f"empty response for conv={conv}"
            return prompts,  STARTING_MSG + responses
        with _timer("1", timing):
            p_r = [get_prompt_and_response(conv) for out in gen_output_list for conv in out.conversations]
        with _timer("2", timing):
            encoded_prompt = tokenizer.apply_chat_template([p for p, _ in p_r],
                add_generation_prompt=True,
                return_tensors="pt",
                padding="longest",
                tokenizer_kwargs=dict(padding_side="left"),
                return_dict=False,
                tokenize=False
            )
        with _timer("3", timing):
            encoded_response = tokenizer.apply_chat_template([r for _, r in p_r],
                add_generation_prompt=False,
                return_tensors="pt",
                padding="longest",
                tokenizer_kwargs=dict(padding_side="right"),
                return_dict=False,
                return_assistant_tokens_mask=False,
                tokenize=False
            )
    ret = concat_output(output_list)
    print(timing)

def test3():
    timing = {}
    def concat_output(gen_output_list):
        STARTING_MSG = [{"role": "user", "content": "padding"}]
        def get_prompt_and_response(conv):
            if conv[0]["role"] == "system":
                prompts = conv[:2]
                responses = conv[2:]
            else:
                prompts = conv[:1]
                responses = conv[1:]
            assert len(responses), f"empty response for conv={conv}"
            return prompts,  STARTING_MSG + responses
        with _timer("1", timing):
            p_r = [get_prompt_and_response(conv) for out in gen_output_list for conv in out.conversations]
        with _timer("2", timing):
            encoded_prompt = tokenizer.apply_chat_template([p for p, _ in p_r],
                add_generation_prompt=True,
                return_dict=False,
                tokenize=False
            )
        print(encoded_prompt[0][-100:])
        print(len(encoded_prompt), max(len(p) for p in encoded_prompt))
        with _timer("2.1", timing):
            tokenizer(encoded_prompt,
                return_tensors="pt",
                padding="longest",padding_side="left")
    ret = concat_output(output_list)
    print(timing)

print("test3, chat template only ") 
test3()
print("test2, chat template only ") 
test2()
print("test1, tokenizer")
test1()
