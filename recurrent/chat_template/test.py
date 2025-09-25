import pytest
from transformers import AutoTokenizer
from recurrent.chat_template.utils import set_chat_template, __registered_tokenizer__
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
# 全局测试数据
TEST_MESSAGES = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice."},
    {"role": "tool", "content": "I love fresh lemon juice."}
]

TEST_MESSAGES_NOT_FINISH = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice.", "finished": False},
]

@pytest.fixture(params=list(__registered_tokenizer__.keys()))
def tokenizers(request):
    model_name = request.param
    tokenizer_org = AutoTokenizer.from_pretrained(
        __registered_tokenizer__[model_name], 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        __registered_tokenizer__[model_name], 
        trust_remote_code=True
    )
    set_chat_template(tokenizer)
    
    return {
        'model_name': model_name,
        'tokenizer_org': tokenizer_org,
        'tokenizer': tokenizer
    }

def test_tokenizer_type(tokenizers):
    model_name = tokenizers['model_name']
    tokenizer_org = tokenizers['tokenizer_org']
    
    assert isinstance(tokenizer_org, globals()[model_name]), (
        f"tokenizer of model {__registered_tokenizer__[model_name]} is not {model_name}, "
        f"but {type(tokenizer_org)}"
    )

def test_template_file(tokenizers):
    model_name = tokenizers['model_name']
    tokenizer_org = tokenizers['tokenizer_org']
    from pathlib import Path
    template_path =  Path(__file__).parent / f"{model_name}_org.j2"
    # with open(template_path, 'w') as f:
    #     f.write(tokenizer_org.chat_template)
    with open(template_path, 'r') as f:
        assert tokenizer_org.chat_template == f.read(), (
            f"tokenizer template of model {__registered_tokenizer__[model_name]} "
            f"does not match {model_name}_org.j2"
        )

def test_message_string_consistency(tokenizers):
    # Qwen2 add an extra "\n" to the end of assistant message, which is not what we want for RL training
    tokenizer_org = tokenizers['tokenizer_org']
    tokenizer = tokenizers['tokenizer']
    
    msg_str_org = tokenizer_org.apply_chat_template(
        TEST_MESSAGES, return_dict=False, tokenize=False
    ).rstrip()
    
    msg_str = tokenizer.apply_chat_template(
        TEST_MESSAGES, return_dict=False, tokenize=False
    )
    
    assert msg_str == msg_str_org, f"\nOriginal: {repr(msg_str_org)}\nCustom: {repr(msg_str)}"

def test_token_consistency(tokenizers):
    # Qwen2 add an extra "\n" to the end of assistant message, which is not what we want for RL training
    tokenizer_org = tokenizers['tokenizer_org']
    tokenizer = tokenizers['tokenizer']
    
    tokens_org = tokenizer_org.apply_chat_template(
        TEST_MESSAGES, return_dict=True, return_assistant_tokens_mask=True, return_tensors='pt'
    )
    
    tokens = tokenizer.apply_chat_template(
        TEST_MESSAGES, return_dict=True, return_assistant_tokens_mask=True, return_tensors='pt'
    )
    
    assert (tokens.input_ids == tokens_org.input_ids[:,:-1]).all(), (
        f"Token mismatch:\nCustom: {tokens.input_ids}\nOriginal: {tokens_org.input_ids[:,:-1]}"
    )

def test_assistant_reply(tokenizers):
    tokenizer = tokenizers['tokenizer']
    
    tokens = tokenizer.apply_chat_template(
        TEST_MESSAGES, return_dict=True, return_assistant_tokens_mask=True, return_tensors='pt'
    )

    left = tokenizer.decode(tokens['input_ids'][tokens['assistant_masks'].bool()])
    right = TEST_MESSAGES[1]['content'] + tokenizer.eos_token
    
    assert left == right, f"\nDecoded: {repr(left)}\nExpected: {repr(right)}"    

def test_finish(tokenizers):
    tokenizer = tokenizers['tokenizer']
    tokens = tokenizer.apply_chat_template(
        TEST_MESSAGES_NOT_FINISH, return_dict=False, tokenize=False
    )
    assert not tokens.endswith(tokenizer.eos_token), tokens