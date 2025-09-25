from pathlib import Path
__registered_tokenizer__ = {
    "Qwen2TokenizerFast": "Qwen/Qwen2.5-0.5B-Instruct",
}
__template_dir = Path(__file__).parent

def set_chat_template(tokenizer):
    """
    Not used, since we may want to process reward and other things...
    For Qwen2TokenizerFast
    1. add {% generation %} block to support assistant_mask
    2. fix extra "\n" in the end of assistant's response
    3. if finish_reason is length and it is the final turn, then do not add <|im_end|>
    """
    name = type(tokenizer).__name__
    if name not in __registered_tokenizer__:
        raise ValueError(f"tokenizer {name} not registered")
    with open(__template_dir / f"{name}.j2" , 'r') as f:
        tokenizer.chat_template = f.read()
    return tokenizer