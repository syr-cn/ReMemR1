## Setup
### Installation

```bash
pip install httpx==0.23.1 aiohttp -U ray[serve,default] vllm

pip install nltk pyyaml beautifulsoup4 html2text wonderwords tenacity fire
pip install vllm==0.9 --index-url https://download.pytorch.org/whl/cu126
pip install "sglang==0.4.6"
pip install hydra-core accelerate tensordict torchdata wandb "tensordict<=0.6.2"
```


## Licensing

This project is licensed under the MIT License.
It includes components from [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent), licensed under the Apache License 2.0. Thanks for their awesome work!