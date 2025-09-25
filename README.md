# ReMemR1

This repo contains the official implementation of paper `Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents`.

## Installation

```bash
conda create -n rememr1 python=3.11
conda activate rememr1
pip install httpx==0.23.1 aiohttp -U ray[serve,default] vllm

pip install nltk pyyaml beautifulsoup4 html2text wonderwords tenacity fire
pip install vllm==0.9 --index-url https://download.pytorch.org/whl/cu126
pip install "sglang==0.4.6"
pip install hydra-core accelerate tensordict torchdata wandb "tensordict<=0.6.2"
```

## Data Processing

**Trianing Data:** 
This research use the same data as [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent).
The training data is publicly available, and can be downloaded from [huggingface](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main).
After the download is finished, put `hotpotqa_train_32k.parquet` and `hotpotqa_dev.parquet` under `data/train/`.

**Data for Evaluation:** 
The data for evaluation is sourced from HotpotQA and 2WikiMultiHopQA.
To process the data for long-context QA, simply run:
```bash
bash scripts/0_run_data_process.sh
```

## Training

To start multi-node training, you should first ray servers on the head and worker nodes.
```bash
# head node
ray start --head --dashboard-host=0.0.0.0
```

```bash
# worker node
ray start --address=<head_node_address>
```
More references can be found in ray's documentation [here](https://verl.readthedocs.io/en/latest/start/multinode.html).

After all the nodes are ready, the training can be launched via:
```bash
bash scripts/1_run_train_ReMemR1_3B.sh
bash scripts/1_run_train_ReMemR1_7B.sh
```

## Evaluation

Once the training is converged, use `scripts/merge_ckpt.sh` to merge the checkpoints before evaluation.
For example,
```bash
bash scripts/merge_ckpt.sh "results/memory_agent/ReMemR1_3B/global_step_200/actor"
```
This will automatically put the merged checkpoint into `results/memory_agent/ReMemR1_3B/global_step_200/actor/hf_ckpt`.

Once the checkpoint merging is done, run the below script to run evaluation:
```bash
bash scripts/2_run_eval_ReMemR1.sh
```

## Licensing

This project is licensed under the MIT License.
It includes components from [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent), licensed under the Apache License 2.0. Thanks for their awesome work!