# [ICLR 2026] Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents

This repo contains the official implementation of ICLR 2026 paper **ReMemR1**: `Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents`.

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-2509.23040-b31b1b.svg)](https://arxiv.org/abs/2509.23040)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://github.com/syr-cn/ReMemR1)
[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm-dark.svg)](https://huggingface.co/papers/2509.23040)

## News
- [Jan 2026] Our paper get accepted by [ICLR 2026](https://openreview.net/forum?id=1cymflI2Lh) 🎉🎉🎉
- [Jan 2026] Nvidia propose [GDPO](https://arxiv.org/abs/2601.05242), which shares the same design logic as our multi-level reward aggregation
- [Sep 2025] Our paper is released on [Huggingface](https://huggingface.co/papers/2509.23040) and [Alphaxiv](https://www.alphaxiv.org/abs/2509.23040). Please upvote our paper if you like this work :)

## Overview

- **Conceptual Example:** wheat gleaning in the field

<p align="center">
  <img width="600" alt="image" src="./conceptual_example.png" />
</p>

- Q1: How we address the constraints in linear doc scan?
- A1: We introduce **Callback Mechanism** to allow non-linear memory re-visit.

<p align="center">
  <img width="600" alt="image" src="./teaser.png" />
</p>

<p align="center">
  <img width="600" alt="image" src="./framework.png" />
</p>

- Q2: How we precisely reward callback/update behaviors?
- A2: We introduce Multi-Level Rewarding, aggregated at advantage level.

<p align="center">
  <img width="600" alt="image" src="./reward_design.png" />
</p>


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
This research use the same training data as [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent).
The data files are publicly available, and can be downloaded from [huggingface](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main).
After the download is finished, put `hotpotqa_train_32k.parquet` and `hotpotqa_dev.parquet` under `data/train/`.

**Data for Evaluation:** 
The data for evaluation is sourced from HotpotQA and 2WikiMultiHopQA.
To process the data for long-context QA, simply run:
```bash
bash scripts/0_run_data_process.sh
```

## Training

### Prepare for Multi-Node Training

You may skip this step if you only want to start a single-node trianing.

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

### Start the Training

After all the nodes are ready (or you prefer single-node training), the training can be launched via:
```bash
bash scripts/1_run_train_ReMemR1_3B.sh
bash scripts/1_run_train_ReMemR1_7B.sh
```

You might adjust the `N_NODE` variable to match your number of devices.

## Evaluation

Once the training is converged, use `scripts/merge_ckpt.sh` to merge the checkpoints before evaluation.
For example,
```bash
bash scripts/merge_ckpt.sh "results/memory_agent/ReMemR1_3B/global_step_200/actor"
```
This will automatically put the merged checkpoint into `results/memory_agent/ReMemR1_3B/global_step_200/actor/hf_ckpt`.

Once the checkpoint merging is done, run the below script for evaluation:
```bash
bash scripts/2_run_eval_ReMemR1.sh
```

## Licensing

This project is licensed under the MIT License.
It includes components from [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent), licensed under the Apache License 2.0. Thanks for their awesome work!

## Citation

```latex
@article{rememr1,
  author       = {Yaorui Shi and
                  Yuxin Chen and
                  Siyuan Wang and
                  Sihang Li and
                  Hengxing Cai and
                  Qi Gu and
                  Xiang Wang and
                  An Zhang},
  title        = {Look Back to Reason Forward: Revisitable Memory for Long-Context {LLM}
                  Agents},
  journal      = {CoRR},
  volume       = {abs/2509.23040},
  year         = {2025},
  eprinttype    = {arXiv},
  eprint       = {2509.23040},
}
```
