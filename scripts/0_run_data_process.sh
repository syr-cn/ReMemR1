# Step 1 Download the benchmarks from internet. Huggingface will do it automatically.
# https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets

# Step 2
PROJECT_ROOT=`pwd`
DATAROOT="${PROJECT_ROOT}/data/test"
mkdir -p $DATAROOT
cd taskutils/data_synthesis

export TOKENIZERS_PARALLELISM=false
data_sources="hotpotqa,2wikimultihopqa"
n_subset=128
python3 process_test.py --local_dir $DATAROOT --data_sources $data_sources --n_subset $n_subset