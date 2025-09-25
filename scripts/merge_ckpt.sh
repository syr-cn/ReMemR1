MODEL_ROOT=$1
if [ -z "$MODEL_ROOT" ]; then
    echo "MODEL_ROOT is not set"
    exit 1
fi
if [ ! -d "$MODEL_ROOT" ]; then
    echo "Model directory $MODEL_ROOT does not exist."
    exit 1
fi

python scripts/model_merger.py \
  --backend fsdp \
  --hf_model_path $MODEL_ROOT/huggingface \
  --local_dir $MODEL_ROOT \
  --target_dir $MODEL_ROOT/hf_ckpt

# Copy the tokenizer configs to the merged checkpoint
cp $MODEL_ROOT/huggingface/* $MODEL_ROOT/hf_ckpt/
