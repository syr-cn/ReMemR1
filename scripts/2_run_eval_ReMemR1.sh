export PROJECT_ROOT=`pwd`
export DATAROOT="$PROJECT_ROOT/data/test"
export REVERSED="0"

EXP_NAME="run_eval"
mkdir -p $PROJECT_ROOT/log/eval
python taskutils/memory_eval/run_eval.py 2>&1 | tee $PROJECT_ROOT/log/eval/$EXP_NAME.log