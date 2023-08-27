#! /bin/bash
MASTER_ADDR=localhost
MASTER_PORT=2113
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH="/home/ylu130/workspace/in-context-generalization"
MODEL_NAME="llama2-13b"
MODEL_TYPE="llama"
MODEL_PATH="/scratch/ylu130/model-hf"
MODEL_HF_NAME="meta-llama/Llama-2-13b-hf"
# data
DATA_NAMES="nq"
DATA_DIR="/scratch/ylu130/data/adam"
NUM_WORKERS=0
# generation
SAVE_PATH="${BASE_PATH}/results"
TEMPERATURE=1
# hp
BATCH_SIZE=2

OPTS=""
# model
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --model-type ${MODEL_TYPE}"
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --model-hf-name ${MODEL_HF_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --is-opensource"
# OPTS+=" --model-parallel"
# OPTS+=" --model-parallel-size ${GPUS_PER_NODE}"
# data
OPTS+=" --data-name ${DATA_NAMES}"
OPTS+=" --num-workers ${NUM_WORKERS}"
OPTS+=" --num-out-domain 0"
# generation
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --do-sample"
OPTS+=" --top-k 50"
OPTS+=" --top-p 1"
OPTS+=" --temperature 1"
OPTS+=" --batch-size ${BATCH_SIZE}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
export CUDA_VISIBLE_DEVICES=2,3,4,5

INDEX=${1-"0 4 9"}
MAX_PROMPT_LENGTH=4096

for NUM in $INDEX
do  
    OPTS_BACKUP=${OPTS}
    OPTS_BACKUP+=" --data-dir ${DATA_DIR}/nq-test-10_total_documents_gold_at_${NUM}-llama-predictions-scored.jsonl"
    OPTS_BACKUP+=" --num-in-domain ${NUM}"
    OPTS_BACKUP+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"
    CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/inference.py ${OPTS_BACKUP} $@"
    echo ${CMD}
    ${CMD}
done