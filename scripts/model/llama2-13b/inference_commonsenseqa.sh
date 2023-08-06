#! /bin/bash
MASTER_ADDR=localhost
MASTER_PORT=${2-2113}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-3}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/ylu130/workspace/in-context-generalization"}
MODEL_NAME="llama2-13b"
MODEL_PATH="/scratch/ylu130/model/llama-2-13b"
# data
DATA_NAMES="commonsenseqa"
DATA_DIR="/scratch/ylu130/processed_data/commonsenseqa"
NUM_EVL=5
NUM_INDOMAIN=1
NUM_WORKERS=0
# generation
SAVE_PATH="${BASE_PATH}/results"
# hp
BATCH_SIZE=16
SEED=42
RATIONAL="True"

OPTS=""
# model
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --is-opensource"
# data
OPTS+=" --data-dir ${DATA_DIR}/n${NUM_INDOMAIN}-seed${SEED}-rationales${RATIONAL}"
OPTS+=" --data-name ${DATA_NAMES}"
OPTS+=" --num-eval ${NUM_EVL}"
OPTS+=" --num-in-domain ${NUM_INDOMAIN}"
OPTS+=" --num-workers ${NUM_WORKERS}"
# generation
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# hp
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --seed ${SEED}"

export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/inference.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
${CMD}
