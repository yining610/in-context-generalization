#! /bin/bash
MASTER_ADDR=localhost
MASTER_PORT=${2-2113}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/ylu130/workspace/in-context-generalization"}
MODEL_NAME="llama2-7b"
MODEL_TYPE="llama"
MODEL_PATH="/scratch/ylu130/model/llama-2-7b"
# data
DATA_NAMES="commonsenseqa"
DATA_DIR="/scratch/ylu130/processed_data/commonsenseqa"
NUM_EVL=800
NUM_INDOMAIN=1
NUM_WORKERS=0
# generation
SAVE_PATH="${BASE_PATH}/results"
TEMPERATURE=1
# hp
BATCH_SIZE=8
SEED=42
RATIONAL="True"

OPTS=""
# model
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --model-type ${MODEL_TYPE}"
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --is-opensource"
# OPTS+=" --model-parallel"
# OPTS+=" --model-parallel-size ${GPUS_PER_NODE}"
# data
OPTS+=" --data-name ${DATA_NAMES}"
OPTS+=" --num-eval ${NUM_EVL}"
OPTS+=" --num-workers ${NUM_WORKERS}"
OPTS+=" --num-out-domain 0"
# generation
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --do-sample"
OPTS+=" --top-k 50"
OPTS+=" --top-p 1"
OPTS+=" --temperature 1"
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
export CUDA_VISIBLE_DEVICES=0,1,2,3


echo "PYTHONPATH=${PYTHONPATH}"

for RATIONAL in "True" "False"
do
    for NUM_INDOMAIN in 0 1 2 3 4
    do  
        OPTS_BACKUP=${OPTS}
        OPTS_BACKUP+=" --data-dir ${DATA_DIR}/i${NUM_INDOMAIN}-o0-s${SEED}-r${RATIONAL}"
        if [ ${RATIONAL} == "True" ]
        then
            OPTS_BACKUP+=" --rationales"
        fi
        OPTS_BACKUP+=" --num-in-domain ${NUM_INDOMAIN}"
        CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/inference.py ${OPTS_BACKUP} $@"
        echo ${CMD}
        ${CMD}
    done
done