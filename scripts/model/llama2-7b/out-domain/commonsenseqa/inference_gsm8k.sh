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
DATA_NAMES="gsm8k"
DATA_DIR="/scratch/ylu130/processed_data/gsm8k"
NUM_EVL=1000
NUM_WORKERS=0
# generation
SAVE_PATH="${BASE_PATH}/results"
TEMPERATURE=1
# hp
BATCH_SIZE=5
OUT_DOMAIN_TASK_NAME="commonsenseqa"

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
OPTS+=" --num-in-domain 0"
OPTS+=" --out-domain-data-name ${OUT_DOMAIN_TASK_NAME}"
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

export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
export CUDA_VISIBLE_DEVICES=2,3,4,5

echo "PYTHONPATH=${PYTHONPATH}"

for SEED in 1 10 20 30 40 50 60
do 
    for RATIONALE in "True" "False"
    do
        for NUM_OUTDOMAIN in 1 2 3 4 5 6 7 8 9
        do  
            OPTS_BACKUP=${OPTS}
            OPTS_BACKUP+=" --data-dir ${DATA_DIR}/out-domain/o${NUM_OUTDOMAIN}-t${OUT_DOMAIN_TASK_NAME}-s${SEED}-r${RATIONALE}"
            OPTS_BACKUP+=" --seed ${SEED}"
            if [ ${RATIONALE} == "True" ]
            then
                OPTS_BACKUP+=" --rationales"
            fi
            OPTS_BACKUP+=" --num-out-domain ${NUM_OUTDOMAIN}"
            CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/inference.py ${OPTS_BACKUP} $@"
            echo ${CMD}
            ${CMD}
        done
    done
done