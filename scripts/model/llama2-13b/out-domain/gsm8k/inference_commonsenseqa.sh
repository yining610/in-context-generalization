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
DATA_NAMES="commonsenseqa"
DATA_DIR="/scratch/ylu130/processed_data/commonsenseqa"
NUM_EVL=20
NUM_WORKERS=0
# generation
SAVE_PATH="${BASE_PATH}/results"
TEMPERATURE=1
# hp
BATCH_SIZE=5
OUT_DOMAIN_TASK_NAME="gsm8k"

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
OPTS+=" --batch-size ${BATCH_SIZE}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
export CUDA_VISIBLE_DEVICES=2,3,4,5

NUM_OUTDOMAIN_LIST=${1-"21"}
RATIONALE_LIST=${2-"False"}
MAX_PROMPT_LENGTH=${3-2048}

for SEED in 1 10 20 30 40 50 60
do 
    for RATIONALE in $RATIONALE_LIST
    do
        for NUM_OUTDOMAIN in $NUM_OUTDOMAIN_LIST
        do  
            OPTS_BACKUP=${OPTS}
            OPTS_BACKUP+=" --data-dir ${DATA_DIR}/out-domain/o${NUM_OUTDOMAIN}-t${OUT_DOMAIN_TASK_NAME}-s${SEED}-r${RATIONALE}"
            OPTS_BACKUP+=" --seed ${SEED}"
            OPTS_BACKUP+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"
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