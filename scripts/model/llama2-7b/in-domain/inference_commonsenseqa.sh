#! /bin/bash
# MASTER_ADDR=localhost
# MASTER_PORT=29501
# NNODES=1
# NODE_RANK=0
# GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $SLURM_NTASKS_PER_NODE \
                  --nnodes $SLURM_NNODES \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH="/home/ylu130/workspace/in-context-generalization"
MODEL_NAME="llama2-7b"
MODEL_TYPE="llama"
MODEL_PATH="/scratch4/danielk/ylu130/model-hf"
MODEL_HF_NAME="meta-llama/Llama-2-7b-hf"
# data
DATA_NAMES="commonsenseqa"
DATA_DIR="/scratch4/danielk/ylu130/processed_data/commonsenseqa"
NUM_EVL=1000
NUM_WORKERS=0
# generation
SAVE_PATH="${BASE_PATH}/results"
# hp
BATCH_SIZE=${4-4}

OPTS=""
# model
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --model-type ${MODEL_TYPE}"
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --model-hf-name ${MODEL_HF_NAME}"
OPTS+=" --is-opensource"
OPTS+=" --is-slurm"
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

export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
# export CUDA_VISIBLE_DEVICES=2,3,4,5

NUM_INDOMAIN_LIST=${1-"0"}
RATIONALE_LIST=${2-"True False"}
MAX_PROMPT_LENGTH=${3-4096}

for SEED in 1 10 20 30
do 
    for RATIONALE in $RATIONALE_LIST
    do
        for NUM_INDOMAIN in $NUM_INDOMAIN_LIST
        do  
            if [ ${NUM_INDOMAIN} == 0 ] && [ ${RATIONALE} == "False" ]
            then
                continue
            fi
            OPTS_BACKUP=${OPTS}
            OPTS_BACKUP+=" --data-dir ${DATA_DIR}/in-domain/i${NUM_INDOMAIN}-s${SEED}-r${RATIONALE}"
            OPTS_BACKUP+=" --seed ${SEED}"
            OPTS_BACKUP+=" --max-prompt-length ${MAX_PROMPT_LENGTH}"
            if [ ${RATIONALE} == "True" ]
            then
                OPTS_BACKUP+=" --rationales"
            fi
            OPTS_BACKUP+=" --num-in-domain ${NUM_INDOMAIN}"
            CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/inference.py ${OPTS_BACKUP} $@"
            echo ${CMD}
            ${CMD}
        done
    done
done