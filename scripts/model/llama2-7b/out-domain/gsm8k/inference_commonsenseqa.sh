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

# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
#                   --nnodes $NNODES \
#                   --node_rank $NODE_RANK \
#                   --master_addr $MASTER_ADDR \
#                   --master_port $MASTER_PORT"

# model
BASE_PATH="/home/ylu130/workspace/in-context-generalization"
MODEL_NAME="llama2-7b"
MODEL_TYPE="llama"
MODEL_PATH="/scratch/ylu130/model-hf"
MODEL_HF_NAME="meta-llama/Llama-2-7b-hf"
# data
DATA_NAMES="commonsenseqa"
DATA_DIR="/scratch/ylu130/processed_data/commonsenseqa"
NUM_EVL=1000
NUM_WORKERS=0
# generation
SAVE_PATH="${BASE_PATH}/results"
# hp
BATCH_SIZE=${4-4}
OUT_DOMAIN_TASK_NAME="gsm8k"

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
# export CUDA_VISIBLE_DEVICES=6,7,8,9

NUM_OUTDOMAIN_LIST=${1-"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"}
RATIONALE_LIST=${2-"True"}
MAX_PROMPT_LENGTH=${3-4096}

for SEED in 1 10 20 30
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
            CMD="python -m torch.distributed.launch --use-env ${DISTRIBUTED_ARGS} ${BASE_PATH}/inference.py ${OPTS_BACKUP} $@"
            echo ${CMD}
            ${CMD}
        done
    done
done