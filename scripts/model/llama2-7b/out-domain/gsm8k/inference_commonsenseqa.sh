# deepspeed config
INCLUDE="localhost:1,2,3,4"

DEEPSPEED_CONFIG="--include $INCLUDE"
# model
BASE_PATH="/home/ylu130/workspace/in-context-generalization"
MODEL_NAME="llama2-7b"
MODEL_TYPE="llama"
MODEL_PATH="/scratch/ylu130/model/llama-2-7b"
# data
DATA_NAMES="commonsenseqa"
DATA_DIR="/scratch/ylu130/processed_data/commonsenseqa"
NUM_EVL=40
# generation
SAVE_PATH="${BASE_PATH}/results"
TEMPERATURE=1
BATCH_SIZE=10
OUT_DOMAIN_TASK_NAME="gsm8k"

OPTS=""
# model
OPTS+=" --model-name ${MODEL_NAME}"
OPTS+=" --model-type ${MODEL_TYPE}"
OPTS+=" --model-path ${MODEL_PATH}"
OPTS+=" --is-opensource"
# data
OPTS+=" --data-name ${DATA_NAMES}"
OPTS+=" --num-eval ${NUM_EVL}"
OPTS+=" --num-in-domain 0"
OPTS+=" --out-domain-data-name ${OUT_DOMAIN_TASK_NAME}"
# generation
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --do-sample"
OPTS+=" --top-k 50"
OPTS+=" --top-p 1"
OPTS+=" --temperature 1"
OPTS+=" --batch-size ${BATCH_SIZE}"

export NCCL_DEBUG=""
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8

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
            CMD="deepspeed ${DEEPSPEED_CONFIG} ${BASE_PATH}/inference.py ${OPTS_BACKUP} $@"
            echo ${CMD}
            ${CMD}
        done
    done
done