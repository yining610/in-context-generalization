BASE_PATH=${1-"/home/ylu130/workspace/in-context-generalization"}
DATA_PATH=${2-"/scratch/ylu130"}

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/data_utils/process_data_commonsenseqa.py \
    --data-dir ${DATA_PATH}/data/commonsenseqa/ \
    --processed-data-dir ${DATA_PATH}/processed_data/commonsenseqa/ \
    --data-name "commonsenseqa" \
    --num-in-domain 3 \
    --seed 42 \
    --provide-rationals

