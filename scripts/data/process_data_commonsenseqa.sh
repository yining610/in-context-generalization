BASE_PATH=${1-"/home/ylu130/workspace/in-context-generalization"}
DATA_PATH=${2-"/scratch/ylu130"}

for num in 0 1 2 3 4
do  
    echo "Processing commonsenseqa with ${num} in-domain examples with rationales"
    PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/data_utils/process_data_commonsenseqa.py \
        --data-dir ${DATA_PATH}/data/commonsenseqa/ \
        --processed-data-dir ${DATA_PATH}/processed_data/commonsenseqa/ \
        --data-name "commonsenseqa" \
        --num-in-domain ${num} \
        --seed 42 \
        --rationales
    
    echo "Processing commonsenseqa with ${num} in-domain examples without rationales"
    PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/data_utils/process_data_commonsenseqa.py \
        --data-dir ${DATA_PATH}/data/commonsenseqa/ \
        --processed-data-dir ${DATA_PATH}/processed_data/commonsenseqa/ \
        --data-name "commonsenseqa" \
        --num-in-domain ${num} \
        --seed 42
done