BASE_PATH=${1-"/home/ylu130/workspace/in-context-generalization"}
DATA_PATH=${2-"/scratch/ylu130"}

for indomain in 0 1 2 3 4
do  
    for outdomain in 0 1 2 3 4
    do  
        echo "Processing commonsenseqa with ${indomain} in-domain examples and ${outdomain} GSM8K out-domain examples with rationales"
        PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/data_utils/process_data_commonsenseqa.py \
            --data-dir ${DATA_PATH}/data/commonsenseqa/ \
            --processed-data-dir ${DATA_PATH}/processed_data/commonsenseqa/ \
            --data-name "commonsenseqa" \
            --num-in-domain ${indomain} \
            --num-out-domain ${outdomain} \
            --out-domain-data-name "gsm8k" \
            --out-domain-data-dir ${DATA_PATH}/data/gsm8k/ \
            --seed 42 \
            --rationales

        if [ ${indomain} -ne 0 ] || [ ${outdomain} -ne 0 ]
        then
            echo "Processing commonsenseqa with ${indomain} in-domain examples and ${outdomain} GSM8K out-domain examples without rationales"
            PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/data_utils/process_data_commonsenseqa.py \
                --data-dir ${DATA_PATH}/data/commonsenseqa/ \
                --processed-data-dir ${DATA_PATH}/processed_data/commonsenseqa/ \
                --data-name "commonsenseqa" \
                --num-in-domain ${indomain} \
                --num-out-domain ${outdomain} \
                --out-domain-data-name "gsm8k" \
                --out-domain-data-dir ${DATA_PATH}/data/gsm8k/ \
                --seed 42
        fi

    done
done