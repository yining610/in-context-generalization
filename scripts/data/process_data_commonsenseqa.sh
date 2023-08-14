BASE_PATH=${2-"/home/ylu130/workspace/in-context-generalization"}
DATA_PATH=${3-"/scratch/ylu130"}
OUT_DOMAIN_NAME=${4-"gsm8k"}

for seed in 1 10 20 30 40 50 60
do
    for indomain in 0
    do  
        for outdomain in 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
        do  
            # echo "Processing commonsenseqa with ${indomain} in-domain examples and ${outdomain} ${OUT_DOMAIN_NAME} out-domain examples with rationales"
            # PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/data_utils/process_data_commonsenseqa.py \
            #     --data-dir ${DATA_PATH}/data/commonsenseqa/ \
            #     --processed-data-dir ${DATA_PATH}/processed_data/commonsenseqa/ \
            #     --data-name "commonsenseqa" \
            #     --num-in-domain ${indomain} \
            #     --num-out-domain ${outdomain} \
            #     --out-domain-data-name ${OUT_DOMAIN_NAME} \
            #     --out-domain-data-dir ${DATA_PATH}/data/${OUT_DOMAIN_NAME}/ \
            #     --seed ${seed} \
            #     --rationales

            # i0-o0 has no difference in rationales
            if [ ${indomain} -ne 0 ] || [ ${outdomain} -ne 0 ]
            then
                echo "Processing commonsenseqa with ${indomain} in-domain examples and ${outdomain} ${OUT_DOMAIN_NAME} out-domain examples without rationales"
                PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/data_utils/process_data_commonsenseqa.py \
                    --data-dir ${DATA_PATH}/data/commonsenseqa/ \
                    --processed-data-dir ${DATA_PATH}/processed_data/commonsenseqa/ \
                    --data-name "commonsenseqa" \
                    --num-in-domain ${indomain} \
                    --num-out-domain ${outdomain} \
                    --out-domain-data-name ${OUT_DOMAIN_NAME} \
                    --out-domain-data-dir ${DATA_PATH}/data/${OUT_DOMAIN_NAME}/ \
                    --seed ${seed}
            fi
        done
    done
done