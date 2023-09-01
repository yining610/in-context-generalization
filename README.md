# in-context-generalization

## Requirements
```
bash install.sh
```

## Running
Prepare Datasets
```
bash ./scripts/data/process_data_[DATASETNAME].sh
```

Inference (e.g., in-domain test)
```
bash ./scripts/model/[MODELNAME]/in-domain/inference_[DATASETNAME].sh [NUM_INDOMAIN_LIST] [RATIONALE_LIST] [MAX_PROMPT_LENGTH]
```

## Experiment Log
#### LLama2-7b
- [x] ```in-domain: CommonsenseQA / out-domain: GSM8K / seed: [1 10 20 30 40 50 60] / test: [-9,-1] / both / max prompt token: 2048 ```
- [x] ```in-domain: GSM8K / out-domain: CommonsenseQA / seed: [1 10 20 30 40 50 60] / test: [-9,-1] / both / max prompt token: 2048 ```
- [x] ```in-domain: CommonsenseQA / out-domain: None / seed: [1 10 20 30 40 50 60] / test: [0, 9] / both / max prompt token: 2048 ```
- [x] ```in-domain: CommonsenseQA / out-domain: GSM8K / seed: [1 10 20 30 40 50 60] / test: [-25, -10] / without rationale / max prompt token: 2048```


- [x] ```in-domain: CommonsenseQA / out-domain: GSM8K / seed: [1 10 20 30] / test: [-25,-1] / Without rationale / max prompt token: 4096 ```
- [x] ```in-domain: CommonsenseQA / out-domain: None / seed: [1 10 20 30] / test: [0,9] / both / max prompt token: 4096 ```
- [x] ```in-domain: GSM8K / out-domain: CommonsenseQA / seed: [1 10 20 30] / test: [-8, -4, -2, -1] / Without rationale / max prompt token: 4096 ```
- [x] ```in-domain: CommonsenseQA / out-domain: GSM8K / seed: [1 10 20 30] / test: [-10,-1] / With rationale / max prompt token: 4096 ```

#### LLama2-13b
- [ ] ```in-domain: CommonsenseQA / out-domain: GSM8K / seed: [1 10 20 30] / test: [-25,-1] / Without rationale / max prompt token: 4096 ```