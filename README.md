# in-context-generalization

## Requirements
```
bash install.sh
```

## Experiment Log
- [x] ```in-domain: CommonsenseQA / out-domain: GSM8K / seed: [1 10 20 30 40 50 60] / test: [-9,4] / both / max prompt token: 2048```
- [x] ```in-domain: CommonsenseQA / out-domain: GSM8K /seed: [1 10 20 30 40 50 60] / test: [-25,-10] / without rationale / max prompt token: 2048```
- [x] ```in-domain: GSM8K / out-domain: CommonsenseQA / seed: [1 10 20 30 40 50 60] / test: [-9,-1] / both / max prompt token: 2048 ```
- [x] ```in-domain: GSM8K / out-domain: CommonsenseQA / seed: [1 10 20 30 40 50 60] / test: [-9, -4] / both / max prompt token: 4096```
- [x] ```in-domain: CommonsenseQA / out-domain: None /  seed: [1 10 20 30 40 50 60] / test [5,9] / rationale / max prompt token: 2048```

----

#### LLama2-7b
- [x] ```in-domain: CommonsenseQA / out-domain: GSM8K / seed: [1 10 20 30 40 50 60] / test: [-9,-1] / both / max prompt token: 2048 ```
- [x] ```in-domain: GSM8K / out-domain: CommonsenseQA / seed: [1 10 20 30 40 50 60] / test: [-9,-1] / both / max prompt token: 2048 ```
- [x] ```in-domain: CommonsenseQA / out-domain: None / seed: [1 10 20 30 40 50 60] / test: [0, 9] / both / max prompt token: 2048 ```
- [x] ```in-domain: CommonsenseQA / out-domain: GSM8K / seed: [1 10 20 30 40 50 60] / test: [-25, -10] / without rationale / max prompt token: 2048```

#### GPT2-XL
- [ ] ```in-domain: CommonsenseQA / out-domain: GSM8K / seed: [1 10 20 30 40 50 60] / test: [-9, -1] / both / max prompt token: 2048```