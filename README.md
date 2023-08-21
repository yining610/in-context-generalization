# in-context-generalization

## Requirements
```
bash install.sh
```

## Experiment Log
- [x] ```in-domain: CommonsenseQA / out-domain: GSM8K / seed: [1 10 20 30 40 50 60] / test: [4,-9] / both / max prompt token: 2048```
- [x] ```in-domain: CommonsenseQA / out-domain: GSM8K /seed: [1 10 20 30 40 50 60] / test: [-10,-25] / without rationale / max prompt token: 2048```
- [x] ```in-domain: GSM8K / out-domain: CommonsenseQA / seed: [1 10 20 30 40 50 60] / test: [-1,-9] / both / max prompt token: 2048 ```
- [x] ```in-domain: GSM8K / out-domain: CommonsenseQA / seed: [1 10 20 30 40 50 60] / test: [-4, -9] / both / max prompt token: 4096```
- [x] ```in-domain: CommonsenseQA / out-domain: None /  seed: [1 10 20 30 40 50 60] / test [5,9] / rationale / max prompt token: 2048```

**New Demonstration**

- [x] ```in-domain: CommonsenseQA / out-domain: GSM8K / seed: [1 10 20 30 40 50 60] / test: [-1,-9] / both / max prompt token: 2048 ```
- [x] ```in-domain: GSM8K / out-domain: CommonsenseQA / seed: [1 10 20 30 40 50 60] / test: [-1,-9] / both / max prompt token: 2048 ```
- [ ] ```in-domain: CommonsenseQA / out-domain: None / seed: [1 10 20 30 40 50 60] / test: [0, 9] / both / max prompt token: 2048 ```
- [ ] ```in-domain: CommonsenseQA / out-domain: GSM8K / seed: [1 10 20 30 40 50 60] / test: [-10, -25] / without rationale / max prompt token: 2048```