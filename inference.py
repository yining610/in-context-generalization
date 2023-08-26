import sys
import time
import os

import torch
import torch.distributed as dist

import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    )

parallel_model_map = {
    "opt": ParallelOPTForCausalLM,
    "gptj": ParallelGPTJForCausalLM,
    "gpt2": ParallelGPT2LMHeadModel,
    "llama": ParallelLlamaForCausalLM
}

from args import get_args

from utils import initialize, print_args
from utils import print_rank
from utils import save_rank
from utils import load_parallel, save_parallel


from inference_main import evaluate_main, prepare_dataset_main


torch.set_num_threads(4)


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.model_type in ["gpt2", "opt", "llama", "gptj"]:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_model(args, device):
    if args.model_parallel:
        config = AutoConfig.from_pretrained(args.model_path)
        config.is_model_parallel = True
        model = parallel_model_map[args.model_type](config).half()
        # save_parallel(model, args.model_path)
        load_parallel(model, args.model_path)
        model.eval()

        if mpu.get_data_parallel_rank() == 0:
            print(' > number of parameters on model parallel rank {}: {}'.format(
                mpu.get_model_parallel_rank(),
                sum([p.nelement() for p in model.parameters()])), flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)

    return model


def setup_model_and_optimizer(args, device):
    # get the model
    model = get_model(args, device)
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model

def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    # get the tokenizer
    if args.is_opensource:
        tokenizer = get_tokenizer(args)
        dataset = prepare_dataset_main(args, tokenizer)
    else:
        # TODO: prepare dataset for OpenAI Models
        pass

    model = setup_model_and_optimizer(args, device)
    
    evaluate_main(args, tokenizer, model, dataset["test"], device)

if __name__ == "__main__":
    main()