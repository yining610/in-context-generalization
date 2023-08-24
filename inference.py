import sys
import time
import os

import torch
import torch.distributed as dist
import deepspeed

import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    mpu,
    ParallelOPTForCausalLM,
    ParallelLlamaForCausalLM,
    ParallelGPTJForCausalLM,
    ParallelGPT2LMHeadModel,)

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if args.model_type in ["gpt2", "opt", "llama", "gptj"]:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer



def setup_model(args):
    # get the model
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    model = deepspeed.init_inference(model=model,
                                     mp_size=args.model_parallel_size if args.model_parallel else 1,
                                     mpu=mpu if args.model_parallel else None,
                                     dtype=torch.float32,
                                     )
    
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

    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    
    # get the tokenizer
    if args.is_opensource:
        tokenizer = get_tokenizer(args)
        dataset = prepare_dataset_main(args, tokenizer)
    else:
        # TODO: prepare dataset for OpenAI Models
        pass

    model = setup_model(args)
    
    evaluate_main(args, tokenizer, model, dataset["test"], device)

if __name__ == "__main__":
    main()