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
    )

from args import get_args

from utils import initialize, print_args

from data_utils.prompt_datasets import PromptDataset
from inference_main import inference_main


torch.set_num_threads(4)


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_hf_name, 
                                              cache_dir=args.model_path, 
                                              padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def setup_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_hf_name, 
                                                 cache_dir=args.model_path)
    
    model = deepspeed.init_inference(
        model,
        mp_size=args.world_size,
        dtype=torch.float16,
        replace_with_kernel_inject=True
    )

    model = model.module
    return model

def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    # get the tokenizer
    if args.is_opensource:
        tokenizer = get_tokenizer(args)
        dataset = PromptDataset(args, tokenizer, args.data_dir, args.num_eval)
    else:
        pass

    model = setup_model(args)
    
    inference_main(args, tokenizer, model, dataset)

if __name__ == "__main__":
    main()