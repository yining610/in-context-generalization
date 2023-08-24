import sys
import time
import os

import torch
import torch.distributed as dist

import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer)

from args import get_args

from utils import initialize, print_args
from utils import print_rank
from utils import save_rank

from data_utils.prompt_datasets import PromptDataset
from inference_main import inference_main


torch.set_num_threads(4)


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def setup_model(args):
    # get the model
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

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
        dataset = PromptDataset(args, tokenizer, args.data_dir, args.num_eval)
    else:
        pass

    model = setup_model(args)
    
    inference_main(args, tokenizer, model, dataset, device)

if __name__ == "__main__":
    main()