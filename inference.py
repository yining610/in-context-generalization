import sys
import time
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    )

from args import get_args

from utils import initialize, print_args
from utils import print_rank

from inference_main import inference_main, prepare_dataset_main


torch.set_num_threads(4)


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_hf_name, cache_dir=args.model_path)
    if args.model_type in ["gpt2", "opt", "llama", "gptj"]:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    return tokenizer

def setup_model(args, device):
    # get the model
    model = AutoModelForCausalLM.from_pretrained(args.model_hf_name, cache_dir=args.model_path,
                                                 device_map={"": device}, torch_dtype=torch.float16)
    if dist.get_rank() == 0:
        print(' > number of parameters: {}'.format(
            sum([p.nelement() for p in model.parameters()])), flush=True)
    model = DDP(model)
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
   
    # get the tokenizer
    if args.is_opensource:
        tokenizer = get_tokenizer(args)
        dataset = prepare_dataset_main(args, tokenizer)
    else:
        # TODO: prepare dataset for OpenAI Models
        pass

    model = setup_model(args, device)
    
    inference_main(args, tokenizer, model, dataset, device)

if __name__ == "__main__":
    main()