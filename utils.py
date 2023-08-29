from collections import defaultdict
from typing import Dict
import numpy as np
import os
import torch.distributed as dist
import torch.nn as nn
import random
import torch
from datetime import timedelta


def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)

def print_rank(*args, rank=0, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == rank:
        print(*args, **kwargs)


def all_gather(t, dim=0, world_size=None, group=None, op="cat"):
    if world_size is None:
        world_size = dist.get_world_size()
    all_t = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(all_t, t, group=group)
    if op == "cat":
        all_t = torch.cat(all_t, dim=dim)
    elif op == "stack":
        all_t = torch.stack(all_t, dim=dim)
    return all_t


def set_random_seed(seed):
    """Set random seed for reproducability."""
    seed = dist.get_rank() + seed
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def init_distributed(args):
    if args.is_slurm:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        args.local_rank = args.rank - args.gpus_per_node * (args.rank // args.gpus_per_node)
        args.word_size = int(os.environ["WORLD_SIZE"], "1")
    else:
        args.rank = int(os.getenv("RANK", "0"))             # this is the rank of the current GPU
        args.world_size = int(os.getenv("WORLD_SIZE", "1")) # this is the number of GPUs
        args.local_rank = int(os.getenv("LOCAL_RANK", "0")) # this is the rank of the current GPU within the node

    if args.rank == 0:
        print(f"using world size: {args.world_size}")

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)

    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size, timeout=timedelta(seconds=30))

def initialize(args):
    
    init_distributed(args)

    set_random_seed(args.seed)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)

    # finish the program if the following files exist
    if os.path.exists(os.path.join(args.save, "answers.jsonl")):
        print("Answers already exist, exiting...")
        exit()

