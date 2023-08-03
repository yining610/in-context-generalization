import argparse
import os
import deepspeed

def add_model_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-name', type=str)
    group.add_argument("--n-gpu", type=int, default=1)
    group.add_argument("--n-nodes", type=int, default=1)
    group.add_argument("--is-opensource", action="store_true")
    return parser

def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument("--data-name", type=str)
    group.add_argument("--data-dir", type=str)
    group.add_argument("--cache-data-dir", type=str, default="/scratch/ylu130/datasets")
    return parser

def add_generation_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    group.add_argument('--temperature', type=float, default=0.7)
    group.add_argument('--max-tokens', type=int, default=1000)
    group.add_argument('--n', type=int, default=1)
    group.add_argument('--sample', type=int, default=0)
    group.add_argument('--batch', type=int)
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save generated results.')
    return parser

def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_data_args(parser)
    parser = add_generation_args(parser)
    parser = deepspeed.add_config_arguments(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        
    args.n_gpu = args.n_gpu * args.n_nodes
    
    save_path = os.path.join(
        args.save,
        (f"{args.model_name}"),
        (f"{args.data_name}"),
        (f"t{args.temperature}-n{args.n}-m{args.max_tokens}"),
    )
    args.save = save_path

    return args

