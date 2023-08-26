import argparse
import os

def add_model_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument('--model-name', type=str)
    group.add_argument("--model-type", type=str)
    group.add_argument("--model-path", type=str, default="/scratch/ylu130/model")
    group.add_argument("--model-hf-name", type=str, default=None)
    group.add_argument("--is-opensource", action="store_true")
    group.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    
    return parser

def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('data', 'data configurations')
    group.add_argument("--data-name", type=str)
    group.add_argument("--data-dir", type=str)
    group.add_argument("--processed-data-dir", type=str)
    group.add_argument("--num-eval", type=int)
    group.add_argument("--num-in-domain", type=int)
    group.add_argument("--num-out-domain", type=int)
    group.add_argument("--out-domain-data-name", type=str, default=None)
    group.add_argument("--out-domain-data-dir", type=str, default=None)
    return parser

def add_generation_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('generation', 'generation configurations')
    group.add_argument('--max-prompt-length', type=int, default=2048)
    group.add_argument('--max-length', type=int, default=512)
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save generated results.')
    group.add_argument("--rationales", action="store_true")
    group.add_argument("--top-k", type=int, default=50)
    group.add_argument("--top-p", type=float, default=1.0)
    group.add_argument("--do-sample", action="store_true")
    group.add_argument("--num-beams", type=int, default=1)
    group.add_argument("--temperature", type=float, default=1)
    group.add_argument("--no-repeat-ngram-size", type=int, default=6)
    return parser

def add_hp_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("hp", "hyper parameter configurations")
    group.add_argument('--batch-size', type=int, default=32,
                       help='Data Loader batch size')
    group.add_argument("--seed", type=int, default=42)
    
    return parser

def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_data_args(parser)
    parser = add_generation_args(parser)
    parser = add_hp_args(parser)
    
    args, unknown = parser.parse_known_args()
    
    assert all(["--" not in x for x in unknown]), unknown
    
    if args.model_name is not None:
        if args.num_out_domain == 0:
            save_path = os.path.join(
                args.save,
                (f"{args.model_name}"),
                (f"{args.data_name}"),
                "in-domain",
                (f"i{args.num_in_domain}-s{args.seed}-r{args.rationales}"),
            )
        elif args.num_in_domain == 0:
            save_path = os.path.join(
                args.save,
                (f"{args.model_name}"),
                (f"{args.data_name}"),
                "out-domain",
                (f"o{args.num_out_domain}-t{args.out_domain_data_name}-s{args.seed}-r{args.rationales}"),
            )
        else:
            save_path = os.path.join(
                args.save,
                (f"{args.model_name}"),
                (f"{args.data_name}"),
                "mixed-domain",
                (f"i{args.num_in_domain}-o{args.num_out_domain}-t{args.out_domain_data_name}-s{args.seed}-r{args.rationales}"),
            )
        
        if args.max_prompt_length != 2048:
            save_path += f"-m{args.max_prompt_length}"

        args.save = save_path

    return args

