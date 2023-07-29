# from read_data import read_tellmewhy
from datasets import load_dataset
from models import lm
from functools import partial
import argparse
import logging
import random 

random.seed(42)

DATA_DIR = '/scratch/ylu130/datasets'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003', 'llama-7b'], default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=1000)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--stop', type=str, default=None)
    parser.add_argument('--sample', type=int, default=0)

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.INFO,
        # default=logging.INFO,
        dest="logging_level",
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    return args

def main(args):
    data = load_dataset('StonyBrookNLP/tellmewhy', cache_dir=DATA_DIR)

    # combine the narratives and questions to form the prompts
    prompts = ["Context: " + data["validation"]['narrative'][i] + ' Question: ' + data['validation']['question'][i] for i in range(len(data['validation']))]

    lm = partial(lm, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens, n=args.n, stop=args.stop)

    if args.sample > 0 and args.sample < len(prompts):
        prompts = random.sample(prompts, args.sample)
        
    outputs = lm(prompts)

    logging.info(outputs)

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
                        filename='./log/log.txt',
                        # level=args.logging_level
                        )
    main(args)