# from read_data import read_tellmewhy
from datasets import load_dataset
from models import lm, gpt_usage
from functools import partial
import argparse
import logging
import random

random.seed(42)

DATA_DIR = '/scratch/ylu130/datasets'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003', 'llama-13b'], default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=1000)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--stop', type=str, default=None)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--batch', type=int, default=4)

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
    global lm
    data = load_dataset('StonyBrookNLP/tellmewhy', cache_dir=DATA_DIR)

    lm = partial(lm, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens, n=args.n, stop=args.stop)

    # use validation data
    if args.sample > 0 and args.sample < len(data['validation']):
        data = data['validation'].shuffle(seed=42).select(range(args.sample))

    logging.info(f"Data Size: {len(data)}")

    for i in range(0, len(data), args.batch):
        narratives = data['narrative'][i:i+args.batch]
        questions = data['question'][i:i+args.batch]
        answers = data['answer'][i:i+args.batch]
        is_answerables = data['is_ques_answerable'][i:i+args.batch] 

        prompts = ['Context: ' + narratives[j] + '\nQuestion: ' + questions[j] for j in range(len(narratives))]
        outputs = lm(prompts)
        
        for k in range(len(outputs)):
            logging.info(f"prompt: {prompts[k]}")
            logging.info(f"output: {outputs[k]}")
            logging.info(f"gold answer: {answers[k]}")
            logging.info(f"is_answerable: {is_answerables[k]}")
            logging.info(f'-' * 50)
        
    logging.info(f"usage: {gpt_usage(args.model)}")

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
                        filename='./log/log.txt',
                        # level=args.logging_level
                        )
    main(args)