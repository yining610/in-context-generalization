# from read_data import read_tellmewhy
from datasets import load_dataset
from models import llama
import argparse
import logging
import random

random.seed(42)

DATA_DIR = '/scratch/ylu130/datasets'


def main(args):
    data = load_dataset('StonyBrookNLP/tellmewhy', cache_dir=DATA_DIR)
    model = llama(temperature=args.temperature, max_tokens=args.max_tokens, n=args.n)

    # use validation data
    if args.sample > 0 and args.sample < len(data['validation']):
        data = data['validation'].shuffle(seed=42).select(range(args.sample))

    logging.info(f"Data Size: {len(data)}")

    for i in range(0, len(data), args.batch):
        narratives = data['narrative'][i:i+args.batch]
        questions = data['question'][i:i+args.batch]
        answers = data['answer'][i:i+args.batch]
        is_answerables = data['is_ques_answerable'][i:i+args.batch] 

        prompts = ['Context: ' + narratives[j] + '\nQuestion: ' + questions[j] + "\nLet's think step by step: " for j in range(len(narratives))]
        outputs = model(prompts)
        
        for k in range(len(outputs)):
            logging.info(f"prompt: {prompts[k]}")
            logging.info(f"output: {outputs[k]}")
            logging.info(f"gold answer: {answers[k]}")
            logging.info(f"is_answerable: {is_answerables[k]}")
            logging.info(f'-' * 50)

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
                        filename='./log/log.txt',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=args.logging_level
                        )

    logging.info("Running Urban Planning")
    main(args)