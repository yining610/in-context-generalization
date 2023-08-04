import os
import json
import random


def generate_rationals(args):


def main():
    args = get_args()
        
    args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_name)
    os.makedirs(args.processed_data_dir, exist_ok=True)

    # load commonsensenqa data
    with open(os.path.join(args.data_dir, "train_rand_split.jsonl"), "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    template = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:{instruction}\n\n### Demonstration:{demonstration}\n]n### Input:{input}\n\n### Response:\n"
    )
    instruction = "Answer the following multiple choice question."
    
    json_file = open(os.path.join(args.processed_data_dir, f"{args.data_name}.jsonl"), "w")

    # randomly select k items from list data
    demonstration_lines = random.sample(data, )
    
    for line in data:
        question_with_choices = line["question"]["stem"] + " " + "\n".join(line["question"]["choices"])
        
        

        


if __name__ == '__main__':
    main()