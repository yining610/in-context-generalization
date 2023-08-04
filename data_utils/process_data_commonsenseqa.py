import os
import json
import random
from args import get_args
import openai
import backoff 

completion_tokens = prompt_tokens = 0

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def generate_rationals(args, questions):
    rationals = []
    for question in questions:
        prompt = "Below is a question provided with the answer. Write rationales step by step to explain how you get the answer.\n" + question
        messages = [{"role": "user", "content": prompt}]
        res = completions_with_backoff(model="gpt-4", 
                                       messages=messages,
                                       temperature=args.temperature,
                                       max_tokens=args.max_tokens,
                                       n=args.n,)

    
def completiongpt(prompt, model, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    res = completions_with_backoff(model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    outputs.extend([choice["text"] for choice in res["choices"]])
    # log completion tokens
    completion_tokens += res["usage"]["completion_tokens"]
    prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs
    
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

    random.seed(args.seed)
    indomain_examples = random.sample(data, args.num_in_domain)
    indomain_demonstrations = [d["question"]["stem"] + " " + \
                               "\n".join(d["question"]["choices"]) + \
                               "\nThe answer is " + d["answerKey"] for d in indomain_examples]
    
    if args.provide_rationals:    
        indomain_rationals = generate_rationals(args, indomain_demonstrations)

    
    for line in data:
        question_with_choices = line["question"]["stem"] + " " + "\n".join(line["question"]["choices"])
        gold_answer = line["answerKey"]
        
        

        


if __name__ == '__main__':
    main()