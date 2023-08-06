import os
import json
import random
from args import get_args
from utils import print_args
import openai
import backoff 

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def generate_rationals(args, questions):
    rationals = []
    for question in questions:
        prompt = "Below is a question provided with the answer. Write rationales step by step to explain how you get the answer.\n" + question
        messages = [{"role": "user", "content": prompt}]
        res = completions_with_backoff(model="gpt-4", 
                                       messages=messages,
                                       temperature=args.temperature,
                                       max_tokens=args.max_length,
                                       )
        rationals.append(res["choices"][0]["message"]["content"])
    return rationals
        
    
def parse_commonsenseqa(line):
    question = line["question"]["stem"] + " Choices: "
    for choice in line["question"]["choices"]:
        question = question + " " + choice['label'] + ": " + choice["text"]
    gold_answer = line["answerKey"] + ": " + line["question"]["choices"][ord(line["answerKey"]) - ord("A")]["text"]
    question_with_answer = question + " The answer is " + gold_answer

    return question, gold_answer, question_with_answer

def main():
    args = get_args()
    print_args(args)

    args.processed_data_dir = os.path.join(args.processed_data_dir,
                                           (f"n{args.num_in_domain}-seed{args.seed}-rationales{args.rationales}"))
    os.makedirs(args.processed_data_dir, exist_ok=True)

    # load commonsensenqa data
    with open(os.path.join(args.data_dir, "train_rand_split.jsonl"), "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    template = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:{instruction}\n\n### Demonstration:{demonstration}\n\n### Input:{input}\n\n### Response:"
    )

    instruction = "Answer the following multiple choice question."
    
    json_file = open(os.path.join(args.processed_data_dir, f"{args.data_name}.jsonl"), "w")

    if args.num_in_domain > 0:
        random.seed(args.seed)
        indomain_examples = random.sample(data, args.num_in_domain)
        # exclude indomain_examples from data
        data = [d for d in data if d not in indomain_examples]

        indomain_questions_answer_pair = []
        indomain_questions = []
        indomain_answers = []
        for line in indomain_examples:
            question, gold_answer, question_with_answer = parse_commonsenseqa(line)
            indomain_questions_answer_pair.append(question_with_answer)
            indomain_questions.append(question)
            indomain_answers.append(gold_answer)
    
        if args.rationales:    
            indomain_rationals = generate_rationals(args, indomain_questions_answer_pair)
            indomain_demonstrations = "\n\n".join([f"Input: {q.strip()}\nRationales: {r.strip()}\nAnswer: {a.strip()}" for q, r, a in zip(indomain_questions, indomain_rationals, indomain_answers)])
        else:
            indomain_demonstrations = "\n\n".join([f"Input: {q.strip()}\nAnswer: {a.strip()}" for q, a in zip(indomain_questions, indomain_answers)])
    
    else:
        indomain_demonstrations = None

    for line in data:
        question, gold_answer, _ = parse_commonsenseqa(line)
        json_file.write(json.dumps({
            "prompt": template.format(instruction=instruction, 
                                      demonstration=indomain_demonstrations,
                                      input=question),
            "output": gold_answer,
        }) + "\n")
    
    json_file.close()
    print("Data num", len(data))

if __name__ == '__main__':
    main()