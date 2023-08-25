from data_utils.prompt_datasets import PromptDataset
from transformers import (
    GenerationConfig
    )

import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from utils import print_rank

torch.set_num_threads(4)


def run_model(args, tokenizer, model, dataset: PromptDataset):
    
    collate_fn = dataset.collate
        
    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_fn)
    model.eval()
    
    all_query_ids = []
    all_response_ids = []
    all_answer = []
    
    generation_config = GenerationConfig (
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_new_tokens=args.max_length,
        return_dict_in_generate=True,
        output_scores=True,
        num_beams=args.num_beams,
    )

    with torch.no_grad():
        for it, (prompt_ids, answer_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {args.data_name} ", disable=(dist.get_rank() != 0))): 
            if it == 0:
                print_rank("############### Example ###############")
                print_rank(tokenizer.decode(prompt_ids["input_ids"][0], skip_special_tokens=True))
                print_rank("############### End ###############")
                print_rank(f"Experiment Save Path: {args.save}")
            dataset.move_to_device(prompt_ids, torch.cuda.current_device())
            query_ids = prompt_ids["input_ids"]
            gen_out = model.generate(
                    **prompt_ids,
                    generation_config=generation_config
                )         
            full_ids = gen_out.sequences
            response_ids = full_ids[:, query_ids.size(1):] # remove prompt (may include start token)
            all_query_ids.extend(query_ids)
            all_response_ids.extend(response_ids)
            all_answer.extend(answer_batch)

    return (
        all_query_ids,
        all_response_ids,
        all_answer)

def inference_main(args, tokenizer, model, dataset: PromptDataset):
    start_time = time.time()

    query_ids, response_ids, answer_strs = run_model(args, tokenizer, model, dataset)
    query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
    response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    with open(os.path.join(args.save, "preds.txt"), "a") as f:
        for q, r in zip(query_strs, response_strs):
            f.write(q.replace("\n", "<n>") + "\t\t" + r.replace("\n", "<n>") + "\n")

    all_preds = [[]]
    for q, r in zip(query_strs, response_strs):
        all_preds[0].append((q, q + r))

    all_responses = []

    with open(os.path.join(args.save, "answers.jsonl"), "a") as f:    
        for p, a in zip(all_preds[0], answer_strs):
            q, r = p
            r = r[len(q):]
            idx = r.find("<|endoftext|>")
            if idx >= 0:
                r = r[:idx]
            f.write(json.dumps({
                "text": r.replace("<n>", "\n").strip(),
                "answer": a
            }) + "\n")
            all_responses.append(r.replace("<n>", "\n").strip())

    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])

    end_time = time.time()
    log_str = f"name: {args.data_name} | avg. gen lenth: {mean_gen_length} | time: {end_time - start_time}s"
    print_rank(log_str)
