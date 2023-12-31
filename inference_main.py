from data_utils.prompt_datasets import PromptDataset
from transformers import GenerationConfig

import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import numpy as np
import json
from utils import print_rank


torch.set_num_threads(4)


def prepare_dataset_main(args, tokenizer):
    return PromptDataset(args, tokenizer, args.data_dir, args.num_eval)

def run_model(args, tokenizer, model, dataset: PromptDataset, device):
    
    collate_fn = dataset.collate
    
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, shuffle=False, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    model.eval()
    
    all_query_ids = []
    all_response_ids = []
    all_answer = []
    all_indices = []
    
    generation_config = GenerationConfig (
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        num_beams=args.num_beams,
    )

    with torch.no_grad():
        for it, (indices, prompt_ids, answer_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {args.data_name} ", disable=(dist.get_rank() != 0))): 
            if it == 0:
                print_rank("############### Example ###############")
                print_rank(tokenizer.decode(prompt_ids["input_ids"][0], skip_special_tokens=True))
                print_rank("############### End ###############")
                print_rank(f"Experiment Save Path: {args.save}")
            dataset.move_to_device(prompt_ids, device)
            query_ids = prompt_ids["input_ids"]
            gen_out = model.generate(
                    **prompt_ids,
                    generation_config=generation_config
                )         
            full_ids = gen_out.sequences
            response_ids = full_ids[:, query_ids.size(1):] # remove prompt (may include start token)
            all_query_ids.extend(query_ids)
            all_response_ids.extend(response_ids)
            all_indices.extend(indices)
            if isinstance(answer_batch[0], list):
                for a in answer_batch:
                    all_answer.append(a)
            else:
                all_answer.extend(answer_batch)

    return (
        all_query_ids,
        all_response_ids,
        all_answer,
        all_indices)


def inference_main(args, tokenizer, model, dataset: PromptDataset, device):
    start_time = time.time()

    query_ids, response_ids, answers, indices = run_model(args, tokenizer, model, dataset, device)
    query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
    response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    with open(os.path.join(args.save, "preds.txt"), "a") as f:
        for q, r in zip(query_strs, response_strs):
            f.write(q.replace("\n", "<n>") + "\t\t" + r.replace("\n", "<n>") + "\n")

    all_preds = [[]]
    for q, r in zip(query_strs, response_strs):
        all_preds[0].append((q, q + r))

    with open(os.path.join(args.save, "answers.jsonl"), "a") as f:    
        for p, a, i in zip(all_preds[0], answers, indices):
            q, r = p
            r = r[len(q):]
            idx = r.find("<|endoftext|>")
            if idx >= 0:
                r = r[:idx]
            f.write(json.dumps({
                "idx": i+1,
                "text": r.replace("<n>", "\n").strip(),
                "answer": a
            }) + "\n")


    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])

    end_time = time.time()
    log_str = f"name: {args.data_name} | avg. gen lenth: {mean_gen_length} | time: {end_time - start_time}s"
    print_rank(log_str)