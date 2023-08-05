import random
import torch
import os
from torch.utils.data import Dataset

from torch.distributed import get_rank
from utils import print_rank
from tqdm import tqdm
import json


class PromptDataset(Dataset):
    def __init__(self, args, tokenizer, data_path, num=-1):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.pad_id = self.tokenizer.eos_token_id
        self.max_prompt_length = args.max_prompt_length

        self.data, _ = self.load_data_json(data_path)
        
        if os.path.exists(os.path.join(data_path, f"{self.data_name}.jsonl")):
            with open(os.path.join(data_path, f"{self.data_name}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.raw]
        else:
            print_rank("WARNING: No answers exist")
        
        if num > 0:
            self.data = self.data[:num]
            self.answers = self.answers[:num]
            
        self.label_map = {tokenizer.encode(x[0], add_special_tokens=False)[0]: x[0] for x in self.answers}
            
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
        print_rank(f"Num instances: {len(self.data)}")
            
    def __len__(self):
        return self.num

    def load_data_json(self, data_path):
        if os.path.exists(os.path.join(data_path, f"{self.data_name}.jsonl")):
            data_path = os.path.join(data_path, f"{self.data_name}.jsonl")
        else:
            print_rank(f"WARNING: {os.path.join(data_path, f'{self.data_name}.jsonl')} does not exist")

        with open(data_path) as f:
            lines = f.readlines()
        data_origin = [json.loads(line) for line in lines]
        data = []
        print_rank("Loading Data")
        for d in tqdm(data_origin, disable=(get_rank() != 0)):
            prompt_ids = self.tokenizer.encode(d["prompt"])
            output_ids = None
            if "output" in d:
                if isinstance(d["output"], list):
                    output_ids = self.tokenizer.encode(d["output"][0])
                else:
                    output_ids = self.tokenizer.encode(d["output"])
            data.append({
                "prompt_ids": prompt_ids,
                "output_ids": output_ids
            })
        print_rank("Load End")
        return data, data_origin


    def verbalizer(self):
        return self.label_map

    def __getitem__(self, index: int):
        data = self.data[index]

        output_ids = data["output_ids"]
        data = data["prompt_ids"]
        
        prompt_length = self.max_prompt_length

        prompt = data[:prompt_length]
        rest = data[prompt_length:]  
        if output_ids is not None:
            rest = output_ids  
    
        return index, prompt, rest
    
    def collate(self, samples):
        bs = len(samples)
        
        max_prompt_length = self.max_prompt_length
        max_rest_length = max([len(samp[2]) for samp in samples])
        
        model_batch = {
            "input_ids": torch.ones(bs, max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_prompt_length, dtype=torch.long),
            # "position_ids": torch.zeros(bs, max_prompt_length, dtype=torch.long)
        }
        
        no_model_batch = {
            "idx": torch.zeros(bs, dtype=torch.long),
            "rest_ids": torch.ones(bs, max_rest_length, dtype=torch.long) * self.pad_id
        }
        
        for i, (idx, prompt, rest) in enumerate(samples):
            # left padding
            model_batch["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            model_batch["attention_mask"][i][-len(prompt):] = 1
            # model_batch["position_ids"][i][-len(prompt):] = torch.arange(len(prompt))
            no_model_batch["idx"][i] = idx
            no_model_batch["rest_ids"][i][:len(rest)] = torch.tensor(rest, dtype=torch.long)
        
        return model_batch, no_model_batch

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)        
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch
