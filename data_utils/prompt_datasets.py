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

        self.data = self.load_data_json(data_path)
                
        if num > 0:
            self.data = self.data[:num]
            
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
        print_rank(f"Num instances: {len(self.data)}")
            
    def __len__(self):
        return self.num

    def load_data_json(self, data_path):
        if os.path.exists(os.path.join(data_path, f"{self.args.data_name}.jsonl")):
            data_path = os.path.join(data_path, f"{self.args.data_name}.jsonl")
        else:
            print_rank(f"WARNING: {os.path.join(data_path, f'{self.args.data_name}.jsonl')} does not exist")

        with open(data_path) as f:
            lines = f.readlines()
        data_origin = [json.loads(line) for line in lines]
        data = []
        for d in tqdm(data_origin, disable=(get_rank() != 0), desc="Loading data"):
            data.append({
                "prompt": d["prompt"].replace("<n>", "\n"),
                "output": d["output"]
            })
        return data

    def __getitem__(self, index: int):
        data = self.data[index]

        output = data["output"]
        prompt = data["prompt"]
        
        return index, prompt, output
    
    def collate(self, samples):     
        prompt_batch = [sample[1] for sample in samples]
        prompt_ids = self.tokenizer.batch_encode_plus(prompt_batch, 
                                                 return_tensors="pt", 
                                                 max_length=self.max_prompt_length, 
                                                 truncation=True, 
                                                 padding='max_length',
                                                 return_token_type_ids=False,)
        
        answer_batch = [sample[2] for sample in samples]
        indices = [sample[0] for sample in samples]
        
        return indices, prompt_ids, answer_batch

    def move_to_device(self, prompt_ids, device):
        for t in prompt_ids:
            if torch.is_tensor(prompt_ids[t]):
                prompt_ids[t] = prompt_ids[t].to(device)