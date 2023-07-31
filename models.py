import os
import openai
import backoff 
from tensor_parallel import TensorParallelPreTrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

MODEL_CACHE_DIR="/scratch/ylu130/models"
completion_tokens = prompt_tokens = 0

api_key = os.environ["OPENAI_API_KEY"]
if api_key == None or api_key == "":
    raise Exception("OPENAI_API_KEY not found")
else:
    openai.api_key = api_key

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def chatcompletions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def chatgpt(messages, model, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    res = chatcompletions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    outputs.extend([choice["message"]["content"] for choice in res["choices"]])
    # log completion tokens
    completion_tokens += res["usage"]["completion_tokens"]
    prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs

def completiongpt(prompt, model, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    res = completions_with_backoff(model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    outputs.extend([choice["text"] for choice in res["choices"]])
    # log completion tokens
    completion_tokens += res["usage"]["completion_tokens"]
    prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs

def gpt_usage(model="gpt-4"):
    global completion_tokens, prompt_tokens
    if model == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif model == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif "davinci" in model:
        cost = completion_tokens / 1000 * 0.02 + prompt_tokens / 1000 * 0.02
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

def lm(prompt, model, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    if "davinci" in model:
        return completiongpt(prompt, model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    # elif "llama" in model:
    #     return llama(prompt, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    else:
        messages = [{"role": "user", "content": prompt}]
        return chatgpt(messages, model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


class llama():
    def __init__(self, temperature=0.7, max_tokens=1000, n=1) -> None:
        MODEL_CACHE_DIR="/scratch/ylu130/models"
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", cache_dir=MODEL_CACHE_DIR)
        self.model = TensorParallelPreTrainedModel(self.model, ["cuda:8", "cuda:9"])
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", cache_dir=MODEL_CACHE_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n

    def __call__(self, prompts) -> list:
        # support batch inference
    
        stop_words = ['\n', '.\n']
        stop_ids = [self.tokenizer.encode(w)[0] for w in stop_words]
        stop_criteria = KeywordsStoppingCriteria(stop_ids)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        PROMPT_LENGTH = len(inputs['input_ids'][0])
        outputs_ids = self.model.generate(inputs["input_ids"].cuda(8), 
                                          attention_mask=inputs["attention_mask"].cuda(8), 
                                          temperature=self.temperature, 
                                          max_length=self.max_tokens, 
                                          num_return_sequences=self.n, 
                                          do_sample=True,
                                          stopping_criteria=StoppingCriteriaList([stop_criteria]))
        outputs_ids = outputs_ids[:, PROMPT_LENGTH:]
        return self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
