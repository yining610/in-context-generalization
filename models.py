import os
import openai
import backoff 
from tensor_parallel import TensorParallelPreTrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from typing import List
from typeguard import typechecked

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

def llama(prompts, model, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    MODEL_CACHE_DIR="/scratch/ylu130/models"

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", cache_dir=MODEL_CACHE_DIR)
    model = TensorParallelPreTrainedModel(model, ["cuda:8", "cuda:9"])
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", cache_dir=MODEL_CACHE_DIR)

    # batch inference
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs["input_ids"].cuda(8), attention_mask=inputs["attention_mask"].cuda(8), temperature=temperature, max_length=max_tokens, num_return_sequences=n, do_sample=True)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def lm(prompt, model, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    if "davinci" in model:
        return completiongpt(prompt, model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    elif "llama" in model:
        return llama(prompt, model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    else:
        messages = [{"role": "user", "content": prompt}]
        return chatgpt(messages, model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
        