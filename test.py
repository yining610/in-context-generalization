from tensor_parallel import TensorParallelPreTrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

MODEL_CACHE_DIR="/scratch/ylu130/models"

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=MODEL_CACHE_DIR)
model = TensorParallelPreTrainedModel(model, ["cuda:8", "cuda:9"])

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir=MODEL_CACHE_DIR)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

prompts = ["How are you?", "How old are you?"]
prompts_tokens = tokenizer(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
PROMPT_LENGTH = len(prompts_tokens['input_ids'][0])
# support batch inference
output_tokens = model.generate(prompts_tokens["input_ids"].cuda(8),
                               attention_mask=prompts_tokens["attention_mask"].cuda(8),
                               max_length=512,
                                 temperature=0.7,
                                    num_beams=1,
                                    num_return_sequences=1,
                                    do_sample=True,
                                    eos_token_id=tokenizer.eos_token_id,
)

output_tokens = output_tokens[:, PROMPT_LENGTH:]
outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
print(outputs)