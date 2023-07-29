from tensor_parallel import TensorParallelPreTrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_CACHE_DIR="/scratch/ylu130/models"

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", cache_dir=MODEL_CACHE_DIR)
model = TensorParallelPreTrainedModel(model, ["cuda:6", "cuda:7"])


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", cache_dir=MODEL_CACHE_DIR)

while True:
    prompt = input(">>> ")
    tokens = tokenizer(prompt, return_tensors="pt")
    output = model.generate(tokens["input_ids"].cuda(6), attention_mask=tokens["attention_mask"].cuda(6))[0]
    print(tokenizer.decode(output))


