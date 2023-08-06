# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", cache_dir="/scratch/ylu130/model-hf")

tokenizer.save_pretrained("/scratch/ylu130/model/llama-2-7b")


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", cache_dir="/scratch/ylu130/model-hf")
model.save_pretrained("/scratch/ylu130/model/llama-2-13b")
