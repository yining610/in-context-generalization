import transformers
import tensor_parallel as tp
tokenizer = transformers.AutoTokenizer.from_pretrained("/scratch/ylu130/model/llama-2-7b")
model = transformers.AutoModelForCausalLM.from_pretrained("/scratch/ylu130/model/llama-2-7b").to("cuda:0")

# model = tp.tensor_parallel(model, ["cuda:0", "cuda:1"])  # <- each GPU has half the weights
prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: Answer the following multiple choice question.

### Demonstration:Input: He fantasied about getting a what while driving to work and the pros and cons of the extra responsibilities and benefits? Choices:  A: new car B: promotion C: boredom D: impatience E: pressure
Rationales: 1. The question is about someone who is fantasizing about something while driving to work. This person is also considering the pros and cons of the extra responsibilities and benefits. This helps narrow down the choices to something that would involve extra responsibilities and benefits.

2. Option A: A new car does not involve extra responsibilities and benefits. The responsibilities and benefits of a car remain the same, regardless of whether it's new or old. Hence, this option is incorrect.

3. Option B: A promotion at work would definitely involve extra responsibilities. It would also come with benefits, such as a higher salary or more influence within the company. Hence, this option fits the criteria mentioned in the question.

4. Option C: Boredom does not involve extra responsibilities and benefits. Hence, this option is incorrect.

5. Option D: Impatience does not involve extra responsibilities and benefits. Hence, this option is incorrect.

6. Option E: Pressure does not offer benefits, although it might involve extra responsibilities. However, considering the context of the question, it's unlikely that someone would fantasize about having more pressure. Hence, this option is incorrect.

7. Therefore, by process of elimination and by matching the criteria given in the question to the options, the correct answer is B: promotion.
Answer: B: promotion

### Input:He wanted a house that was gated off from other places, where should he start looking? Choices:  A: neighborhood B: subdivision C: city D: suburbs E: street

### Response:"""

inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda:0")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0])) 

