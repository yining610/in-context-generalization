import transformers
import tensor_parallel as tp
tokenizer = transformers.AutoTokenizer.from_pretrained("/scratch/ylu130/model/llama-2-7b")
model = transformers.AutoModelForCausalLM.from_pretrained("/scratch/ylu130/model/llama-2-7b").to("cuda:0")

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:Answer the following multiple choice question.

### Demonstration:
Input: He fantasied about getting a what while driving to work and the pros and cons of the extra responsibilities and benefits? Choices:  A: new car B: promotion C: boredom D: impatience E: pressure
Rationales: The question talks about "extra responsibilities and benefits", both of which are aspects associated with getting a promotion at work. 

A new car (Choice A) wouldn't necessarily add extra responsibilities and benefits in the context of work. Boredom (Choice C) and impatience (Choice D) are states of mind, not things that could cause added responsibilities and benefits at work. Pressure (Choice E), although it can be increased by a promotion, does not inherently carry with it benefits. Thus, none of these options fit the context of the question as well as a promotion (Choice B).

Therefore, the answer is B: promotion.
Answer: B: promotion

Input: He was good at traditional science but excelled at social science, his favorite subject was what? Choices:  A: geography B: history studies C: math D: religion E: dancing
Rationales: 1. The question highlights that the person in discussion is good at traditional science but excels at social science.
2. First, since he excels in social science, options C (math) and E (dancing) can be eliminated because they do not fall under the category of social sciences.
3. The social sciences are branches of study that deal with societal issues and human behavior, including subjects like history, geography, anthropology, sociology, etc.
4. Given the remaining options are geography, history studies,  and religion which are all considered as social sciences, we now refer to the next part of the question.
5. It is stated that his favorite subject is being asked, which we can assume is the area he "excelled" at.
6. The answer provided is "B: history studies," implying that it was his favorite subject, and he excelled in it.
Answer: B: history studies

Input: Jan wasn't very good at studying.  What might help him study better? Choices:  A: having a bigger brain B: headaches C: inspiration D: more intelligence E: understanding
Rationales: Intermediate Reasoning Steps:
1. The question presents a problem, which is Jan's difficulty with studying.
2. It then asks for a solution that could potentially help Jan study better.
3. Each of the provided choices can be evaluated and either ruled out or considered based on their potential effect on studying ability.
4. The first choice "A: having a bigger brain" could provide some benefits in that more brain matter could theoretically allow for greater neural connections and learning potential. However, the size of the brain does not directly correlate to increased learning or studying capability. Thus, it is not the best answer.
5. The second choice "B: headaches" would hinder, rather than help studying, so it can be eliminated.
6. Option "C: inspiration" could motivate Jan to study but does not necessarily improve his studying skills or increase his learning capacity. It is a factor that could lead to more effective studying, but it does not directly improve one's study skills, hence we can rule this out.
7. Choice "D: more intelligence" implies an increased capability to understand, learn, reason, and problem solve, all of which can directly impact studying effectiveness. Thus, it is a viable answer.
8. The final choice "E: understanding" is also an important factor in effective studying. However, it is more a result of effective studying, not necessarily a contributing factor. It is possible to be good at understanding yet still struggle with studying. 
9. Comparatively, more intelligence implies a broader range of cognitive abilities which directly aid in studying.
10. Thus, out of all the choices provided, "D: more intelligence" is the most likely to help Jan study better.
Answer: D: more intelligence

Input: For every 12 cans you recycle, you receive $0.50, and for every 5 kilograms of newspapers, you receive $1.50. If your family collected 144 cans and 20 kilograms of newspapers, how much money would you receive?
Rationales: The question provides that the family collected 144 cans and each set of 12 cans equals $0.50. To find out how much the family would earn from recycling the cans, we divide the total number of cans by the cans in each set. This gives us 12 sets. Multiplying the number of sets by the dollar amount received for each, we find that the family would receive $6 for the cans.

Furthermore, the question states the family has collected 20 kilograms of newspapers and for every 5 kilograms, they receive $1.50. To find out how much the family would earn from recycling the newspapers, we divide the total kilograms of newspapers by the kilograms in each set, yielding 4 sets. By multiplying the number of sets by the dollar amount received per set, we find that the family would receive $6 for the newspapers.

To find out the total amount the family would receive, we sum the money received from recycling the cans and the newspapers. The family would receive a total of $12.
Answer: 12

Input: Betty picked 16 strawberries. Matthew picked 20 more strawberries than Betty and twice as many as Natalie. They used their strawberries to make jam. One jar of jam used 7 strawberries and they sold each jar at $4. How much money were they able to make from the strawberries they picked?
Rationales: First, we need to calculate how many strawberries each person collected. We already know that Betty picked 16 strawberries. According to the problem, Matthew picked 20 more strawberries than Betty, so he collected 16 + 20 = 36 strawberries. It also states that Matthew picked twice as many strawberries as Natalie, so we can divide Matthew's total by 2 to find out that Natalie picked 36/2 = 18 strawberries.

Next, we add the number of strawberries each person collected to find the total number of strawberries: 16 (Betty's) + 36 (Matthew's) + 18 (Natalie's) = 70 strawberries.

The problem states that they used their strawberries to make jam, and that each jar of jam uses 7 strawberries. To find out how many jars of jam they can make, we divide the total number of strawberries by the number of strawberries per jar: 70/7 = 10 jars of jam.

Finally, we calculate how much money they were able to make by selling the jars of jam. According to the problem, each jar is sold for $4. Therefore, they earn 10 (jars) x $4 (per jar) = $40 from the strawberries they picked. 

So, they were able to make $40 from the strawberries they picked.
Answer: 40

Input: Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?
Rationales: Step 1: Determine the total number of pages in the stack of books. Since 80 pages equate to an inch of thickness, multiply the total thickness of the stack by 80 to get this number. That is, 80 pages/inch * 12 inches = 960 pages.
Step 2: To find the average number of pages per book, divide the total number of pages by the total number of books. That is, 960 pages / 6 books = 160 pages per book on average.
Answer: 160

### Input:The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Choices:  A: ignore B: enforce C: authoritarian D: yell at E: avoid

### Response:"""

inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda:0")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0])) 


# import json

# with open("/scratch/ylu130/processed_data/commonsenseqa/mixed-domain/i4-o4-tgsm8k-s42-rFalse/commonsenseqa.jsonl", "r") as f:
#     data = [json.loads(line) for line in f.readlines()]

# print(data[0]["prompt"])

# # count tokens
# from transformers import AutoTokenizer
# text = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:Answer the following multiple choice question.

# ### Demonstration:
# Input: He fantasied about getting a what while driving to work and the pros and cons of the extra responsibilities and benefits? Choices:  A: new car B: promotion C: boredom D: impatience E: pressure
# Rationales: The question talks about "extra responsibilities and benefits", both of which are aspects associated with getting a promotion at work. 

# A new car (Choice A) wouldn't necessarily add extra responsibilities and benefits in the context of work. Boredom (Choice C) and impatience (Choice D) are states of mind, not things that could cause added responsibilities and benefits at work. Pressure (Choice E), although it can be increased by a promotion, does not inherently carry with it benefits. Thus, none of these options fit the context of the question as well as a promotion (Choice B).

# Therefore, the answer is B: promotion.
# Answer: B: promotion

# Input: He was good at traditional science but excelled at social science, his favorite subject was what? Choices:  A: geography B: history studies C: math D: religion E: dancing
# Rationales: 1. The question highlights that the person in discussion is good at traditional science but excels at social science.
# 2. First, since he excels in social science, options C (math) and E (dancing) can be eliminated because they do not fall under the category of social sciences.
# 3. The social sciences are branches of study that deal with societal issues and human behavior, including subjects like history, geography, anthropology, sociology, etc.
# 4. Given the remaining options are geography, history studies,  and religion which are all considered as social sciences, we now refer to the next part of the question.
# 5. It is stated that his favorite subject is being asked, which we can assume is the area he "excelled" at.
# 6. The answer provided is "B: history studies," implying that it was his favorite subject, and he excelled in it.
# Answer: B: history studies

# Input: Jan wasn't very good at studying.  What might help him study better? Choices:  A: having a bigger brain B: headaches C: inspiration D: more intelligence E: understanding
# Rationales: Intermediate Reasoning Steps:
# 1. The question presents a problem, which is Jan's difficulty with studying.
# 2. It then asks for a solution that could potentially help Jan study better.
# 3. Each of the provided choices can be evaluated and either ruled out or considered based on their potential effect on studying ability.
# 4. The first choice "A: having a bigger brain" could provide some benefits in that more brain matter could theoretically allow for greater neural connections and learning potential. However, the size of the brain does not directly correlate to increased learning or studying capability. Thus, it is not the best answer.
# 5. The second choice "B: headaches" would hinder, rather than help studying, so it can be eliminated.
# 6. Option "C: inspiration" could motivate Jan to study but does not necessarily improve his studying skills or increase his learning capacity. It is a factor that could lead to more effective studying, but it does not directly improve one's study skills, hence we can rule this out.
# 7. Choice "D: more intelligence" implies an increased capability to understand, learn, reason, and problem solve, all of which can directly impact studying effectiveness. Thus, it is a viable answer.
# 8. The final choice "E: understanding" is also an important factor in effective studying. However, it is more a result of effective studying, not necessarily a contributing factor. It is possible to be good at understanding yet still struggle with studying. 
# 9. Comparatively, more intelligence implies a broader range of cognitive abilities which directly aid in studying.
# 10. Thus, out of all the choices provided, "D: more intelligence" is the most likely to help Jan study better.
# Answer: D: more intelligence

# Input: For every 12 cans you recycle, you receive $0.50, and for every 5 kilograms of newspapers, you receive $1.50. If your family collected 144 cans and 20 kilograms of newspapers, how much money would you receive?
# Rationales: The question provides that the family collected 144 cans and each set of 12 cans equals $0.50. To find out how much the family would earn from recycling the cans, we divide the total number of cans by the cans in each set. This gives us 12 sets. Multiplying the number of sets by the dollar amount received for each, we find that the family would receive $6 for the cans.

# Furthermore, the question states the family has collected 20 kilograms of newspapers and for every 5 kilograms, they receive $1.50. To find out how much the family would earn from recycling the newspapers, we divide the total kilograms of newspapers by the kilograms in each set, yielding 4 sets. By multiplying the number of sets by the dollar amount received per set, we find that the family would receive $6 for the newspapers.

# To find out the total amount the family would receive, we sum the money received from recycling the cans and the newspapers. The family would receive a total of $12.
# Answer: 12

# Input: Betty picked 16 strawberries. Matthew picked 20 more strawberries than Betty and twice as many as Natalie. They used their strawberries to make jam. One jar of jam used 7 strawberries and they sold each jar at $4. How much money were they able to make from the strawberries they picked?
# Rationales: First, we need to calculate how many strawberries each person collected. We already know that Betty picked 16 strawberries. According to the problem, Matthew picked 20 more strawberries than Betty, so he collected 16 + 20 = 36 strawberries. It also states that Matthew picked twice as many strawberries as Natalie, so we can divide Matthew's total by 2 to find out that Natalie picked 36/2 = 18 strawberries.

# Next, we add the number of strawberries each person collected to find the total number of strawberries: 16 (Betty's) + 36 (Matthew's) + 18 (Natalie's) = 70 strawberries.

# The problem states that they used their strawberries to make jam, and that each jar of jam uses 7 strawberries. To find out how many jars of jam they can make, we divide the total number of strawberries by the number of strawberries per jar: 70/7 = 10 jars of jam.

# Finally, we calculate how much money they were able to make by selling the jars of jam. According to the problem, each jar is sold for $4. Therefore, they earn 10 (jars) x $4 (per jar) = $40 from the strawberries they picked. 

# So, they were able to make $40 from the strawberries they picked.
# Answer: 40

# Input: Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?
# Rationales: Step 1: Determine the total number of pages in the stack of books. Since 80 pages equate to an inch of thickness, multiply the total thickness of the stack by 80 to get this number. That is, 80 pages/inch * 12 inches = 960 pages.
# Step 2: To find the average number of pages per book, divide the total number of pages by the total number of books. That is, 960 pages / 6 books = 160 pages per book on average.
# Answer: 160

# ### Input:The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Choices:  A: ignore B: enforce C: authoritarian D: yell at E: avoid

# ### Response:"""

# tokenizer = AutoTokenizer.from_pretrained("/scratch/ylu130/model/llama-2-7b")
# print(len(tokenizer(text)["input_ids"]))