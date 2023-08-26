# import transformers
# import tensor_parallel as tp
# tokenizer = transformers.AutoTokenizer.from_pretrained("/scratch/ylu130/model/llama-2-7b")
# model = transformers.AutoModelForCausalLM.from_pretrained("/scratch/ylu130/model/llama-2-7b").to("cuda:0")

# prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:Answer the following multiple choice question.

# ### Demonstration:
# Input: For every 12 cans you recycle, you receive $0.50, and for every 5 kilograms of newspapers, you receive $1.50. If your family collected 144 cans and 20 kilograms of newspapers, how much money would you receive?
# Rationales: First, calculate how many sets of 12 cans the family collected by dividing the total number of cans (144) by the number of cans in a set (12). This gives you 12 sets of 12 cans.
# Next, multiply the rate of return for cans ($0.50) by the number of sets (12) to find the total return for the cans. This gives you $6.00.
# Then, calculate how many sets of 5 kilograms of newspapers the family collected by dividing the total weight of newspapers (20 kg) by the weight in a set (5 kg). This gives you 4 sets of 5 kilograms.
# After, multiply the rate of return for newspapers ($1.50) by the number of sets (4) to find the total return for the newspapers. This gives you $6.00.
# Finally, add the total return for the cans and the newspapers together ($6.00 + $6.00) to find the total return the family would receive. This results in $12.00. So, the answer is $12.
# Answer: 12

# Input: Betty picked 16 strawberries. Matthew picked 20 more strawberries than Betty and twice as many as Natalie. They used their strawberries to make jam. One jar of jam used 7 strawberries and they sold each jar at $4. How much money were they able to make from the strawberries they picked?
# Rationales: Betty picked 16 strawberries.
# Matthew picked 20 more strawberries than Betty. Hence he picked 16 + 20 = 36 strawberries.
# Matthew also picked twice as many strawberries as Natalie. Hence Natalie picked 36/2 = 18 strawberries.
# Together, they picked a total of 16 + 36 + 18 = 70 strawberries.
# They used these strawberries to make jam. Each jar of jam required 7 strawberries. Hence they were able to make 70/7 = 10 jars of jam.
# Lastly, Each jar of jam was sold for $4. Hence, they made 10 x $4 = $40 from the strawberries they picked.
# Answer: 40

# Input: Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?
# Rationales: Jack measures the thickness of his books and finds out that it is 12 inches.
# He knows that 1 inch can accommodate 80 pages.
# So, to find out how many pages, in total, are in his books we multiply the total thickness by the number of pages per inch.
# This gives us a total of 960 pages, since 12 inches times 80 pages per inch equals 960 pages.

# Each book, on average, contains 160 pages.
# This is calculated by dividing the total number of pages, which is 960, by the number of books, which is 6.
# So, 960 pages divided by 6 books equals 160 pages per book.
# Answer: 160

# Input: James dumps his whole collection of 500 Legos on the floor and starts building a castle out of them.  He uses half the pieces before finishing and is told to put the rest away.  He puts all of the leftover pieces back in the box they came from, except for 5 missing pieces that he can't find.  How many Legos are in the box at the end?
# Rationales: First, determine how many Legos James used by dividing the initial number of Legos by 2. This is because James used half of the pieces, thus: 500/2 = 250 Legos used.
# Next, subtract the number of missing Legos from the remaining unused Legos. So: 250 Legos unused - 5 missing Legos = 245 Legos.
# Therefore, the number of Legos James puts back into the box is 245.
# Answer: 245

# Input: Ines had $20 in her purse. She bought 3 pounds of peaches, which are $2 per pound at the local farmers’ market. How much did she have left?
# Rationales: Ines starts with $20.
# The cost of the peaches is determined by how many pounds she bought multiplied by the price per pound.
# The peaches are $2 per pound.
# Ines bought 3 pounds of peaches, so the cost is 3 * $2 = $6.
# This means she spent $6 on the peaches.
# She originally had $20, and she spent $6, so she now has $20 - $6 = $14 left.
# Therefore, Ines has $14 left.
# Answer: 14

# Input: Aaron pays his actuary membership fees each year. The membership fee increases yearly by $10. If he pays $80 in the first year, how much does his membership cost, in dollars, in the sixth year?
# Rationales: The cost of Aaron's membership in the first year is $80.
# The membership fee increases by $10 each year.
# Therefore, in the second year, the cost of his membership would be his first year's fee plus $10, which equals $80 + $10 = $90.
# In the third year, his membership fee would be his second year's fee plus $10, which equals $90 + $10 = $100.
# Following this pattern, in the fourth year, the cost of his membership would be his third year's fee plus $10, which equals $100 + $10 = $110.
# In the fifth year, his membership fee would be his fourth year's fee plus $10, which equals $110 + $10 = $120.
# Finally, in the sixth year, the cost of his membership would be his fifth year's fee plus $10, which equals $120 + $10 = $130.
# Therefore, his membership cost in the sixth year would be $130.
# Answer: 130

# ### Input:The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Choices:  A: ignore B: enforce C: authoritarian D: yell at E: avoid

# ### Response:"""

# inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda:0")
# outputs = model.generate(inputs, do_sample=True, top_p=1, top_k=50, max_new_tokens=512, temperature=1, no_repeat_ngram_size=6, num_beams=1)
# print(tokenizer.decode(outputs[0])) 


import json

with open("/scratch/ylu130/processed_data/commonsenseqa/out-domain/o2-tgsm8k-s1-rTrue/commonsenseqa.jsonl", "r") as f:
    data = [json.loads(line) for line in f.readlines()]

print(data[1]["prompt"])


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

# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", cache_dir = "/scratch/ylu130/model-hf")
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", cache_dir = "/scratch/ylu130/model-hf")

# # save model
# model.save_pretrained("/scratch/ylu130/model/opt-1.3b")
# tokenizer.save_pretrained("/scratch/ylu130/model/opt-1.3b")

