import transformers
# set visible gpu to cuda:2


tokenizer = transformers.AutoTokenizer.from_pretrained("/scratch/ylu130/model/llama-2-7b", padding_side="left")
model = transformers.AutoModelForCausalLM.from_pretrained("/scratch/ylu130/model/llama-2-7b").to("cuda:2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
prompt = """### Instruction:Answer the following multiple choice question.

Input: Mary had 89 stickers.  She used 3 large stickers on the front page of her journal and 7 stickers each to 6 other pages of her journal. How many stickers does Mary have remaining?
Output: 44

Input: Zach is saving his money to buy a brand new bike that costs $100.  His weekly allowance is $5.  His parent will pay him an extra $10 to mow the lawn.  His neighbor will pay him $7 per hour to babysit their son.  He has already saved up $65.  He'll receive his allowance on Friday and he's planning on babysitting for 2 hours this Saturday after he mows the lawn.  How much more money does Zach need to earn before he can buy the bike?
Output: 6

Input: Mark has kangaroos and goats.  Kangaroos have two legs and goats have four legs.  If he has 23 kangaroos and three times as many goats as kangaroos what is the total number of legs of all his animals?
Output: 322

Input: Josh’s mom gives him $20 to go shopping at the mall. He buys a hat for $10 and a pencil for $2. Then he buys four cookies. If each cookie costs $1.25, how much money does Josh have left?
Output: 3

Input: George's bowling team is one round away from breaking the league record for most points scored in a season. The old record is an average score per player of 287 per round. Each team has 4 players and there are 10 rounds in the season. Through the first 9 rounds, his team has scored a total of 10,440. How many points less than the current league record per game average is the minimum average they need to score, per player, in the final round to tie the league record?
Output: 27

Input: Max was doing homework in three different subjects. It took him 20 minutes to finish tasks from biology and two times more time to finish history. Geography took him the most time, three times more than history. How much time did Max spend on doing his homework?
Output: 180

Input: Sophia ate 1/6 of her pie and she put the rest on the fridge. If the pie left in the fridge weighs 1200 grams, how many grams did Sophia eat?
Output: 240

Input: Sarah, Mary, and Tuan decided to go to the restaurant for a meal. They decided to split the cost of the meal evenly. If the total price of the meal comes to $67 and they have a coupon for $4, how much does each person need to contribute to the bill?
Output: 21

Input: Tom's brother is 4 times as old as Tom's dog. If in 6 years, Tom's brother will be 30 years, how old is Tom's dog going to be in six years?
Output: 12

Input: There are 50 children at the party. Three-fifths of them are boys. How many of the children are girls?
Output: 20

Input: Gail has some bills in her wallet which amount to $100. She has four $5 bills and three $20 bills, and the rest are $10 bills. How many $10 bills are in her wallet?
Output: 2

Input: A 220-liter barrel has a small leak. It lost 10% of its contents before anyone noticed. How many liters are left in the barrel?
Output: 198

Input: Markese earned 5 fewer dollars than Evan. Together they earned $37. How many dollars did Markese earn? Use E to represent how many dollars Evan earned.
Output: 16

Input: Lou Senior took 3 cookies out of the cookie jar and ate them.  Since he didn't get caught by his wife, he went back the next day and took another 3 cookies out of the jar.  But after eating just one of the cookies, he felt guilty about it and put the other two cookies back.  His son, Louie Junior saw that his Dad was eating cookies.  So, Louie Junior took seven cookies out of the jar and hid them in his bedroom for later.  The next morning, Debra, Lou's wife looked into the cookie jar and reacted by accusing her husband of eating half of the cookies out of the cookie jar.  How many cookies remained in the jar?
Output: 11

Input: John had $200. He gave 3/8 of his money to his mother and 3/10 to his father. How much money did John have left?
Output: 65

Input: Tonya has $150.00 on her credit card.  If she leaves any balance on her card at the end of the month, she is charged 20% interest.  If she makes a $50.00 payment on her card, what will be the new balance?
Output: 120

Input: In her first term, Governor Sandoval gave twice as many commencement addresses as Governor Hawkins. Governor Sloan gave ten more commencement addresses than Governor Sandoval in the same amount of time. If Governor Sandoval gave 12 commencement addresses, how many commencement addresses did the three of them give altogether?
Output: 40

Input: If Buzz bought a pizza with 78 slices at a restaurant and then decided to share it with the waiter in the ratio of 5:8, with Buzz's ratio being 5, what's twenty less the number of slices of pizza that the waiter ate?
Output: 28

Input: Jolene and Phil have four children, each with the same birthday.  They gave birth to their first child exactly 15 years ago.  They gave birth to their second child exactly one year after the birth of their first child.  They gave birth to their third child on the fourth birthday of their second child. Two years after the birth of their third child, they gave birth to their fourth child.  How old, in years, is their fourth child?
Output: 8

Input: Each purple book has 230 pages. Each orange book contains 510 pages. Mirella read 5 purple books and 4 orange books. How many more orange pages did she read than purple pages?
Output: 890

Input: Isabella has $45 more than Sam but only $15 more than Giselle. If Giselle has $120, calculate the total amount of money each shopper will receive if Isabella, Sam, and Giselle donate the money to three shoppers at their local town's supermarket who then decides to share it equally.
Output: 115

Input:The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change? Choices:  A: ignore B: enforce C: authoritarian D: yell at E: avoid
Output:"""

inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True, padding='max_length')

outputs = model.generate(
                         inputs["input_ids"].to("cuda:2"),
                         attention_mask=inputs["attention_mask"].to("cuda:2"),
                         do_sample=True, 
                         top_p=1, 
                         top_k=50, 
                         max_new_tokens=512, 
                         temperature=1, 
                         no_repeat_ngram_size=6, 
                         num_beams=1)
print(inputs['input_ids'].shape)
print(inputs['attention_mask'].shape)
print(inputs['input_ids'])
print(inputs['attention_mask'])
print(outputs.shape)
print(tokenizer.decode(outputs[0], skip_special_tokens=True)) 


# import json

# with open("/scratch/ylu130/processed_data/commonsenseqa/out-domain/o2-tgsm8k-s1-rTrue/commonsenseqa.jsonl", "r") as f:
#     data = [json.loads(line) for line in f.readlines()]

# print(data[1]["prompt"])


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

