import transformers
# set visible gpu to cuda:2


tokenizer = transformers.AutoTokenizer.from_pretrained("/scratch/ylu130/model/llama-2-7b")
model = transformers.AutoModelForCausalLM.from_pretrained("/scratch/ylu130/model/llama-2-7b").to("cuda:2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
prompt = """### Instruction:Answer the following multiple choice question.

Input: The difference in ages between Richard and Hurley is 20. If Hurley is 14 years old, what are their combined ages 40 years from now?
Output: 128

Input: Pablo’s mother agrees to pay him one cent for every page he reads. He plans to save the money for some candy. Pablo always checks out books that are exactly 150 pages. After reading his books, he went to the store and bought $15 worth of candy and had $3 leftover. How many books did Pablo read?
Output: 12

Input: A group of six friends planned to buy a car. The cost of the car is $1700 and they plan to share the cost equally. They had a car wash to help raise funds, which would be taken out of the total cost. The remaining cost would be split between the six friends. At the car wash, they earn $500. However, Brad decided not to join in the purchase of the car. How much more does each friend have to pay now that Brad isn't participating?
Output: 40

Input: Harper needs to buy teacher appreciation gifts for her children’s teachers.  Her son has 3 different teachers and her daughter has 4.  If she spent $70 on gifts, how much did each gift cost?
Output: 10

Input: The chef has 60 eggs. He puts 10 eggs in the fridge and uses the rest to make cakes. If he used 5 eggs to make one cake, how many cakes did the chef make?
Output: 10

Input: After collecting all the old electronics in their house, Lauryn made $2000 from selling the items on eBay. If her friend Aurelia also made 70% of what she sold on eBay after selling her used electronics, calculate the total amount of money the two friends made on eBay.
Output: 3400

Input: $240 was divided between Kelvin and Samuel. Samuel received 3/4 of the money. From his share, Samuel then spent 1/5 of the original $240 on drinks. How much does Samuel have left?
Output: 132

Input: John and Sam were hungry so they ordered an extra-large pizza that was pre-sliced into 12 pieces.  John ate 3 slices while Sam ate twice the amount that John ate.  How many slices of pizza were left?
Output: 3

Input: On Thursday the Meat Market sold 210kg of ground beef. On Friday they sold twice that amount. On Saturday they only sold 130kg. On Sunday they sold half of what they sold Saturday. If they originally planned to sell only 500kg, how much meat did they sell beyond their original plans?
Output: 325

Input: There is a massive rainstorm lasting 4 days.  There is an area that collects water to prevent flooding in the area.  It ends up overflowing on the 4th day.  The area can hold the equivalent of 6 feet of rain.  It can also drain out the equivalent of 3 inches of rain per day to the nearby river without causing problems.  The first day it rained 10 inches.  The second day it rained twice that much.  On the third day, it rained 50% more than the second day.  It flooded the fourth day before getting a chance to do any of the draining.  What is the minimum amount it rained on the fourth day?
Output: 21

Input: There are 6 boxes of crayons that hold 8 orange crayons. There are 7 boxes of crayons that have 5 blue crayons. There is 1 box of 11 red crayons. How many crayons are there in total?
Output: 94

Input: Joey studies for his SAT exams 2 hours per night 5 nights a week.  On the weekends, he studies 3 hours a day.  If his SAT exam is 6 weeks away, how much time will Joey spend studying?
Output: 96

Input: Alexis can sew a skirt in 2 hours and a coat in 7 hours. How long does it take for Alexis to sew 6 skirts and 4 coats?
Output: 40

Input: Amy bought a 15-foot spool of string to cut up into wicks for making candles.   If she cuts up the entire string into an equal number of 6-inch and 12-inch wicks, what is the total number of wicks she will have cut?
Output: 20

Input: Jacob loves to build things. In Jacob's toy bin there are 18 red blocks. There are 7 more yellow blocks than red blocks. There are also 14 more blue blocks than red blocks. How many blocks are there in all?
Output: 75

Input: Five adults and two children go to see a movie and buy $12 worth of concessions. The total cost of their trip is $76. If each child's ticket is $7, how much, in dollars, are the adult tickets?
Output: 10

Input: James writes a comic every other day for 4 years. If there was no leap year, how many comics has he written?
Output: 730

Input: On a 16 GB (gigabyte) capacity USB drive, 50% is already busy. Calculate the number of gigabytes still available.
Output: 8

Input: The middle school sold 6 more than two times the number of fair tickets as it did tickets to the baseball game. If 25 fair tickets were sold, how many baseball game tickets did the school sell?
Output: 56

Input: Ethyl bought Lucy two new dolls for her doll collection.  This increased the doll collection by 25%.  After the addition of the two new dolls, how many dolls are in Lucy's collection?
Output: 10

Input: Matthew asked his children how many hotdogs they wanted for dinner.  Both Ella and Emma agreed they wanted 2 hotdogs each.  Luke said he could eat twice the amount of hotdogs as his sisters while Hunter said he could only eat 1 and half times the total amount of his sisters.  How many hotdogs did Matthew need to cook?
Output: 14

Input: A porcelain vase was originally priced at $200 but went on sale for 25% off. If Donna bought the porcelain vase and paid 10% sales tax, how much did she pay in total?
Output: 165

Input: The difference in ages between Richard and Hurley is 20. If Hurley is 14 years old, what are their combined ages 40 years from now?
Output: 128

Input: Pablo’s mother agrees to pay him one cent for every page he reads. He plans to save the money for some candy. Pablo always checks out books that are exactly 150 pages. After reading his books, he went to the store and bought $15 worth of candy and had $3 leftover. How many books did Pablo read?
Output: 12

Input: A group of six friends planned to buy a car. The cost of the car is $1700 and they plan to share the cost equally. They had a car wash to help raise funds, which would be taken out of the total cost. The remaining cost would be split between the six friends. At the car wash, they earn $500. However, Brad decided not to join in the purchase of the car. How much more does each friend have to pay now that Brad isn't participating?
Output: 40

Input: Harper needs to buy teacher appreciation gifts for her children’s teachers.  Her son has 3 different teachers and her daughter has 4.  If she spent $70 on gifts, how much did each gift cost?
Output: 10

Input: The chef has 60 eggs. He puts 10 eggs in the fridge and uses the rest to make cakes. If he used 5 eggs to make one cake, how many cakes did the chef make?
Output: 10

Input: After collecting all the old electronics in their house, Lauryn made $2000 from selling the items on eBay. If her friend Aurelia also made 70% of what she sold on eBay after selling her used electronics, calculate the total amount of money the two friends made on eBay.
Output: 3400

Input: $240 was divided between Kelvin and Samuel. Samuel received 3/4 of the money. From his share, Samuel then spent 1/5 of the original $240 on drinks. How much does Samuel have left?
Output: 132

Input: John and Sam were hungry so they ordered an extra-large pizza that was pre-sliced into 12 pieces.  John ate 3 slices while Sam ate twice the amount that John ate.  How many slices of pizza were left?
Output: 3

Input: On Thursday the Meat Market sold 210kg of ground beef. On Friday they sold twice that amount. On Saturday they only sold 130kg. On Sunday they sold half of what they sold Saturday. If they originally planned to sell only 500kg, how much meat did they sell beyond their original plans?
Output: 325

Input: There is a massive rainstorm lasting 4 days.  There is an area that collects water to prevent flooding in the area.  It ends up overflowing on the 4th day.  The area can hold the equivalent of 6 feet of rain.  It can also drain out the equivalent of 3 inches of rain per day to the nearby river without causing problems.  The first day it rained 10 inches.  The second day it rained twice that much.  On the third day, it rained 50% more than the second day.  It flooded the fourth day before getting a chance to do any of the draining.  What is the minimum amount it rained on the fourth day?
Output: 21

Input: There are 6 boxes of crayons that hold 8 orange crayons. There are 7 boxes of crayons that have 5 blue crayons. There is 1 box of 11 red crayons. How many crayons are there in total?
Output: 94

Input: Joey studies for his SAT exams 2 hours per night 5 nights a week.  On the weekends, he studies 3 hours a day.  If his SAT exam is 6 weeks away, how much time will Joey spend studying?
Output: 96

Input: Alexis can sew a skirt in 2 hours and a coat in 7 hours. How long does it take for Alexis to sew 6 skirts and 4 coats?
Output: 40

Input: Amy bought a 15-foot spool of string to cut up into wicks for making candles.   If she cuts up the entire string into an equal number of 6-inch and 12-inch wicks, what is the total number of wicks she will have cut?
Output: 20

Input: Jacob loves to build things. In Jacob's toy bin there are 18 red blocks. There are 7 more yellow blocks than red blocks. There are also 14 more blue blocks than red blocks. How many blocks are there in all?
Output: 75

Input: Five adults and two children go to see a movie and buy $12 worth of concessions. The total cost of their trip is $76. If each child's ticket is $7, how much, in dollars, are the adult tickets?
Output: 10

Input: James writes a comic every other day for 4 years. If there was no leap year, how many comics has he written?
Output: 730

Input: On a 16 GB (gigabyte) capacity USB drive, 50% is already busy. Calculate the number of gigabytes still available.
Output: 8

Input: The middle school sold 6 more than two times the number of fair tickets as it did tickets to the baseball game. If 25 fair tickets were sold, how many baseball game tickets did the school sell?
Output: 56

Input: Ethyl bought Lucy two new dolls for her doll collection.  This increased the doll collection by 25%.  After the addition of the two new dolls, how many dolls are in Lucy's collection?
Output: 10

Input: Matthew asked his children how many hotdogs they wanted for dinner.  Both Ella and Emma agreed they wanted 2 hotdogs each.  Luke said he could eat twice the amount of hotdogs as his sisters while Hunter said he could only eat 1 and half times the total amount of his sisters.  How many hotdogs did Matthew need to cook?
Output: 14

Input: A porcelain vase was originally priced at $200 but went on sale for 25% off. If Donna bought the porcelain vase and paid 10% sales tax, how much did she pay in total?
Output: 165

Input: The difference in ages between Richard and Hurley is 20. If Hurley is 14 years old, what are their combined ages 40 years from now?
Output: 128

Input: Pablo’s mother agrees to pay him one cent for every page he reads. He plans to save the money for some candy. Pablo always checks out books that are exactly 150 pages. After reading his books, he went to the store and bought $15 worth of candy and had $3 leftover. How many books did Pablo read?
Output: 12

Input: A group of six friends planned to buy a car. The cost of the car is $1700 and they plan to share the cost equally. They had a car wash to help raise funds, which would be taken out of the total cost. The remaining cost would be split between the six friends. At the car wash, they earn $500. However, Brad decided not to join in the purchase of the car. How much more does each friend have to pay now that Brad isn't participating?
Output: 40

Input: Harper needs to buy teacher appreciation gifts for her children’s teachers.  Her son has 3 different teachers and her daughter has 4.  If she spent $70 on gifts, how much did each gift cost?
Output: 10

Input: The chef has 60 eggs. He puts 10 eggs in the fridge and uses the rest to make cakes. If he used 5 eggs to make one cake, how many cakes did the chef make?
Output: 10

Input: After collecting all the old electronics in their house, Lauryn made $2000 from selling the items on eBay. If her friend Aurelia also made 70% of what she sold on eBay after selling her used electronics, calculate the total amount of money the two friends made on eBay.
Output: 3400

Input: $240 was divided between Kelvin and Samuel. Samuel received 3/4 of the money. From his share, Samuel then spent 1/5 of the original $240 on drinks. How much does Samuel have left?
Output: 132

Input: John and Sam were hungry so they ordered an extra-large pizza that was pre-sliced into 12 pieces.  John ate 3 slices while Sam ate twice the amount that John ate.  How many slices of pizza were left?
Output: 3

Input: On Thursday the Meat Market sold 210kg of ground beef. On Friday they sold twice that amount. On Saturday they only sold 130kg. On Sunday they sold half of what they sold Saturday. If they originally planned to sell only 500kg, how much meat did they sell beyond their original plans?
Output: 325

Input: There is a massive rainstorm lasting 4 days.  There is an area that collects water to prevent flooding in the area.  It ends up overflowing on the 4th day.  The area can hold the equivalent of 6 feet of rain.  It can also drain out the equivalent of 3 inches of rain per day to the nearby river without causing problems.  The first day it rained 10 inches.  The second day it rained twice that much.  On the third day, it rained 50% more than the second day.  It flooded the fourth day before getting a chance to do any of the draining.  What is the minimum amount it rained on the fourth day?
Output: 21

Input: There are 6 boxes of crayons that hold 8 orange crayons. There are 7 boxes of crayons that have 5 blue crayons. There is 1 box of 11 red crayons. How many crayons are there in total?
Output: 94

Input: Joey studies for his SAT exams 2 hours per night 5 nights a week.  On the weekends, he studies 3 hours a day.  If his SAT exam is 6 weeks away, how much time will Joey spend studying?
Output: 96

Input: Alexis can sew a skirt in 2 hours and a coat in 7 hours. How long does it take for Alexis to sew 6 skirts and 4 coats?
Output: 40

Input: Amy bought a 15-foot spool of string to cut up into wicks for making candles.   If she cuts up the entire string into an equal number of 6-inch and 12-inch wicks, what is the total number of wicks she will have cut?
Output: 20

Input: Jacob loves to build things. In Jacob's toy bin there are 18 red blocks. There are 7 more yellow blocks than red blocks. There are also 14 more blue blocks than red blocks. How many blocks are there in all?
Output: 75

Input: Five adults and two children go to see a movie and buy $12 worth of concessions. The total cost of their trip is $76. If each child's ticket is $7, how much, in dollars, are the adult tickets?
Output: 10

Input: James writes a comic every other day for 4 years. If there was no leap year, how many comics has he written?
Output: 730

Input: On a 16 GB (gigabyte) capacity USB drive, 50% is already busy. Calculate the number of gigabytes still available.
Output: 8

Input: The middle school sold 6 more than two times the number of fair tickets as it did tickets to the baseball game. If 25 fair tickets were sold, how many baseball game tickets did the school sell?
Output: 56

Input: Ethyl bought Lucy two new dolls for her doll collection.  This increased the doll collection by 25%.  After the addition of the two new dolls, how many dolls are in Lucy's collection?
Output: 10

Input: Matthew asked his children how many hotdogs they wanted for dinner.  Both Ella and Emma agreed they wanted 2 hotdogs each.  Luke said he could eat twice the amount of hotdogs as his sisters while Hunter said he could only eat 1 and half times the total amount of his sisters.  How many hotdogs did Matthew need to cook?
Output: 14

Input: A porcelain vase was originally priced at $200 but went on sale for 25% off. If Donna bought the porcelain vase and paid 10% sales tax, how much did she pay in total?
Output: 165

Input: The difference in ages between Richard and Hurley is 20. If Hurley is 14 years old, what are their combined ages 40 years from now?
Output: 128

Input: Pablo’s mother agrees to pay him one cent for every page he reads. He plans to save the money for some candy. Pablo always checks out books that are exactly 150 pages. After reading his books, he went to the store and bought $15 worth of candy and had $3 leftover. How many books did Pablo read?
Output: 12

Input: A group of six friends planned to buy a car. The cost of the car is $1700 and they plan to share the cost equally. They had a car wash to help raise funds, which would be taken out of the total cost. The remaining cost would be split between the six friends. At the car wash, they earn $500. However, Brad decided not to join in the purchase of the car. How much more does each friend have to pay now that Brad isn't participating?
Output: 40

Input: Harper needs to buy teacher appreciation gifts for her children’s teachers.  Her son has 3 different teachers and her daughter has 4.  If she spent $70 on gifts, how much did each gift cost?
Output: 10

Input: The chef has 60 eggs. He puts 10 eggs in the fridge and uses the rest to make cakes. If he used 5 eggs to make one cake, how many cakes did the chef make?
Output: 10

Input: After collecting all the old electronics in their house, Lauryn made $2000 from selling the items on eBay. If her friend Aurelia also made 70% of what she sold on eBay after selling her used electronics, calculate the total amount of money the two friends made on eBay.
Output: 3400

Input: $240 was divided between Kelvin and Samuel. Samuel received 3/4 of the money. From his share, Samuel then spent 1/5 of the original $240 on drinks. How much does Samuel have left?
Output: 132

Input: John and Sam were hungry so they ordered an extra-large pizza that was pre-sliced into 12 pieces.  John ate 3 slices while Sam ate twice the amount that John ate.  How many slices of pizza were left?
Output: 3

Input: On Thursday the Meat Market sold 210kg of ground beef. On Friday they sold twice that amount. On Saturday they only sold 130kg. On Sunday they sold half of what they sold Saturday. If they originally planned to sell only 500kg, how much meat did they sell beyond their original plans?
Output: 325

Input: There is a massive rainstorm lasting 4 days.  There is an area that collects water to prevent flooding in the area.  It ends up overflowing on the 4th day.  The area can hold the equivalent of 6 feet of rain.  It can also drain out the equivalent of 3 inches of rain per day to the nearby river without causing problems.  The first day it rained 10 inches.  The second day it rained twice that much.  On the third day, it rained 50% more than the second day.  It flooded the fourth day before getting a chance to do any of the draining.  What is the minimum amount it rained on the fourth day?
Output: 21

Input: There are 6 boxes of crayons that hold 8 orange crayons. There are 7 boxes of crayons that have 5 blue crayons. There is 1 box of 11 red crayons. How many crayons are there in total?
Output: 94

Input: Joey studies for his SAT exams 2 hours per night 5 nights a week.  On the weekends, he studies 3 hours a day.  If his SAT exam is 6 weeks away, how much time will Joey spend studying?
Output: 96

Input: Alexis can sew a skirt in 2 hours and a coat in 7 hours. How long does it take for Alexis to sew 6 skirts and 4 coats?
Output: 40

Input: Amy bought a 15-foot spool of string to cut up into wicks for making candles.   If she cuts up the entire string into an equal number of 6-inch and 12-inch wicks, what is the total number of wicks she will have cut?
Output: 20

Input: Jacob loves to build things. In Jacob's toy bin there are 18 red blocks. There are 7 more yellow blocks than red blocks. There are also 14 more blue blocks than red blocks. How many blocks are there in all?
Output: 75

Input: Five adults and two children go to see a movie and buy $12 worth of concessions. The total cost of their trip is $76. If each child's ticket is $7, how much, in dollars, are the adult tickets?
Output: 10

Input: James writes a comic every other day for 4 years. If there was no leap year, how many comics has he written?
Output: 730

Input: On a 16 GB (gigabyte) capacity USB drive, 50% is already busy. Calculate the number of gigabytes still available.
Output: 8

Input: The middle school sold 6 more than two times the number of fair tickets as it did tickets to the baseball game. If 25 fair tickets were sold, how many baseball game tickets did the school sell?
Output: 56

Input: Ethyl bought Lucy two new dolls for her doll collection.  This increased the doll collection by 25%.  After the addition of the two new dolls, how many dolls are in Lucy's collection?
Output: 10

Input: Matthew asked his children how many hotdogs they wanted for dinner.  Both Ella and Emma agreed they wanted 2 hotdogs each.  Luke said he could eat twice the amount of hotdogs as his sisters while Hunter said he could only eat 1 and half times the total amount of his sisters.  How many hotdogs did Matthew need to cook?
Output: 14

Input: A porcelain vase was originally priced at $200 but went on sale for 25% off. If Donna bought the porcelain vase and paid 10% sales tax, how much did she pay in total?
Output: 165

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

