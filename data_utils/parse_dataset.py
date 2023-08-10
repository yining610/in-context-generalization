def parse(data_name, line):
    if data_name == "commonsenseqa":
        question = line["question"]["stem"] + " Choices: "
        for choice in line["question"]["choices"]:
            question = question + " " + choice['label'] + ": " + choice["text"]
        gold_answer = line["answerKey"] + ": " + line["question"]["choices"][ord(line["answerKey"]) - ord("A")]["text"]
        question_with_answer = question + " The answer is " + gold_answer
        rationales = None
    elif data_name == "gsm8k":
        question = line["question"]
        gold_answer = line["answer"].split("####")[1]
        question_with_answer = question + " The answer is " + gold_answer
        rationales = line["answer"]
    
    return question, gold_answer, question_with_answer, rationales
