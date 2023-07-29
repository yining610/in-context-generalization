import json

# read the tellmewhy dataset
def read_tellmewhy(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    narratives = []
    questions = []
    answers = []
    is_answerables = []
    for item in data:
        narratives.append(item['narrative'])
        questions.append(item['question'])
        answers.append(item['answer'])
        is_answerables.append(item['is_ques_answerable'])
    
    return narratives, questions, answers, is_answerables

