import json
import os
from rouge_score import rouge_scorer
import numpy as np
from typing import *

default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def rouge(prediction, ground_truth, xlingual=False):
    scorer = default_rouge_scorer
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure

def match_multiplechoice_answer(answer, text):
    if text is None:
        return False
    answer_label = answer.split(":")[0].strip().lower()
    if answer_label == "choice1":
        answer_label = "a"
    elif answer_label == "choice2":
        answer_label = "b"
    elif answer_label == "choice3":
        answer_label = "c"
    elif answer_label == "choice4":
        answer_label = "d"
    elif answer_label == "choice5":
        answer_label = "e"

    answer_text = answer.split(":")[1].strip().lower().replace(" ", "")

    texts = text.split(":")
    text_label = texts[0].strip().lower().split(" ")[-1].strip().lower()
    if text_label == "choice1":
        text_label = "a"
    elif text_label == "choice2":
        text_label = "b"
    elif text_label == "choice3":
        text_label = "c"
    elif text_label == "choice4":
        text_label = "d"
    elif text_label == "choice5":
        text_label = "e"
    elif text_label == "1":
        text_label = "a"
    elif text_label == "2":
        text_label = "b"
    elif text_label == "3":
        text_label = "c"
    elif text_label == "4":
        text_label = "d"
    elif text_label == "5":
        text_label = "e"
    
    if answer_label == text_label:
        return True
    
    if answer_text in text.strip().lower().replace(" ", ""):
        return True

    return False

    # answer = answer.strip().lower()
    # text = text.strip().lower()
    # if answer in text or answer.replace(" ", "") in text:
    #     return True
    # return False

def compute_mc_acc(path):
    if os.path.exists(os.path.join(path, "answers.jsonl")):
        with open(os.path.join(path, "answers.jsonl"), "r") as f:
            lines = f.readlines()
        results = [match_multiplechoice_answer(json.loads(line)["answer"], json.loads(line)["text"]) for line in lines]
        return sum(results) / len(results)
    else:
        return 0

def compute_math_acc(path):
    if os.path.exists(os.path.join(path, "answers.jsonl")):
        with open(os.path.join(path, "answers.jsonl"), "r") as f:
            lines = f.readlines()
        results = [json.loads(line)["answer"] in json.loads(line)["text"] for line in lines]
        return sum(results) / len(results)
    else:
        return 0

def compute_rouge(path):
    if os.path.exists(os.path.join(path, "answers.jsonl")):
        with open(os.path.join(path, "answers.jsonl"), "r") as f:
            answers = f.readlines()
        preds = [json.loads(x)["text"] for x in answers]
        answers = [json.loads(x)["answer"] for x in answers]
        return np.mean([rouge(pred, answer) for pred, answer in zip(preds, answers)])
    else:
        return 0
