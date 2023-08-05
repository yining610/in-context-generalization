import json

# read commonsenseqa.jsonl
n = 0
with open("/scratch/ylu130/processed_data/commonsenseqa/n3-seed42-rationalsTrue/commonsenseqa.jsonl", "r") as f:
    for line in f.readlines():
        line = json.loads(line)
        print(line["input_with_demonstration"])
        print(line["input_without_demonstration"])
        print(line["output"])
        n+=1

        if n == 10:
            break