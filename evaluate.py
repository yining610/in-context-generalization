from metric import rouge, compute_mc_acc, compute_rouge
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import pandas as pd

# get all the folder names in the directory
def get_folders(path):
    folders = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folders.append(folder)
    return folders

def plot(x, y1, y2, title, y_label="Accuracy"):
    # Set Seaborn style and color palette
    sns.set(style="whitegrid", palette="deep")

    # Create a line plot with markers
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=x, y=y1, marker="o", label="with rationales")
    sns.lineplot(x=x, y=y2, marker="s", label="without rationales")

    # Set plot labels and title
    plt.xlabel("Number of Demonstrations")
    plt.ylabel(y_label)
    plt.title(title)

    # Show legend
    plt.legend()

    # Show blue grid background
    sns.despine(left=True)
    plt.grid(True, color='0.8', linestyle='--', linewidth=1)

    # Show the plot
    plt.show()

def compute_metric(path, metric_fn):
    folders = get_folders(path)

    with_rationales, without_rationales = [], []

    for folder in folders:
        if "True" in folder:
            with_rationales.append((int(folder.split("-")[0][1]),metric_fn(os.path.join(path, folder))))
        else:
            without_rationales.append((int(folder.split("-")[0][1]),metric_fn(os.path.join(path, folder))))

    with_rationales.sort(key=lambda x: x[0])
    without_rationales.sort(key=lambda x: x[0])

    num_demonstrations = [x[0] for x in with_rationales]
    with_rationales = [x[1] for x in with_rationales]
    without_rationales = [x[1] for x in without_rationales]

    if "in-domain" in path:
        without_rationales.insert(0, with_rationales[0])

    return num_demonstrations, with_rationales, without_rationales

path = "./results/llama2-7b/commonsenseqa/out-domain"
num_demonstrations, with_rationales_acc, without_rationales_acc = compute_metric(path, compute_mc_acc)
plot(num_demonstrations, with_rationales_acc, without_rationales_acc, "Out-domain CommonsenseQA Accuracy")


num_demonstrations, with_rationales_rougeL, without_rationales_rougeL = compute_metric(path, compute_rouge)
plot(num_demonstrations, with_rationales_rougeL, without_rationales_rougeL, "Out-domain CommonsenseQA ROUGE", y_label="ROUGE")