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

def plot(results: pd.DataFrame, title: str, y_label="Accuracy"):
    # using seaborn to draw the lineplot, where the x axis is the number of demonstrations and the y axis is the accuracy
    # draw in an professional style used for academic paper
    sns.set(style="darkgrid", font_scale=1.5)
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(x="num_demonstrations", y="acc", 
                      hue="rationales", style="rationales",
                      data=results, markers=True, dashes=False
                      )
    ax.set_title(title)
    ax.set_ylabel(y_label)
    plt.show()


def compute_metric(path, metric_fn):
    folders = get_folders(path)

    results = {"num_demonstrations": [], "seed": [], "acc": [], "rationales": []}

    for folder in folders:
        results["num_demonstrations"].append(int(folder.split("-")[0][1:]))
        results["seed"].append(int(folder.split("-")[-2][1:]))
        if "True" in folder:
            results["rationales"].append("With Rationales")
        else:
            results["rationales"].append("Without Rationales")
        results['acc'].append(metric_fn(os.path.join(path, folder)))

    df = pd.DataFrame(results)
    df.sort_values(by=["num_demonstrations", "seed"], inplace=True, ignore_index=True)

    return df

path = "./results/llama2-7b/commonsenseqa/in-domain"
acc_results = compute_metric(path, compute_mc_acc)
plot(acc_results, "In-domain CommonsenseQA Accuracy")


num_demonstrations, with_rationales_rougeL, without_rationales_rougeL = compute_metric(path, compute_rouge)
plot(num_demonstrations, with_rationales_rougeL, without_rationales_rougeL, "Out-domain CommonsenseQA ROUGE", y_label="ROUGE")

# Note: RuntimeError: probability tensor contains either `inf`, `nan` or element < 0 -> Probability distribution has been messed up


x = "o1-gsm8k-s2-rTrue"
x.split("-")[0][1]