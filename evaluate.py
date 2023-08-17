from metric import rouge, compute_mc_acc, compute_rouge
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import json

# get all the folder names in the directory
def get_folders(path):
    folders = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            folders.append(folder)
    return folders


def compute_metric(result_path: str, model_path: str, data_path: str, data_name: str, metric_fn: callable):
    folders = get_folders(result_path)

    results = {"num_demonstrations": [], "seed": [], "rationales": [], "tokens":[], "acc": [], "max_prompt_len": []}
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    for folder in folders:
        results["num_demonstrations"].append(int(folder.split("-")[0][1:]))
        results["seed"].append(int(folder.split("-s")[-1].split("-")[0]))
        if "True" in folder:
            results["rationales"].append("With Rationales")
        else:
            results["rationales"].append("Without Rationales")
        results['acc'].append(metric_fn(os.path.join(result_path, folder)))

        if "-m" in folder:
            results["max_prompt_len"].append(int(folder.split("-m")[-1]))
        else:
            results["max_prompt_len"].append(2048) # default max prompt length

        with open(os.path.join(data_path, folder.split("-m")[0], f"{data_name}.jsonl"), "r") as f:
            data = f.readlines()[0]
        demonstrations = json.loads(data)['prompt']
        # demonstrations = demonstrations.split("### Demonstration:")[1].split("### Input:")[0].strip()
        demonstrations_ids = tokenizer(demonstrations, return_tensors="pt")
        results["tokens"].append(demonstrations_ids["input_ids"].shape[1])

    df = pd.DataFrame(results)
    df.sort_values(by=["num_demonstrations", "seed"], inplace=True, ignore_index=True)

    return df

def demo_plot(results: pd.DataFrame, title: str, y_label="Accuracy", limits=None, show_tokens=True):
    """Lineplot: x axis is the number of demonstrations and the y axis is the accuracy
       plot the graph in a professional way for academic paper
    """
    results = results[results["num_demonstrations"] <= limits] if limits else results
    results_tokens = results.groupby(["num_demonstrations", "rationales"]).mean().reset_index()
    results_tokens['tokens'] = results_tokens['tokens'].apply(lambda x: int(np.round(x)))
    sns.set(style="darkgrid", font_scale=1.5)
    plt.figure(figsize=(14, 6))

    if len(results['max_prompt_len'].unique()) == 1:
        ax = sns.lineplot(x="num_demonstrations", y="acc", 
                          hue="rationales", style="rationales",
                          data=results, markers=True, dashes=False, 
                          markersize=10
                          )
    else:
        ax = sns.lineplot(x="num_demonstrations", y="acc", 
                          hue="rationales", style="max_prompt_len",
                          data=results, markers=True, dashes=False,
                          markersize=10
                         )

    if show_tokens:
        # label each point with the number of tokens
        for i in range(len(results_tokens)):
            ax.text(results_tokens.iloc[i]['num_demonstrations'], 
                    results_tokens.iloc[i]['acc'], 
                    results_tokens.iloc[i]['tokens'], 
                    horizontalalignment='left', 
                    size='medium', 
                    color='black', 
                    weight='semibold')

    ax.set_title(title)
    ax.set_ylabel(y_label)
    plt.show()

def token_plot(results: pd.DataFrame, title: str, y_label="Accuracy"):
    """
    draw smoothed line plot for each rationales: x axis is the number of tokens in the demonstration and the y axis is the accuracy
    """
    sns.set(style="darkgrid", font_scale=1.5)
    plt.figure(figsize=(14, 6))
    if len(results['max_prompt_len'].unique()) == 1:
        ax = sns.lineplot(x="tokens", y="acc", 
                          hue="rationales", style="rationales",
                          data=results, markers=True, dashes=False, 
                          markersize=10
                          )
    else:
        ax = sns.lineplot(x="tokens", y="acc", 
                          hue="rationales", style="max_prompt_len",
                          data=results, markers=True, dashes=False,
                          markersize=10
                         )
    ax.set_title(title)
    ax.set_ylabel(y_label)
    plt.show()

result_path = "./results/llama2-7b/commonsenseqa/out-domain"
model_path = "/scratch/ylu130/model/llama-2-7b/"
data_path = "/scratch/ylu130/processed_data/commonsenseqa/out-domain"
acc_results1 = compute_metric(result_path, model_path, data_path, "commonsenseqa", compute_mc_acc)

demo_plot(acc_results1, "Out-domain Commonsenseqa Accuracy", limits=9)
token_plot(acc_results1, "Out-domain Commonsenseqa Accuracy")


result_path = "./results/llama2-7b/commonsenseqa/in-domain"
model_path = "/scratch/ylu130/model/llama-2-7b/"
data_path = "/scratch/ylu130/processed_data/commonsenseqa/in-domain"
acc_results2 = compute_metric(result_path, model_path, data_path, "commonsenseqa", compute_mc_acc)

demo_plot(acc_results2, "In-domain Commonsenseqa Accuracy", limits=9)
token_plot(acc_results2, "In-domain Commonsenseqa Accuracy")

# combine the results
acc_results1['num_demonstrations'] = acc_results1['num_demonstrations'].apply(lambda x: -x)
acc_results = pd.concat([acc_results1, acc_results2], ignore_index=True)
demo_plot(acc_results, "Combined Commonsenseqa Accuracy", limits=9, show_tokens=False)

result_path = "./results/llama2-7b/gsm8k/out-domain"
model_path = "/scratch/ylu130/model/llama-2-7b/"
data_path = "/scratch/ylu130/processed_data/gsm8k/out-domain"
acc_results3 = compute_metric(result_path, model_path, data_path, "gsm8k", compute_mc_acc)

demo_plot(acc_results3[acc_results3['max_prompt_len'] == 2048], "Out-domain GSM8K Accuracy", limits=9)
token_plot(acc_results3[acc_results3['max_prompt_len'] == 2048], "Out-domain GSM8K Accuracy")

demo_plot(acc_results3, "Out-domain GSM8K Accuracy", limits=9)
token_plot(acc_results3, "Out-domain GSM8K Accuracy")



# Note: RuntimeError: probability tensor contains either `inf`, `nan` or element < 0 -> Probability distribution has been messed

