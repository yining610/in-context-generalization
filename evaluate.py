from metric import rouge, compute_mc_acc, compute_math_acc
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

def get_results(result_path: str, model_path: str, data_path: str, data_name: str, metric_fn: callable) -> pd.DataFrame:
    folders = get_folders(result_path)

    results = {"num_demonstrations": [], "seed": [], "rationales": [], "tokens":[], "acc": [], "max_prompt_len": []}
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/scratch/ylu130/model-hf/")
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

    # if the max prompt length is the same, then we can plot the graph with style as hue
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

base_path = "./results/llama2-7b"
base_path_2 = "/scratch/ylu130/processed_data"
model_path = "meta-llama/Llama-2-7b-hf"
# CommonsenseQA out-domain results
acc_results1 = get_results(result_path=os.path.join(base_path, "commonsenseqa/out-domain"), 
                           model_path=model_path, 
                           data_path=os.path.join(base_path_2, "commonsenseqa/out-domain"), 
                           data_name="commonsenseqa", 
                           metric_fn=compute_mc_acc)

demo_plot(acc_results1, "Out-domain Commonsenseqa Accuracy", limits=9)
token_plot(acc_results1, "Out-domain Commonsenseqa Accuracy")

# CommonsenseQA in-domain results
acc_results2 = get_results(result_path=os.path.join(base_path, "commonsenseqa/in-domain"), 
                           model_path=model_path, 
                           data_path=os.path.join(base_path_2, "commonsenseqa/in-domain"), 
                           data_name="commonsenseqa", 
                           metric_fn=compute_mc_acc)

demo_plot(acc_results2, "In-domain Commonsenseqa Accuracy", limits=9)
token_plot(acc_results2, "In-domain Commonsenseqa Accuracy")

# CommonsenseQA overall results
acc_results1['num_demonstrations'] = acc_results1['num_demonstrations'].apply(lambda x: -x)
acc_results1['tokens'] = acc_results1['tokens'].apply(lambda x: -x)
acc_results = pd.concat([acc_results1, acc_results2], ignore_index=True)

zero_demo_temp = acc_results[acc_results['num_demonstrations'] == 0].copy()
zero_demo_temp['rationales'] = "Without Rationales"
acc_results = pd.concat([acc_results, zero_demo_temp], ignore_index=True)

demo_plot(acc_results, "Commonsenseqa Overall Accuracy", limits=9, show_tokens=False)
token_plot(acc_results, "Commonsenseqa Overall Accuracy")

# GSM8K out-domain results
acc_results3 = get_results(result_path=os.path.join(base_path, "gsm8k/out-domain"),
                            model_path=model_path,
                            data_path=os.path.join(base_path_2, "gsm8k/out-domain"),
                            data_name="gsm8k",
                            metric_fn=compute_math_acc)

# demo_plot(acc_results3[acc_results3['max_prompt_len'] == 2048], "Out-domain GSM8K Accuracy", limits=9)
# token_plot(acc_results3[acc_results3['max_prompt_len'] == 2048], "Out-domain GSM8K Accuracy")

demo_plot(acc_results3, "Out-domain GSM8K Accuracy", limits=9)
token_plot(acc_results3, "Out-domain GSM8K Accuracy")

# LLAMA2-13B
base_path = "./results/llama2-13b"
base_path_2 = "/scratch/ylu130/processed_data"
model_path = "meta-llama/Llama-2-13b-hf"
# CommonsenseQA out-domain results
acc_results4 = get_results(result_path=os.path.join(base_path, "commonsenseqa/out-domain"), 
                           model_path=model_path, 
                           data_path=os.path.join(base_path_2, "commonsenseqa/out-domain"), 
                           data_name="commonsenseqa", 
                           metric_fn=compute_mc_acc)

demo_plot(acc_results4, "Out-domain Commonsenseqa Accuracy", limits=9)
token_plot(acc_results4, "Out-domain Commonsenseqa Accuracy")
