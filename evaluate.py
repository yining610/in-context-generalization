from metric import rouge, compute_mc_acc
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

# outdomain
path = "./results/llama2-7b/commonsenseqa/out-domain"
folders = get_folders(path)

with_rationales_acc, without_rationales_acc = [], []

for folder in folders:
    if "True" in folder:
        with_rationales_acc.append((int(folder.split("-")[0][1]),compute_mc_acc(os.path.join(path, folder))))
    else:
        without_rationales_acc.append((int(folder.split("-")[0][1]),compute_mc_acc(os.path.join(path, folder))))

with_rationales_acc.sort(key=lambda x: x[0])
without_rationales_acc.sort(key=lambda x: x[0])
num_demonstrations = [x[0] for x in with_rationales_acc]
with_rationales_acc = [x[1] for x in with_rationales_acc]
without_rationales_acc = [x[1] for x in without_rationales_acc]


# Sample data
x = [1, 2, 3, 4, 5]
y1 = [10, 15, 8, 12, 20]
y2 = [5, 8, 6, 10, 15]

# Set Seaborn style and color palette
sns.set(style="whitegrid", palette="deep")

# Create a line plot with markers
plt.figure(figsize=(8, 6))
sns.lineplot(x=num_demonstrations, y=with_rationales_acc, marker="o", label="with rationales")
sns.lineplot(x=num_demonstrations, y=without_rationales_acc, marker="s", label="without rationales")

# Set plot labels and title
plt.xlabel("Number of Demonstrations")
plt.ylabel("Accuracy")
plt.title("CommonsenseQA Out-Domain Accuracy")

# Show legend
plt.legend()

# Show blue grid background
sns.despine(left=True)
plt.grid(True, color='0.7', linestyle='--', linewidth=1)

# Show the plot
plt.show()




