from amelio_cp import Process
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data_path = "datasets/sample_2/all_data_28pp.csv"
all_data = Process.load_csv(data_path)

all_data.drop(columns=['Unnamed: 0'], inplace=True)

n_rows, n_cols = 5, 6
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
axes = axes.flatten()

#TODO: adding more details
for i, col in enumerate(all_data.columns):
    axes[i].boxplot(all_data[col])
    axes[i].set_title(col, fontsize=9)
    axes[i].tick_params(axis='x', labelbottom=False)  # hide x labels for neatness

# Hide any unused subplots if df has fewer than grid size
for j in range(len(all_data.columns), n_rows*n_cols):
    fig.delaxes(axes[j])

plt.suptitle("Variability Across Features", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()