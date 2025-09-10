#%% PLOT TRUE VS. PRED
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_hat, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45° line
plt.xlabel("True Δv")
plt.ylabel("Predicted Δv")
plt.title("Predictions vs True Values")
plt.show()

#%% PLOT CORR MAP
corr_matrix_pearson = data.corr(method='pearson')
corr_matrix_spearman = data.corr(method='spearman')
corr_matrix_kendall = data.corr(method='kendall')

plt.figure(figsize=(16,14))
sns.heatmap(corr_matrix_pearson, cmap = 'coolwarm', annot = True, annot_kws={"size": 12}, center = 0, fmt = ' .1g', square=True)
# linewidths=0.5 (pour espacer les cases) //  cbar_kws={"shrink": 0.8} (pour réduire la bar de légend)
plt.title("Correlation Heatmap: Pearson", fontsize=20)
plt.xlabel("Features", fontsize=16)
plt.ylabel("Features", fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.yticks(rotation=0)              # Keep y-axis labels horizontal
plt.show()

plt.figure(figsize=(16,14))
sns.heatmap(corr_matrix_spearman, cmap = 'coolwarm', annot = True, annot_kws={"size": 12}, center = 0, fmt = ' .1g', square=True)
# linewidths=0.5 (pour espacer les cases) //  cbar_kws={"shrink": 0.8} (pour réduire la bar de légend)
plt.title("Correlation Heatmap: Spearman", fontsize=20)
plt.xlabel("Features", fontsize=16)
plt.ylabel("Features", fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.yticks(rotation=0)              # Keep y-axis labels horizontal
plt.show()

plt.figure(figsize=(16,14))
sns.heatmap(corr_matrix_kendall, cmap = 'coolwarm', annot = True, annot_kws={"size": 12}, center = 0, fmt = ' .1g', square=True)
# linewidths=0.5 (pour espacer les cases) //  cbar_kws={"shrink": 0.8} (pour réduire la bar de légend)
plt.title("Correlation Heatmap: Kendall", fontsize=20)
plt.xlabel("Features", fontsize=16)
plt.ylabel("Features", fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.yticks(rotation=0)              # Keep y-axis labels horizontal
plt.show()
