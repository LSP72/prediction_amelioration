import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

svc_pkl_path = 'examples/results/svc_vs_svr_rdm_state/svc.pkl'
svr_pkl_path = 'examples/results/svc_vs_svr_rdm_state/svr.pkl'

with open(svc_pkl_path, "rb") as file:
    svc_dict = pkl.load(file)

with open(svr_pkl_path, "rb") as file:
    svr_dict = pkl.load(file)

seeds = [svc_dict[key]["seed"] for key in svc_dict.keys()]
svc_acc = [svc_dict[key]["precision_score"]*100 for key in svc_dict.keys()]
svr_acc = [svr_dict[key]["precision_score"]*100 for key in svr_dict.keys()]

idx = np.argsort(seeds, kind='stable')   # indices that sort a ascending
seeds_sorted = np.array(seeds)[idx]
svc_acc_sorted = np.array(svc_acc)[idx]
svr_acc_sorted = np.array(svr_acc)[idx]

x = np.linspace(0, len(seeds), len(seeds), endpoint=False)
plt.scatter(x, svc_acc_sorted, label='svc', color='r')
plt.scatter(x, svr_acc_sorted, label='svr', color='b')
for i in range(len(x)):
    plt.plot([x[i], x[i]], [svc_acc_sorted[i], svr_acc_sorted[i]], color='gray', linestyle='-', alpha=0.6) # line
    diff = svc_acc_sorted[i] - svr_acc_sorted[i]
    y_mid = (svc_acc_sorted[i] + svr_acc_sorted[i]) / 2
    plt.text(x[i] + 0.1, y_mid, f"{diff:.2f}", fontsize=7, color='gray')
plt.xlabel("Seeds")
plt.xticks(x, seeds_sorted, rotation=90)
plt.ylabel('Precision scores (%)')
plt.title('Precision scores vs. seeds \n Speed predictions')
plt.legend()
plt.show()
