import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

vit_pkl_path = "examples/results/svc_vs_svr_rdm_state/svr_VIT_gains.pkl"
endurance_pkl_path = "examples/results/svc_vs_svr_rdm_state/svr_6MWT_gains.pkl"

with open(vit_pkl_path, "rb") as file:
    vit_dict = pkl.load(file)

with open(endurance_pkl_path, "rb") as file:
    endurance_dict = pkl.load(file)

def get_param(dict, param:str):
    return [dict[key][param]for key in dict.keys()]

vit_r2 = get_param(vit_dict, "r2")
endurance_r2 = get_param(endurance_dict, "r2")
seeds = get_param(vit_dict, "seed") # random_state now

vit_mean = np.average(vit_r2)
endurance_mean = np.average(endurance_r2)

idx = np.argsort(seeds, kind="stable")  # indices that sort a ascending
seeds_sorted = np.array(seeds)[idx]
vit_r2_sorted = np.array(vit_r2)[idx]
endurance_r2_sorted = np.array(endurance_r2)[idx]
x_index = np.linspace(1,len(seeds), len(seeds))

fig = plt.figure()
plt.scatter(x_index, vit_r2_sorted, label="speed", color='blue')
plt.scatter(x_index, endurance_r2_sorted, label="6mwt", color='red')
plt.xticks(x_index, seeds_sorted)
plt.xlabel("seeds")
plt.ylabel("r2")
plt.text(3, -0.8, f"Speed mean: {vit_mean:.3f}")
plt.text(3, -1, f"6mwt mean: {endurance_mean:.3f}")
plt.legend()
plt.show()
