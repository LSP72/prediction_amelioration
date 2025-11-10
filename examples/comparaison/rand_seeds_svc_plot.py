import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

path = "examples/results/svc_vs_svr_rdm_state/random_seeds/"


def get_param(dict, param: str):
    return [dict[key][param] for key in dict.keys()]


def plot_scatter(path, type, cond):
    full_path = path + type + "_" + cond + ".pkl"
    with open(full_path, "rb") as file:
        dict = pkl.load(file)

    seed_name = "random_state_" + type
    seeds = get_param(dict, seed_name)
    acc = get_param(dict, "precision_score")

    mean = np.mean(acc)
    sd = np.std(acc)

    fig = plt.figure()
    plt.scatter(seeds, acc)
    plt.xlabel("seeds")
    plt.ylabel("accuracy")
    plt.ylim([0, 1])
    plt.text(1, 0.2, f"Mean accuracy: {mean:.2f} ± {sd:.2f}")
    plt.title(f"Accuracy vs. seed\nfor {cond}, varying seed: {type}")
    plt.show()


for type in ["split", "cv", "optim"]:
    for cond in ["VIT", "6MWT"]:
        plot_scatter(path, type, cond)
