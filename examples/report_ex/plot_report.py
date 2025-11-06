import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


def amelio_endurance(end_dict):
    amelio_end = []
    delta_pred = end_dict["svr"][["pre_post_diff"]]
    GMFCS = end_dict["svc"]["GMFCS"]
    GMFCS_MCID = {1: range(4, 29), 2: range(4, 29), 3: range(9, 20), 4: range(10, 28)}
    for i in range(len(delta_pred)):
        amelio_end_min = (delta_pred.iloc[i] / min(GMFCS_MCID[GMFCS.iloc[i]])) * 100
        amelio_end_max = (delta_pred.iloc[i] / max(GMFCS_MCID[GMFCS.iloc[i]])) * 100
        amelio_end.append([amelio_end_min, amelio_end_max])
    return amelio_end


vit_pkl_path = "examples/results/report_ex/VIT.pkl"
end_pkl_path = "examples/results/report_ex/6MWT.pkl"

with open(vit_pkl_path, "rb") as file:
    vit_dict = pkl.load(file)

print(vit_dict)
with open(end_pkl_path, "rb") as file:
    end_dict = pkl.load(file)


amelio_vit = (vit_dict["svr"]["pre_post_diff"] / 0.1) * 100
amelio_end = amelio_endurance(end_dict)
