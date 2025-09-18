import pandas as pd
from amelio_cp import Process
from amelio_cp import SVCModel

# %% Collecting/Loading the data from a csv file already created
file_path = "dataset/sample_1/all_data.csv"

all_data = Process.load_csv(file_path)

# %% Features and labels extraction
VIT_POST = all_data["VIT_POST"]
END_POST = all_data["6MWT_POST"]
data = all_data.drop(columns=["VIT_POST", "6MWT_POST"])

delta_VIT = Process.calculate_MCID(all_data, "VIT")

# delta_END = all_data['6MWT_POST'] - all_data['6MWT_PRE']

# %% Training the models
SVC_VIT = SVCModel()
SVC_VIT.add_data(data, delta_VIT)
SVC_VIT.train_and_tune("random", n_iter=100)

print(SVC_VIT.best_params_)
