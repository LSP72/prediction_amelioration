import pandas as pd
from amelio_cp import Process
from amelio_cp import SVCModel
from amelio_cp import ClassifierMetrics
from sklearn.metrics import classification_report

# %% Collecting/Loading the data from a csv file already created
file_path = "datasets/sample_2/all_data_28pp.csv"

all_data = Process.load_csv(file_path)

# %% Feature selection
features = pd.read_excel("amelio_cp/processing/Features.xlsx")
selected_features = features['19'].dropna().to_list()
features_names = features['19_names'].dropna().to_list()

# %% Features and labels extraction
y_vit = all_data["VIT_POST"]
delta_vit = Process.calculate_MCID(all_data, "VIT")
all_data_vit = all_data.drop(['6MWT_POST'], axis=1)
all_data_vit = all_data_vit.dropna()

data_vit = all_data_vit[selected_features]
print("Number of participants for speed classification:", data_vit.shape[0])
print(data_vit.columns)

# %% Training the models

SVC_vit = SVCModel()
SVC_vit.random_state=42
SVC_vit.add_data(data_vit, delta_vit, 0.2)
SVC_vit.train_and_tune("bayesian_optim")
print("Best parameters found for speed classification model: \n", SVC_vit.best_params)

y_pred_vit_classif = SVC_vit.model.predict(SVC_vit.X_test_scaled)
print("Accuracy test score: ", SVC_vit.model.score(SVC_vit.X_test_scaled, SVC_vit.y_test))
print(classification_report(SVC_vit.y_test, y_pred_vit_classif), flush=True)

output_path = "examples/results/svc_vs_svr_rdm_state/"
ClassifierMetrics.conf_matrix(SVC_vit, SVC_vit.y_test, y_pred_vit_classif, class_names=["Non-Responder", "Responder"], title="Confusion Matrix for speed classification", output_path=output_path)

