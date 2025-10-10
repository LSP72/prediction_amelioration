import pandas as pd
from amelio_cp import Process
from amelio_cp import SVRModel
from amelio_cp import ClassifierMetrics

# %% Collecting/Loading the data from a csv file already created

file_path = "datasets/sample_2/all_data_28pp.csv"

all_data = Process.load_csv(file_path)

# %% Feature selection
features = pd.read_excel("amelio_cp/processing/Features.xlsx")
selected_features = features['19'].dropna().to_list()
features_names = features['19_names'].dropna().to_list()

# %% Dealing with the gait speed
all_data_vit = all_data.drop(['6MWT_POST'], axis=1)
all_data_vit = all_data_vit.dropna()
y_vit = all_data["VIT_POST"]

data_vit = all_data_vit[selected_features]

print("Number of participants for speed regression:", data_vit.shape[0])
print(data_vit.columns)

# %% Model training

SVR_vit = SVRModel()
SVR_vit.random_state=42
SVR_vit.add_data(data_vit, y_vit, 0.2)
SVR_vit.train_and_tune("bayesian_optim")
print("Best parameters found for speed regression model: \n", SVR_vit.best_params)

y_pred_vit_reg = SVR_vit.model.predict(SVR_vit.X_test_scaled)
print("R² set score: ", SVR_vit.model.score(SVR_vit.X_test_scaled, SVR_vit.y_test))

# Classification from the regression results
delta_VIT = [1 if y_pred_vit_reg[i] - SVR_vit.X_test['VIT_PRE'].iloc[i] > 0.1 else 0 for i in range(len(y_pred_vit_reg))]
delta_VIT_true = [1 if SVR_vit.y_test.iloc[i] - SVR_vit.X_test['VIT_PRE'].iloc[i] > 0.1 else 0 for i in range(len(SVR_vit.y_test))]

output_path = "examples/results/svc_vs_svr_rdm_state/"
ClassifierMetrics.conf_matrix(SVR_vit, delta_VIT_true, delta_VIT, class_names=["Non-Responder", "Responder"], title="Confusion Matrix for speed regression", output_path=output_path)
