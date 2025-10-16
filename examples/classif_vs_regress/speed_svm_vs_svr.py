import pandas as pd
from amelio_cp import Process
from amelio_cp import SVCModel
from amelio_cp import SVRModel
from amelio_cp import ClassifierMetrics
from sklearn.metrics import classification_report


# %% Loading the data from an existing .xlsx
file_path = "datasets/sample_2/all_data_28pp.csv"

all_data = Process.load_csv(file_path)

# %% Feature engineering
features = pd.read_excel("amelio_cp/processing/Features.xlsx")
selected_feature = features["19"].dropna().to_list()
features_names = features["19_names"].dropna().to_list()

y_vit = all_data["VIT_POST"]
y_end = all_data["6MWT_POST"]
delta_vit = Process.calculate_MCID(all_data, "VIT")
delta_end = Process.calculate_MCID(all_data, "6MWT")

data_vit = all_data.drop(["VIT_POST"], axis=1)
data_vit.dropna()
data_vit = data_vit[selected_feature]
print("Number of participants for speed predictions:", data_vit.shape[0])
data_end = all_data.drop(["6MWT_POST"], axis=1)
data_end.dropna()
data_end = all_data[selected_feature]
print("Number of participants for endurance (6MWT) predictions:", data_end.shape[0])

# %% SPEED
# Classification
SVC_vit = SVCModel()
SVC_vit.add_data(data_vit, delta_vit, 0.2)
SVC_vit.train_and_tune("bayesian")
print("Best parameters found for speed classification model: \n", SVC_vit.best_params)

y_pred_vit_classif = SVC_vit.model.predict(SVC_vit.X_test_scaled)
print("Accuracy test score: ", SVC_vit.model.score(SVC_vit.X_test_scaled, SVC_vit.y_test))
print(classification_report(SVC_vit.y_test, y_pred_vit_classif), flush=True)

print("-- Confusion matrix for speed classification --")
ClassifierMetrics.conf_matrix(
    SVC_vit.y_test,
    y_pred_vit_classif,
    class_names=["Non-Responder", "Responder"],
    title="Confusion Matrix for speed classification",
)

# Regression
SVR_vit = SVRModel()
SVR_vit.add_data(data_vit, y_vit, 0.2)
SVR_vit.train_and_tune("bayesian")
print("Best parameters found for speed regression model: \n", SVR_vit.best_params)

y_pred_vit_reg = SVR_vit.model.predict(SVC_vit.X_test_scaled)
print("R² set score: ", SVR_vit.model.score(SVR_vit.X_test_scaled, SVR_vit.y_test))

# Classification from the regression results
delta_VIT = [
    1 if y_pred_vit_reg[i] - SVR_vit.X_test["VIT_PRE"].iloc[i] > 0.1 else 0 for i in range(len(y_pred_vit_reg))
]
delta_VIT_true = [
    1 if SVR_vit.y_test.iloc[i] - SVR_vit.X_test["VIT_PRE"].iloc[i] > 0.1 else 0 for i in range(len(SVR_vit.X_train))
]

ClassifierMetrics.conf_matrix(
    SVR_vit.y_test,
    y_pred_vit_reg,
    class_names=["Non-Responder", "Responder"],
    title="Confusion Matrix for speed classification",
)
