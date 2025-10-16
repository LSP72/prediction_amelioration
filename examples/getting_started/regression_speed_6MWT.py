import pandas as pd
from amelio_cp import Process
from amelio_cp import SVRModel
from amelio_cp import ClassifierMetrics
from amelio_cp import SHAPPlots
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# %% Collecting/Loading the data from a csv file already created

# TODO: relative path to fix
file_path = (
    "/Users/mathildetardif/Documents/Python/Biomarkers/prediction_amelioration/datasets/sample_2/all_data_28pp.csv"
)

all_data = Process.load_csv(file_path)

# %% Feature selection
features = pd.read_excel(
    "/Users/mathildetardif/Documents/Python/Biomarkers/prediction_amelioration/amelio_cp/processing/Features.xlsx"
)
selected_features = features["19"].dropna().to_list()
features_names = features["19_names"]

# %% Dealing with the gait speed

# TODO: fonction to do in the Process file
all_data_VIT = all_data.drop(["6MWT_POST"], axis=1)
all_data_VIT = all_data_VIT.dropna()

VIT_POST = all_data_VIT["VIT_POST"]
selected_data_VIT = all_data_VIT[selected_features]

print("Number of participants for VIT classification:", all_data_VIT.shape[0])
print(selected_data_VIT.columns)

# %% Extracting data about 6MWT
all_data_6MWT = all_data.drop(["VIT_POST"], axis=1)
all_data_6MWT = all_data_6MWT.dropna()
delta_6MWT = Process.calculate_MCID(all_data_6MWT, "6MWT")

data_6MWT = all_data_6MWT.drop(["6MWT_POST"], axis=1)
selected_data_6MWT = data_6MWT[selected_features]

print("Number of participants for 6MWT classification:", data_6MWT.shape[0])
print(selected_data_6MWT.columns)

# %% Training the model for vitesse classification

SVR_VIT = SVRModel()
SVR_VIT.add_data(selected_data_VIT, VIT_POST, 0.2)
SVR_VIT.train_and_tune("bayesian_optim", n_iter=100)
print("Best parameters found for speed SVC:", SVR_VIT.best_params, flush=True)

# Predictions on the test set
y_pred_VIT = SVR_VIT.model.predict(SVR_VIT.X_test_scaled)
print("R² set score: ", SVR_VIT.model.score(SVR_VIT.X_test_scaled, SVR_VIT.y_test))

# Confusion matrix
delta_VIT = [1 if y_pred_VIT[i] - SVR_VIT.X_train["VIT_PRE"].iloc[i] > 0.1 else 0 for i in range(len(y_pred_VIT))]
delta_VIT_true = [
    1 if SVR_VIT.y_test[i] - SVR_VIT.X_train["VIT_PRE"].iloc[i] > 0.1 else 0 for i in range(len(SVR_VIT.X_train))
]

ClassifierMetrics.conf_matrix(
    SVR_VIT.y_test,
    y_pred_VIT,
    class_names=["Non-Responder", "Responder"],
    title="Confusion Matrix for speed classification",
)

# SHAP analysis
SVR_VIT.shap_analysis = SHAPPlots.shap_values_calculation(SVR_VIT)

# Shap plots
SHAPPlots.plot_shap_summary(SVR_VIT, selected_features)
SHAPPlots.plot_shap_bar(SVR_VIT, selected_features)


# %% Training the model for 6MWT classification
x_train_6MWT, x_test_6MWT, y_train_6MWT, y_test_6MWT = train_test_split(
    selected_data_6MWT, delta_6MWT, test_size=0.2, random_state=42
)

SVR_6MWT = SVRModel()
SVR_6MWT.add_data(x_train_6MWT, y_train_6MWT)
SVR_6MWT.train_and_tune("bayesian_optim", n_iter=100)

print("Best parameters found for speed SVR:", SVR_6MWT.best_params)

# Predictions on the test set
y_pred_6MWT = SVR_6MWT.model.predict(SVR_6MWT.X_test_scaled)
print("R² set score: ", SVR_6MWT.model.score(SVR_6MWT.X_test_scaled, SVR_6MWT.y_test))

# Confusion matrix
delta_6MWT = [1 if y_pred_6MWT[i] - SVR_6MWT.X_train["6MWT_PRE"].iloc[i] > 0.1 else 0 for i in range(len(y_pred_VIT))]
delta_6MWT_true = [
    1 if SVR_6MWT.y_test[i] - SVR_VIT.X_train["6MWT_PRE"].iloc[i] > 0.1 else 0 for i in range(len(SVR_6MWT.X_train))
]

ClassifierMetrics.conf_matrix(
    SVR_6MWT.y_test,
    y_pred_6MWT,
    class_names=["Non-Responder", "Responder"],
    title="Confusion Matrix for speed classification",
)

# SHAP analysis
SVR_6MWT.shap_analysis = SHAPPlots.shap_values_calculation(SVR_6MWT, "svr", x_train_6MWT, x_test_6MWT)
