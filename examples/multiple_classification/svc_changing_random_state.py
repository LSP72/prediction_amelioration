import pandas as pd
from amelio_cp import Process
from amelio_cp import SVCModel
from amelio_cp import ClassifierMetrics
from amelio_cp import SHAPPlots
from sklearn.metrics import classification_report
import random as rdm

# %% Loading the data from an existing .xlsx
file_path = "datasets/sample_2/all_data_28pp.csv"

all_data = Process.load_csv(file_path)

# %% Feature engineering
features = pd.read_excel("amelio_cp/processing/Features.xlsx")
selected_feature = features["19"].dropna().to_list()
features_names = features["19_names"].dropna().to_list()

y_vit = all_data["VIT_POST"]
delta_vit = Process.calculate_MCID(all_data, "VIT")

data_vit = all_data.drop(["VIT_POST"], axis=1)
data_vit.dropna()
data_vit = data_vit[selected_feature]
print("Number of participants for speed predictions:", data_vit.shape[0])

# %% Running classification with different random state
# Classification
for i in range(10):
    rdm_state = rdm.randint(0, 100)
    print(rdm_state)
    SVC_vit = SVCModel()
    SVC_vit.random_state = rdm_state
    SVC_vit.add_data(data_vit, delta_vit, 0.2)
    SVC_vit.train_and_tune("bayesian")
    print("Best parameters found for speed classification model: \n", SVC_vit.best_params)

    y_pred_vit_classif = SVC_vit.model.predict(SVC_vit.X_test_scaled)
    print("Accuracy test score: ", SVC_vit.model.score(SVC_vit.X_test_scaled, SVC_vit.y_test))
    print(classification_report(SVC_vit.y_test, y_pred_vit_classif), flush=True)

    output_path = "examples/results/rdm_state_changing/"

    print("-- Confusion matrix for speed classification --")
    ClassifierMetrics.conf_matrix(
        SVC_vit.y_test,
        y_pred_vit_classif,
        class_names=["Non-Responder", "Responder"],
        title=f"Confusion Matrix for speed classification (random state:{SVC_vit.random_state})",
        output_path=output_path,
        rdm_state=SVC_vit.random_state,
    )

    SVC_vit.shap_analysis = SHAPPlots.shap_values_calculation(SVC_vit)

    # Shap plots
    SHAPPlots.plot_shap_summary(SVC_vit, features_names, output_path)
    SHAPPlots.plot_shap_bar(SVC_vit, features_names)
