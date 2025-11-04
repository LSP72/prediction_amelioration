import pandas as pd
import pickle as pkl
from amelio_cp import RFCModel
from amelio_cp import SVCModel
from amelio_cp import Process
from amelio_cp import SHAPPlots
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
import time


def build_model(model_name, seed=42):
    if model_name == "svc":
        model = SVCModel()

    elif model_name == "svr":
        pass

    model.random_state = seed
    return model

# will remove the data 20 [0] and 44 [1]
def prepare_data(data_path, features_path, condition_to_predict, model_name, samples_to_keep):
    all_data = Process.load_csv(data_path)
    if condition_to_predict == "VIT":
        all_data = all_data.drop(["6MWT_POST"], axis=1)
        all_data = all_data.dropna()
        if model_name == "svc":
            y = Process.calculate_MCID(all_data["VIT_PRE"], all_data["VIT_POST"], "VIT")
        else:
            y = all_data["VIT_POST"]

    elif condition_to_predict == "6MWT":
        all_data = all_data.drop(["VIT_POST"], axis=1)
        all_data = all_data.dropna()
        if model_name == "svc":
            y = Process.calculate_MCID(all_data["6MWT_PRE"], all_data["6MWT_POST"], "6MWT", all_data["GMFCS"])
        else:
            y = all_data["6MWT_POST"]

    features = pd.read_excel(features_path)
    selected_features = features["19"].dropna().to_list()
    features_names = features["19_names"].dropna().to_list()

    X = all_data[selected_features]

    if all(i in X.index for i in samples_to_keep):
        X_ex, y_ex = X.loc[samples_to_keep], y.loc[samples_to_keep]
        X.drop(samples_to_keep, inplace=True)
        y.drop(samples_to_keep, inplace=True)
    else:
        missing = [i for i in samples_to_keep if i not in X.index]
        raise ValueError(f"Some chosen labels are missing:{missing}")

    return X, y, X_ex, y_ex, features_names


def load_data(X, y, model, X_ex, y_ex):
    model.add_train_data(X, y) # not going to evaluate the model, so using all the data for training
    model.add_test_data(X_ex, y_ex)

def append_data(results_dict, model, id, time, training_accuracy, precision_score, conf_matrix):
    results_dict["id_" + str(id)] = {
        "model_name": model.name,
        "optim_method": model.optim_method,
        "seed": model.random_state,
        "C": model.best_params["C"],
        "gamma": model.best_params["gamma"],
        "degree": model.best_params["degree"],
        "kernel": model.best_params["kernel"],
        "training_accuracy": training_accuracy,
        "precision_score": precision_score,
        "confusion_matrix": conf_matrix,
        "optim_time": time,
    }

    return results_dict


def save_data(results_dict, model_name, output_path):
    pickle_file_name = output_path + model_name + ".pkl"
    with open(pickle_file_name, "wb") as file:
        pkl.dump(results_dict, file)


def main(condition_to_predict):
    results_dict = {}
    data_path = "datasets/sample_2/all_data_28pp.csv"
    features_path = "amelio_cp/processing/Features.xlsx"
    output_path = "examples/results/svc_vs_svr_rdm_state/"
    output_path_shap = output_path + "shap_fig/"
    models = ['svc', 'svr']

    for model_name in enumerate(models):
        starting_time = time.time()

        model = build_model(model_name)
        X, y, X_ex, y_ex, features_names = prepare_data(data_path, features_path, condition_to_predict, model_name)
        load_data(X, y, X_ex, y_ex, model)

        model.model.train_and_tune(model.X_train_scaled, model.y_train)

        y_pred = model.model.predict(model.X_test_scaled)

        optim_time = time.time() - starting_time

        results_dict[model_name] = {
            "C": model.best_params["C"],
            "gamma": model.best_params["gamma"],
            "degree": model.best_params["degree"],
            "kernel": model.best_params["kernel"],
            "prediction": y_pred,
            "optim_time": optim_time
            }

    save_data(results_dict, model_name, output_path)


if __name__ == "__main__":
    main("VIT")
    main("6MWT")
