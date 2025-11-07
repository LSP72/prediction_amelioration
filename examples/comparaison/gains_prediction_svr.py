import pandas as pd
import pickle as pkl
from amelio_cp import SVRModel
from amelio_cp import Process
from amelio_cp import SHAPPlots
from sklearn.metrics import r2_score
import time

def build_model(seed):
    model = SVRModel()
    model.random_state = seed
    return model

def prepare_data(data_path, features_path, condition_to_predict):
    all_data = Process.load_csv(data_path)
    if condition_to_predict == "VIT":
        all_data = all_data.drop(["6MWT_POST"], axis=1)
        all_data = all_data.dropna()
        y = all_data['VIT_POST']-all_data['VIT_PRE']

    elif condition_to_predict == "6MWT":
        all_data = all_data.drop(["VIT_POST"], axis=1)
        all_data = all_data.dropna()
        y = all_data["6MWT_POST"]-all_data["6MWT_PRE"]

    else:
        raise ValueError("Condition to predict not recognized. Choose either 'VIT' or '6MWT'.")
    
    features = pd.read_excel(features_path)
    selected_features = features["19"].dropna().to_list()
    features_names = features["19_names"].dropna().to_list()

    X = all_data[selected_features]

    return X, y, features_names

def load_data(X, y, model, test_size=0.2):
    model.add_data(X, y, test_size)

def append_data(results_dict, model, id, time, r2, y_true, y_pred):
    results_dict["id_" + str(id)] = {
        "optim_method": model.optim_method,
        "random_state": model.random_state,
        "random_state_split": model.random_state_split,
        "random_state_cv": model.random_state_cv,
        "random_state_optim": model.random_state_optim,
        "C": model.best_params["C"],
        "gamma": model.best_params["gamma"],
        "degree": model.best_params["degree"],
        "kernel": model.best_params["kernel"],
        "epsilon": model.best_params["epsilon"],
        "r2": r2,
        "optim_time": time,
        "y_true": y_true,
        "y_pred": y_pred,
        "shap_values": model.shap_analysis["shap_values"]

    }

    return results_dict

def save_data(results_dict, model_name, condition_to_predict, output_path):
    pickle_file_name = output_path + model_name + "_" + condition_to_predict + "_gains.pkl"
    with open(pickle_file_name, "wb") as file:
        pkl.dump(results_dict, file)

def main(seed_list, condition_to_predict):
    results_dict = {}
    data_path = "datasets/sample_2/all_data_28pp.csv"
    features_path = "amelio_cp/processing/Features.xlsx"
    output_path = "examples/results/svc_vs_svr_rdm_state/"
    output_path_shap = output_path + "shap_fig/"

    for i, seed in enumerate(seed_list):
        print("************************************")
        print(f"Running with the seed: {seed}, optimisation number {i}")
        starting_time = time.time()

        model = build_model(seed)
        X, y, features = prepare_data(data_path, features_path, condition_to_predict)
        load_data(X, y, model)

        model.train_and_tune("bayesian_optim", n_iter=100)
        optim_time = time.time() - starting_time
        y_pred = model.model.predict(model.X_test_scaled)

        r2 = r2_score(model.y_test, y_pred)

        model.shap_analysis = SHAPPlots.shap_values_calculation(model_class=model)
        SHAPPlots.plot_shap_summary(model, features, output_path_shap, show=False)

        results_dict = append_data(results_dict, model, seed, optim_time, r2, model.y_test, y_pred)

    save_data(results_dict, model.name, condition_to_predict, output_path)

if __name__ == "__main__":
    conditions_list = ["VIT", "6MWT"]
    # seeds_list = [20]
    seeds_list = [20, 72, 45, 36, 8, 30, 98, 63, 6, 13]
    # seeds_list = [i for i in range(1, 101)]

    for condition_to_predict in conditions_list:
        main(seeds_list, condition_to_predict)