import pandas as pd
import pickle as pkl
from amelio_cp import SVRModel
from amelio_cp import SVCModel
from amelio_cp import Process
from amelio_cp import SHAPPlots
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
import time


def build_model(model_name, seed):
    if model_name == 'svr':
        model = SVRModel()

    elif model_name == 'svc':
        model = SVCModel()

    model.random_state = seed
    return model

def prepare_data(data_path, features_path, condition_to_predict, model_name):
    all_data = Process.load_csv(data_path)
    if condition_to_predict == 'VIT':
        all_data = all_data.drop(['6MWT_POST'], axis=1)
        all_data = all_data.dropna()
        y = all_data["VIT_POST"]
    
    elif condition_to_predict == '6MWT':
        all_data = all_data.drop(['VIT_POST'], axis=1)
        all_data = all_data.dropna()
        y = all_data["6MWT_POST"]
    
    if model_name == 'svc':
        y = Process.calculate_MCID(all_data, condition_to_predict)

    features = pd.read_excel(features_path)
    selected_features = features["19"].dropna().to_list()
    features_names =  features["19_names"].dropna().to_list()
  
    X = all_data[selected_features]

    return X, y, features_names

def load_data(X, y, model, test_size=0.2):
    model.add_data(X, y, test_size)

def append_data(results_dict, model, id, time, precision_score, conf_matrix, r2=None):
    results_dict["id_"+str(id)] = {
        "model_name": model.name,
        "optim_method": model.optim_method,
        "seed": model.random_state, 
        "C": model.best_params["C"],
        "gamma": model.best_params["gamma"],
        "degree": model.best_params["degree"],
        "kernel": model.best_params["kernel"],
        "precision_score": precision_score,
        "confusion_matrix": conf_matrix,
        "optim_time": time
    }

    if model.name == 'svr':
        results_dict["id_"+str(id)]["epsilon"] = model.best_params["epsilon"]
        results_dict["id_"+str(id)]["r2"] = r2
            
    return results_dict

def save_data(results_dict, model_name, output_path):
    pickle_file_name = output_path + model_name + '.pkl'
    with open(pickle_file_name, "wb") as file:
        pkl.dump(results_dict, file)


def main(model_name, seeds_list):
    results_dict = {}
    data_path = "datasets/sample_2/all_data_28pp.csv"
    features_path = "amelio_cp/processing/Features.xlsx"
    output_path = "examples/results/svc_vs_svr_rdm_state/"
    output_path_shap = output_path + "shap_fig/"

    for i, seed in enumerate(seeds_list):
        starting_time = time.time()

        model = build_model(model_name, seed)
        X, y, features_names = prepare_data(data_path, features_path, 'VIT', model_name)
        load_data(X, y, model)

        model.train_and_tune("bayesian_optim")
        y_pred = model.model.predict(model.X_test_scaled)

        optim_time = time.time() - starting_time

        if model_name == 'svr':
            delta_vit_true = [1 if model.y_test.iloc[i] - model.X_test['VIT_PRE'].iloc[i] > 0.1 else 0 for i in range(len(model.y_test))]
            delta_vit_pred = [1 if y_pred[i] - model.X_test['VIT_PRE'].iloc[i] > 0.1 else 0 for i in range(len(y_pred))]
            conf_matrix = confusion_matrix(delta_vit_true, delta_vit_pred)
            r2 = r2_score(model.y_test, y_pred)
            precision_score = accuracy_score(delta_vit_true, delta_vit_pred)

        elif model_name == 'svc':
            conf_matrix = confusion_matrix(model.y_test, y_pred)
            r2 = None
            precision_score = accuracy_score(model.y_test, y_pred)

            model.shap_analysis = SHAPPlots.shap_values_calculation(model)
            SHAPPlots.plot_shap_summary(model, features_names, output_path_shap, show=False)

        results_dict = append_data(results_dict, model, i, optim_time, precision_score, conf_matrix, r2)

    save_data(results_dict, model_name, output_path)
    
if __name__ == "__main__":
    model_name_list =  ['svr', 'svc']
    seeds_list = [20, 72, 45, 36, 8, 30, 98, 63, 6, 13]
    for model_name in model_name_list:
        main(model_name, seeds_list)
