import pandas as pd
import pickle as pkl
from amelio_cp import SVRModel
from amelio_cp import SVCModel
from amelio_cp import Process
from amelio_cp import SHAPPlots
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
import time


def build_model(random_state_split, random_state_cv, random_state_optim):
    model = SVCModel()
    model.random_state_split = random_state_split
    model.random_state_cv = random_state_cv
    model.random_state_optim = random_state_optim

    return model


def prepare_data(data_path, features_path, condition_to_predict):
    all_data = Process.load_csv(data_path)
    if condition_to_predict == "VIT":
        all_data = all_data.drop(["6MWT_POST"], axis=1)
        all_data = all_data.dropna()
        y = Process.calculate_MCID(all_data["VIT_PRE"], all_data["VIT_POST"], "VIT")

    elif condition_to_predict == "6MWT":
        all_data = all_data.drop(["VIT_POST"], axis=1)
        all_data = all_data.dropna()
        y = Process.calculate_MCID(all_data["6MWT_PRE"], all_data["6MWT_POST"], "6MWT", all_data["GMFCS"])

    else:
        raise ValueError("Condition to predict not recognized. Choose either 'VIT' or '6MWT'.")

    features = pd.read_excel(features_path)
    selected_features = features["19"].dropna().to_list()

    X = all_data[selected_features]

    return X, y


def load_data(X, y, model, test_size=0.2):
    model.add_data(X, y, test_size)


def append_data(results_dict, model, id, optim_time, precision_score, precision_score_from_model, conf_matrix, y_true, y_pred):
    results_dict["id_" + str(id)] = {
        "model_name": model.name,
        "optim_method": model.optim_method,
        "random_state": model.random_state,
        "random_state_split": model.random_state_split,
        "random_state_cv": model.random_state_cv,
        "random_state_optim": model.random_state_optim,
        "C": model.best_params["C"],
        "gamma": model.best_params["gamma"],
        "degree": model.best_params["degree"],
        "kernel": model.best_params["kernel"],
        "precision_score": precision_score,
        "precision_score_from_model": precision_score_from_model,
        "confusion_matrix": conf_matrix,
        "optim_time": optim_time,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    return results_dict


def save_data(results_dict, condition_to_predict, randomized_seed_type, output_path):
    pickle_file_name = output_path + randomized_seed_type + '_' + condition_to_predict + ".pkl"
    with open(pickle_file_name, "wb") as file:
        pkl.dump(results_dict, file)


def main(seeds_dict, condition_to_predict, randomized_seed_type):
    results_dict = {}
    data_path = "datasets/sample_2/all_data_28pp.csv"
    features_path = "amelio_cp/processing/Features.xlsx"
    output_path = "examples/results/svc_vs_svr_rdm_state/random_seeds/"

    for i in range(len(seeds_dict["split"])):

        random_state_split = seeds_dict["split"][i]
        random_state_cv = seeds_dict["cv"][i]
        random_state_optim = seeds_dict["optim"][i]
        model = build_model(random_state_split, random_state_cv, random_state_optim)

        print("************************************")
        print(f"Running for {condition_to_predict} with: "
              f"split seed: {model.random_state_split}, ",
              f"cv seed: {model.random_state_cv}, ", 
              f"optim seed: {model.random_state_optim}.")

        starting_time = time.time()
        
        X, y = prepare_data(data_path, features_path, condition_to_predict)
        load_data(X, y, model)

        model.train_and_tune("bayesian_optim")
        y_pred = model.model.predict(model.X_test_scaled)

        optim_time = time.time() - starting_time

        conf_matrix = confusion_matrix(model.y_test, y_pred)
        precision_score = accuracy_score(model.y_test, y_pred)
        precision_score_from_model = model.model.score(model.X_test_scaled, model.y_test)

        results_dict = append_data(
            results_dict, model, i, optim_time, precision_score, precision_score_from_model, conf_matrix, model.y_test, y_pred
        )

    save_data(results_dict, condition_to_predict, randomized_seed_type, output_path)


if __name__ == "__main__":
    nb_iter = 10
    split = [[i for i in range(1, nb_iter+1)], [42]*nb_iter, [42]*nb_iter]
    cv = [[42]*nb_iter, [i for i in range(1, nb_iter+1)], [42]*nb_iter]
    optim = [[42]*nb_iter, [42]*nb_iter, [i for i in range(1, nb_iter+1)]]
    randomized_seed_type_list = ["split", "cv", "optim"]

    for i in range(len(split)):
        seeds_dict = {"split": split[i],
                      "cv": cv[i], 
                      "optim": optim[i]
                      }
        main(seeds_dict, "6MWT", randomized_seed_type_list[i])
        main(seeds_dict, "VIT", randomized_seed_type_list[i])
