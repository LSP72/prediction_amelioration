from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, cross_val_score
from skopt import BayesSearchCV
from bayes_opt import BayesianOptimization
from sklearn.svm import SVC
from skopt.space import Real, Integer, Categorical
from scipy.stats import uniform, randint


# TODO: find a way to collect training accuracies
class OptimisationMethods:
    def __init__(self):
        pass

# %% Functions to get the right pbounds shape
    @staticmethod
    def _get_pbounds_for_svm(model_name: str, optim_method: str, params_distrib: dict):
        C_low, C_high = params_distrib["C"]
        gamma_low, gamma_high = params_distrib["gamma"]
        deg_low, deg_high = params_distrib["degree"]
        if model_name == "svr":
            if "epsilon" in params_distrib.keys():
                epsilon_low, epsilon_high = params_distrib["epsilon"]
            else:
                raise ValueError("epsilon bounds must be provided for SVR model.")

        if optim_method == "random":
            pbounds = {
                "C": uniform(C_low, C_high),
                "gamma": uniform(gamma_low, gamma_high),
                "degree": randint(deg_low, deg_high),
                "kernel": ["linear", "poly", "rbf"]
            }
            if model_name == "svr":
                if "epsilon" in params_distrib.keys():
                    pbounds["epsilon"] = uniform(epsilon_low, epsilon_high)
                else:
                    raise ValueError("epsilon bounds must be provided for SVR model.")
                
        elif optim_method == "bayesian_search":
            pbounds = {
                "C": Real(C_low, C_high), # prior = "log-uniform"
                "gamma": Real(gamma_low, gamma_high), # prior = "log-uniform"
                "degree": Integer(deg_low, deg_high),
                "kernel": Categorical(["linear", "poly", "rbf"])
            }
            if model_name == "svr":
                if "epsilon" in params_distrib.keys():
                    pbounds["epsilon"] = Real(epsilon_low, epsilon_high)
                else:
                    raise ValueError("epsilon bounds must be provided for SVR model.")
                
        elif optim_method == "bayesian_optim":
            pbounds = {
                "C": (C_low, C_high),
                "gamma": (gamma_low, gamma_high),
                "degree": (deg_low, deg_high),
                "kernel": (0, 2)  # 0: 'linear', 1: 'poly', 2: 'rbf'
            }
            if model_name == "svr":
                if "epsilon" in params_distrib.keys():
                    pbounds["epsilon"] = (epsilon_low, epsilon_high)
                else:
                    raise ValueError("epsilon bounds must be provided for SVR model.")
                
        return pbounds

    @staticmethod
    def _get_pbounds_for_rf(params_distrib: dict):
        pass

# %% Optimisation functions    
    @staticmethod
    def random_search(self, model, n_iter, k_folds):

        pbounds = self._get_pbounds_for_random(model.params_distrib)
        print("⚙️ Starting RandomizedSearchCV optimisation...")

        cv_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            model,
            param_distributions=pbounds,
            n_iter=n_iter,
            scoring="accuracy",
            cv=cv_splitter,
            random_state=42,
            verbose=1,
            n_jobs=-1,
        )

        return search

    @staticmethod
    def bayesian_search(self, model, n_iter, k_folds):
        print("⚙️ Starting Bayesian Search Optimization...")

        pbounds = self._get_pbounds_for_bayesian_search(model.params_distrib)

        cv_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        search = BayesSearchCV(
            model,
            search_spaces=pbounds,
            n_iter=n_iter,
            scoring="accuracy",
            cv=cv_splitter,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        return search

    def bayesian_optim(self, model, X, y):

        pbounds, kernel_options = self._get_pbounds_for_bayesian_optim(model.params_distrib)

        def svm_model(C, gamma, degree, kernel):
            params = {"C": C, "gamma": gamma, "degree": int(degree), "kernel": kernel_options[int(kernel)]}
            model_to_optim = model.set_params(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model_to_optim, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        print("⚙️ Starting Bayesian optimisation...")

        optimizer = BayesianOptimization(f=svm_model, pbounds=pbounds, random_state=42, verbose=3)
        optimizer.maximize(init_points=10, n_iter=100)
        best_params = optimizer.max["params"]
        best_params["degree"] = int(best_params["degree"])  # Convert to int
        best_params["C"] = float(best_params["C"])  # Convert to float
        best_params["kernel"] = kernel_options[int(best_params["kernel"])]  # Map back to string

        final_params = {
            "C": float(best_params["C"]),
            "gamma": float(best_params["gamma"]),
            "degree": int(best_params["degree"]),
            "kernel": best_params["kernel"],
        }

        best_model = model.set_params(**final_params)
        best_model.fit(X, y)

        class ResultWrapper:
            def __init__(self, model, params):
                self.best_estimator_ = model
                self.best_params_ = params

        return ResultWrapper(best_model, best_params)
