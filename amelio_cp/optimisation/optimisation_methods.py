from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, cross_val_score
from skopt import BayesSearchCV
from bayes_opt import BayesianOptimization
from sklearn.svm import SVC
from skopt.space import Real, Integer, Categorical
from scipy.stats import uniform, randint


class OptimisationMethods:
    def __init__(self):
        pass

    @staticmethod
    def random_search(model, n_iter, k_folds, primary_scoring):

        pbounds = {
            "C": uniform(1, 1000),
            "gamma": uniform(0.001, 0.1),
            "degree": randint(2, 5),
            "kernel": ["linear", "poly", "rbf"],  # categorical options
        }

        print("⚙️ Starting RandomizedSearchCV optimisation...")

        cv_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            model,
            param_distributions=pbounds,
            n_iter=n_iter,
            scoring=primary_scoring,
            cv=cv_splitter,
            random_state=42,
            verbose=1,
            n_jobs=-1,
        )

        return search

    @staticmethod
    def bayesian_search(model, n_iter, k_folds, primary_scoring):
        print("⚙️ Starting Bayesian Search Optimization...")

        pbounds = {
            "C": Real(1, 1e3, prior="log-uniform"),
            "gamma": Real(1e-3, 1, prior="log-uniform"),
            "degree": Integer(2, 5),
            "kernel": Categorical(["linear", "poly", "rbf"]),
        }

        cv_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        search = BayesSearchCV(
            model,
            search_spaces=pbounds,
            n_iter=n_iter,
            scoring=primary_scoring,
            cv=cv_splitter,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        return search

    @staticmethod
    def bayesian_optim(model, X, y):

        pbounds = {
            "C": (1, 1000),
            "gamma": (0.001, 1),
            "degree": (2, 5),
            "kernel": (0, 2),  # 0: 'linear', 1: 'poly', 2: 'rbf'
        }
        kernel_options = ["linear", "poly", "rbf"]

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
