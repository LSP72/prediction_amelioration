from .optimisation_methods import OptimisationMethods
from sklearn.model_selection import KFold, cross_val_score
from bayes_opt import BayesianOptimization


class OptimisationMethodsLin(OptimisationMethods):
    def __init__(self):
        super().__init__()

    @staticmethod
    def bayesian_optim(model, X, y):

        pbounds = {
            "C": (1, 1000),
            "gamma": (0.001, 1),
            "epsilon": (0.001, 1),
            "degree": (2, 5),
            "kernel": (0, 2),  # 0: 'linear', 1: 'poly', 2: 'rbf'
        }
        kernel_options = ["linear", "poly", "rbf"]

        def function_to_min(C, gamma, epsilon, degree, kernel):
            params = {
                "C": C,
                "gamma": gamma,
                "epsilon": epsilon,
                "degree": int(degree),
                "kernel": kernel_options[int(kernel)],
            }
            try_model = model.set_params(**params)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(try_model, X, y, cv=cv, scoring="neg_mean_squared_error")
            return scores.mean()

        print("⚙️ Starting Bayesian optimisation...")

        optimizer = BayesianOptimization(f=function_to_min, pbounds=pbounds, random_state=42, verbose=3)
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
            "epsilon": float(best_params["epsilon"]),
        }

        best_model = model.set_params(**final_params)
        best_model.fit(X, y)

        class ResultWrapper:
            def __init__(self, model, params):
                self.best_estimator_ = model
                self.best_params_ = params

        return ResultWrapper(best_model, best_params)
