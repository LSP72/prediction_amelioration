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
    def random_search(model, pbounds, n_iter, k_folds, primary_scoring):
        
        pbounds = {
            "svc__C": uniform(1, 1000), 
            "svc__gamma": uniform(0.001, 0.1),
            "svc__degree": randint(2, 5), 
            "svc__kernel": ["linear", "poly", "rbf"],  # categorical options
        }
        
        print("⚙️ Starting RandomizedSearchCV optimisation...")

        cv_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=72)

        search = RandomizedSearchCV(
            model,
            param_distributions=pbounds,
            n_iter=n_iter,
            scoring=primary_scoring,
            cv=cv_splitter,
            random_state=72,
            verbose=1,
            n_jobs=-1,
        )

        return search

    @staticmethod
    def bayesian_search(model, n_iter, k_folds, primary_scoring):
        print("⚙️ Starting Bayesian Search Optimization...")
        if model == 'SVC':
            pbounds = {
                'svc__C': Real(1, 1e+3, prior='log-uniform'),
                'svc__gamma': Real(1e-3, 1, prior='log-uniform'),
                'svc__degree': Integer(2,5),
                'svc__kernel': Categorical(['linear', 'poly', 'rbf']),
            }

        cv_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=72)

        search = BayesSearchCV(
            model,
            search_spaces=pbounds,
            n_iter=n_iter,
            scoring=primary_scoring,
            cv=cv_splitter,
            random_state=72,
            n_jobs=-1,
            verbose=1
        )
        
        return search

    @staticmethod
    def bayesian_optim(model, X, y):
        
        pbounds = {
            "svc__C": (1, 1000),
            "svc__gamma": (0.001, 1),
            "svc__degree": (2, 5),
            "svc__kernel": (0, 2),  # 0: 'linear', 1: 'poly', 2: 'rbf'
        }
        kernel_options = ['linear', 'poly', 'rbf']

        def svm_model(svc__C, svc__gamma, svc__degree, svc__kernel):
            params = {
                "C": svc__C,
                "gamma": svc__gamma,
                "degree": int(svc__degree),
                "kernel": kernel_options[int(svc__kernel)]
            }
            try_model = model.set_params(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(try_model, X, y, cv=cv, scoring='accuracy')
            return scores.mean()

        print("⚙️ Starting Bayesian optimisation...")

        optimizer = BayesianOptimization(f = svm_model, pbounds = pbounds, random_state=42, verbose=3)
        optimizer.maximize(init_points=10, n_iter=100) 
        best_params = optimizer.max['params']
        best_params['svc__degree'] = int(best_params['svc__degree'])  # Convert to int
        best_params['svc__C'] = float(best_params['svc__C'])  # Convert to float
        best_params['svc__kernel'] = kernel_options[int(best_params['svc__kernel'])]  # Map back to string

        final_params = {
            "C": float(best_params['svc__C']),
            "gamma": float(best_params['svc__gamma']),
            "degree": int(best_params['svc__degree']),
            "kernel": best_params['svc__kernel']
        }
            
        best_model = model.set_params(**final_params)
        best_model.fit(X, y)

        class ResultWrapper:
            def __init__(self, model, params):
                self.best_estimator_ = model
                self.best_params_ = params

        return ResultWrapper(best_model, best_params)
