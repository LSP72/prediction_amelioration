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
    def random_search(pipeline, pbounds, n_iter, k_folds, primary_scoring):
        
        pbounds = {
            "svc__C": uniform(1, 999),          # samples in [1, 1000)
            "svc__gamma": uniform(0.001, 0.999), # samples in [0.001, 1)
            "svc__degree": randint(2, 6),       # integers {2, 3, 4, 5}
            "svc__kernel": ["linear", "poly", "rbf"],  # categorical options
        }
        
        print("⚙️ Starting RandomizedSearchCV optimisation...")

        cv_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=72)

        search = RandomizedSearchCV(
            pipeline,
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
    def bayesian_search(model, pipeline, n_iter, k_folds, primary_scoring):
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
            pipeline,
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
    def bayesian_optim(pipeline, X, y):
        
        pbounds = {
            "svc__C": (1, 1000),
            "svc__gamma": (0.001, 1),
            "svc__degree": (2, 5),
            "svc__kernel": (0, 2),  # 0: 'linear', 1: 'poly', 2: 'rbf'
        }
        kernel_options = ['linear', 'poly', 'rbf']

        def svm_model(svc__C, svc__gamma, svc__degree, svc__kernel):
            params = {
                "svc__C": svc__C,
                "svc__gamma": svc__gamma,
                "svc__degree": int(svc__degree),
                "svc__kernel": kernel_options[int(svc__kernel)]
            }
            model = pipeline.set_params(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            return scores.mean()

        print("⚙️ Starting Bayesian optimisation...")

        optimizer = BayesianOptimization(f = svm_model, pbounds = pbounds, random_state=42, verbose=3)
        optimizer.maximize(init_points=10, n_iter=100) 
        best_params = optimizer.max['params']
        best_params['svc__degree'] = int(best_params['svc__degree'])  # Convert to int
        best_params['svc__C'] = float(best_params['svc__C'])  # Convert to float
        best_params['svc__kernel'] = kernel_options[int(best_params['svc__kernel'])]  # Map back to string

        best_model = pipeline.set_params(**best_params)
        best_model.fit(X, y)

        class ResultWrapper:
            def __init__(self, model, params):
                self.best_estimator_ = model
                self.best_params_ = params

        return ResultWrapper(best_model, best_params)
