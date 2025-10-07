from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, cross_val_score
from skopt import BayesSearchCV
from bayes_opt import BayesianOptimization
from sklearn.svm import SVC
from skopt.space import Real, Integer, Categorical
from scipy.stats import uniform, randint
from sklearn import metrics, svm


class OptimisationMethods:
    def __init__(self):
        pass

    @staticmethod
    def random_search(model, search_spaces, n_iter, k_folds, primary_scoring):
        
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
    def bayesian_search(model, n_iter, k_folds):
        print("⚙️ Starting Bayesian Search Optimization...")
    
        pbounds = {
            'C': Real(model.search_spaces['C'][0], model.search_spaces['C'][1], prior='log-uniform'),
            'gamma': Real(model.search_spaces['gamma'][0], model.search_spaces['gamma'][1], prior='log-uniform'),
            'degree': Integer(model.search_spaces['degree'][0], model.search_spaces['degree'][0]),
            'kernel': Categorical(model.search_spaces['kernel']),
        }

        cv_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        search = BayesSearchCV(
            model,
            search_spaces=pbounds,
            n_iter=n_iter,
            scoring=model.primary_scoring,
            cv=cv_splitter,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        return search

    @staticmethod
    def bayesian_optim(model, search_spaces, X, y, random_state):
        
        pbounds = {
            "svc__C": (search_spaces['C'][0], search_spaces['C'][1]),
            "svc__gamma": (search_spaces['gamma'][0], search_spaces['gamma'][1]),
            "svc__degree": (search_spaces['degree'][0], search_spaces['degree'][1]),
            "svc__kernel": (0, len(search_spaces['kernel'])-1),  # 0: 'linear', 1: 'poly', 2: 'rbf'
        }
        kernel_options = search_spaces['kernel']

        def svm_model(svc__C, svc__gamma, svc__degree, svc__kernel):
            params = {
                "C": svc__C,
                "gamma": svc__gamma,
                "degree": int(svc__degree),
                "kernel": int(svc__kernel)
            }
            SVM = svm.SVC(kernel=kernel_options[int(params['kernel'])], C=params['C'], gamma=params['gamma'], degree=int(params['degree']), random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(SVM, X, y, cv=cv, scoring='accuracy')
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
            "random_state": random_state,
            "kernel": best_params['svc__kernel']
        }
            
        best_model = model.set_params(**final_params)
        best_model.fit(X, y)

        class ResultWrapper:
            def __init__(self, model, params):
                self.best_estimator_ = model
                self.best_params_ = params

        return ResultWrapper(best_model, best_params)
    
    # function to convert the 
