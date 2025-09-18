from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, cross_val_score
from bayes_opt import BayesianOptimization


class OptimisationMethods:
    def __init__(self):
        pass

    @staticmethod
    def random_search(pipeline, pbounds, n_iter, k_folds):
        print("⚙️ Starting RandomizedSearchCV optimisation...")

        cv_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=72)

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=pbounds,
            n_iter=n_iter,
            scoring="neg_mean_squared_error",  # will try to maximise r2
            cv=cv_splitter,
            random_state=72,
            verbose=1,
            n_jobs=-1,
        )

        return search

    @staticmethod
    def bayesian_search(pipeline, model, X, y, pbounds, n_iter, k_folds):

        def svm_model(C, gamma, degree):
            kernel = "rbf"
            SVM = model(kernel=kernel, C=C, gamma=gamma, degree=int(degree), random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            acc = cross_val_score(SVM, X, y, cv=cv, scoring="accuracy")
            return acc.mean()

        print("⚙️ Starting Bayesian optimisation...")

        cv_splitter = KFold(n_splits=k_folds, shuffle=True, random_state=72)

        search = BayesianOptimization(
            pipeline,
            param_distributions=pbounds,
            n_iter=n_iter,
            scoring="neg_mean_squared_error",  # will try to maximise r2
            cv=cv_splitter,
            random_state=72,
            verbose=1,
            n_jobs=-1,
        )

        return search
