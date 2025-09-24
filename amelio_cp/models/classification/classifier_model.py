import pandas as pd
from amelio_cp.optimisation.optimisation_methods import OptimisationMethods
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import joblib


# %% Base Model for classification
class ClassifierModel:
    def __init__(self):
        # self.name = name          # can name the model to call them then (i.e.SVRModel("Model A")), or can only initiate then such as model_A = SVRModel()
        self.model = None  # will store the best SVR model, updated each time
        self.X = None  # features of training dataset, start with nothing, but will be completed each time w/ a new sample
        self.y = None  # labels of training dataset, IDEM
        self.best_params = None  # stores the best parameters, and updates it everytime the addition of a sample allows better results

        # to be defined in child classes
        self.pipeline = None  # i.e., Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
        self.primary_scoring = None
        self.secondary_scoring = None

    def add_data(self, X, y):
        """Function that will add new samples to the training set."""
        X = pd.DataFrame(X)  # pandas conversion
        y = pd.Series(y)

        if self.X is None:  # if nothing, will just take it
            self.X = X
            self.y = y
        else:  # if already with something in, will append the new sample
            self.X = pd.concat([self.X, X], ignore_index=True)
            self.y = pd.concat([self.y, y], ignore_index=True)

    def perf_estimate(self, n_iter):
        """Check for the overall perf of the model with nested CV method"""

        if self.X is None or self.y is None:  # Check if there is some data
            raise ValueError("No data available for training.")
        if self.pipeline is None or self.param_distributions is None:
            raise ValueError("Child class must define pipeline and param_distributions.")

        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=72)

        print(f"🔍 Starting hyperparameter search...")

        # Define search space
        pbounds = self.param_distributions

        # Create a pipeline for the model: scaling + SVR - everydata will pas through that order
        pipeline = self.pipeline

        # Creating the optimisation loop
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=pbounds,
            n_iter=n_iter,
            scoring=self.primary_scoring,
            cv=inner_cv,
            random_state=72,
            verbose=2,
            n_jobs=-1,
        )

        cv_prim = cross_val_score(search, self.X, self.y, cv=outer_cv, scoring=self.primary_scoring)
        print(f"📊 CV {self.primary_scoring}: {cv_prim.mean():.4f} ± {cv_prim.std():.4f}")

        results = {
            "CV {self.primary_scoring} scores": cv_prim.tolist(),
            "CV {self.primary_scoring} mean": float(cv_prim.mean()),
            "CV {self.primary_scoring} std": float(cv_prim.std()),
        }

        if self.secondary_scoring is not None:
            cv_sec = cross_val_score(search, self.X, self.y, cv=outer_cv, scoring=self.secondary_scoring)
            print(f"📊 CV {self.secondary_scoring}: {cv_sec.mean():.4f} ± {cv_sec.std():.4f}")

        results.update(
            {
                "CV {self.secondary_scoring} scores": cv_sec.tolist(),
                "CV {self.secondary_scoring} mean": float(cv_sec.mean()),
                "CV {self.secondary_scoring} std": float(cv_sec.std()),
            }
        )

        return results

    def train_and_tune(self, method: str, n_iter=100):
        """Tune hyperparameters with choosen method and fit the model."""
        if self.X is None or self.y is None:  # Check if there is some data
            raise ValueError("❌ No data available for training.")

        # Define search space
        pbounds = self.param_distributions

        # Creating the optimisation loop
        if method == "random":
            search = OptimisationMethods.random_search(self.pipeline, pbounds, n_iter, k_folds=5, primary_scoring=self.primary_scoring)
            search.fit(self.X, self.y)  # training
            # self.model = search.best_estimator_  # recover the best model
            # self.best_params = search.best_params_  # recover the best hp
            print("Random search optimisation completed.")

        elif method == "bayesian":
            search = OptimisationMethods.bayesian_search('SVC', self.pipeline, n_iter, k_folds=5, primary_scoring=self.primary_scoring)
            search.fit(self.X, self.y)  # training
            print("Byesian Search optimisation completed.")

        elif method == "bayesian_optim":
            search = OptimisationMethods.bayesian_optim(self.pipeline, self.X, self.y)
            print("Bayesian optimisation completed.")

        else:
            raise ValueError("❌ Unknown optimisation method. Choose 'random', 'bayesian' or 'bayesian_optim'.")
        

        self.model = search.best_estimator_  # recover the best model
        self.best_params = search.best_params_  # recover the best hp
        
        # Evaluate
        preds = self.model.predict(self.X)  # quick check to see if model OK (no overfitting)
        acc = accuracy_score(self.y, preds)  # IDEM
        print(f"Best Params: {self.best_params}")
        print(f"Accuracy on training data: {acc:.4f}")

        # Evaluate with K-Fold CV for stability
        # K-Fold CV setup
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=72)
        cv_acc = cross_val_score(self.model, self.X, self.y, cv=cv_splitter, scoring="accuracy")
        print(f"📊 CV accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

        return {
            "Accuracy": acc,
            "CV accuracy": cv_acc.mean(),
        }

    def fit(self, X, y):
        """
        Train the model with the (X, y) dataset
        Actually, refits the current optimized model on a new dataset,
        without hyperparameter tuning.
        """
        if self.model is None:
            raise ValueError("Model has not been optimised yet.")
        return self.model.fit(X, y)

    def predict(self, X):
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def save(self, path):
        """Save model and training data."""
        joblib.dump({"model": self.model, "X": self.X, "y": self.y, "best_params": self.best_params}, path)
        print(f"💾 Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load a saved model."""
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.X = data["X"]
        obj.y = data["y"]
        obj.best_params_ = data["best_params"]
        print(f"📂 Model loaded from {path}")
        return obj

# %%
