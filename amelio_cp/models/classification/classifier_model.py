import pandas as pd
from amelio_cp.optimisation.optimisation_methods import OptimisationMethods
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


# %% Base Model for classification
class ClassifierModel:
    def __init__(self):
        # TODO: variable and function names explicit at 1st read
        self.name = None          # can name the model to call them then (i.e.SVRModel("Model A")), or can only initiate then such as model_A = SVRModel()
        self.model = None  # will store the best model, should be updated each time
        self.scaler = StandardScaler()
        self.X = None # features of training dataset, start with nothing, but will be completed each time w/ a new sample
        self.X_scaled = None  
        self.y = None  # labels of training dataset, IDEM
        self.X_test = None
        self.X_test_scaled = None
        self.y_test = None
        self.best_params = None  # stores the best parameters, and updates it everytime the addition of a sample allows better results


    # TODO: decide whether you need them or not
        # to be defined in child classes
        self.primary_scoring = None
        self.secondary_scoring = None
        self.shap_analysis = None  # to store the shap analysis object if needed

    # TODO: collect feature keys    
    # TODO: checking if the test data are in the same order than train ones

    def add_train_data(self, X, y):
        """Function that will add new samples to the training set."""
        self.X, self.y = self._add_template(X, y, self.X, self.y)
        self.X_scaled = self.scaler.fit_transform(self.X)

    def add_test_data(self, X, y):
        """Function that will add new samples to the training set."""
        self.X_test, self.y_test = self._add_template(X, y, self.X_test, self.y_test)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def add_data(self, X, y, test_size): 
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=72)        
        print('✅ Split has been done.', flush=True)
        self.add_train_data(x_train, y_train)
        self.add_test_data(x_test, y_test)

    @staticmethod
    def _add_template(X_given, y_given, X_model, y_model):
        X_given = pd.DataFrame(X_given)  # pandas conversion
        y_given = pd.Series(y_given)

        if X_model is None:  # if nothing, will just take it
            X_model = X_given
            y_model = y_given
        else:  # if already with something in, will append the new sample
            X_model = pd.concat([X_model, X_given], ignore_index=True)
            y_model = pd.concat([y_model, y_given], ignore_index=True)
        return X_model, y_model


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
            search = OptimisationMethods.random_search(self.model, pbounds, n_iter, k_folds=5, primary_scoring=self.primary_scoring)
            search.fit(self.X_scaled, self.y)  # training
            print("Random search optimisation completed.")

        elif method == "bayesian":
            search = OptimisationMethods.bayesian_search(self.model, n_iter, k_folds=5, primary_scoring=self.primary_scoring)
            search.fit(self.X_scaled, self.y)  # training
            print("Byesian Search optimisation completed.")

        elif method == "bayesian_optim":
            search = OptimisationMethods.bayesian_optim(self.model, self.X_scaled, self.y)
            print("Bayesian optimisation completed.")

        else:
            raise ValueError("❌ Unknown optimisation method. Choose 'random', 'bayesian' or 'bayesian_optim'.")
        
        self.model = search.best_estimator_  # recover the best model
        self.best_params = search.best_params_  # recover the best hp
        
        # Evaluate
        preds = self.model.predict(self.X_scaled)  # quick check to see if model OK (no overfitting)
        acc = accuracy_score(self.y, preds)  # IDEM
        print(f"Best Params: {self.best_params}")
        print(f"Accuracy on training data: {acc:.4f}")

        # Evaluate with K-Fold CV for stability
        # K-Fold CV setup
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=72)
        cv_acc = cross_val_score(self.model, self.X_scaled, self.y, cv=cv_splitter, scoring="accuracy")
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

    # TODO: (Mathilde) rewrite to consider the new definiton of the class
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
