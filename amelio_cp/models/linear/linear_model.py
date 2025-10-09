import pandas as pd
import numpy as np
from amelio_cp.optimisation.optimisation_methods_for_linear import OptimisationMethodsLin
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
import joblib


# %% SVR
class LinearModel:
    def __init__(self):
        self.name = None  # can name the model to call them then (i.e.SVRModel("Model A")), or can only initiate then such as model_A = SVRModel()
        self.model = None  # will store the best model, should be updated each time
        self.scaler = StandardScaler()  # scaler used in data scaling
        self.X_train = (
            None  # features of training dataset, start with nothing, but will be completed each time w/ a new sample
        )
        self.X_train_scaled = None  # scaled features of training dataset
        self.y_train = None  # labels of training dataset, IDEM
        self.X_test = None  # features of testing dataset
        self.X_test_scaled = None  # scaled features of testing dataset
        self.y_test = None  # labels of testing dataset
        self.best_params = (
            None  # stores the best parameters, and updates it everytime the addition of a sample allows better results
        )
        self.shap_analysis = None  # stores the shap analysis objects, if needed
        self.random_state = 42  # setting a default rdm state

        # to be defined in child classes
        self.primary_scoring = None
        self.secondary_scoring = None

    # Specific function to add the training data
    def add_train_data(self, X, y):
        """Function that will add new samples to the training set."""
        self.X_train, self.y_train = self._add_template(X, y, self.X_train, self.y_train)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)

    # Specific function to add the testing data
    def add_test_data(self, X, y):
        """Function that will add new samples to the training set."""
        self.X_test, self.y_test = self._add_template(X, y, self.X_test, self.y_test)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    # Function that splits and adds datasets
    def add_data(self, X, y, test_size):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        print("✅ Split has been done.", flush=True)
        self.add_train_data(x_train, y_train)
        self.add_test_data(x_test, y_test)

    # Function that handles and correctly stores the data
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

        if self.X_train is None or self.y_train is None:  # Check if there is some data
            raise ValueError("No data available for training.")

        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=72)

        print(f"🔍 Starting hyperparameter search...")

        # Define search space
        pbounds = {
            "svr__C": uniform(1, 500),
            "svr__epsilon": uniform(0.01, 1),
            "svr__kernel": ["linear", "poly", "rbf"],
            "svr__gamma": ["scale", "auto"],  # "scale" = 1/(n_features * X.var())
            # "auto" = 1/n_features
        }

        # Create a pipeline for the model: scaling + SVR - every data will pas through that order
        pipeline_svr = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])

        # Creating the optimisation loop
        search = RandomizedSearchCV(
            pipeline_svr,
            param_distributions=pbounds,
            n_iter=n_iter,
            scoring="neg_mean_squared_error",  # will try to maximise r2
            cv=inner_cv,
            random_state=self.random_state,
            verbose=2,
            n_jobs=1,
        )

        cv_r2 = cross_val_score(search, self.X_train, self.y_train, cv=outer_cv, scoring="r2")
        cv_rmse = np.sqrt(
            -cross_val_score(search, self.X_train, self.y_train, cv=outer_cv, scoring="neg_mean_squared_error")
        )
        print(f"📊 CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        print(f"📊 CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

        return {
            "CV R² mean": cv_r2.mean(),
            "CV R² std": cv_r2.std(),
            "CV RMSE mean": cv_rmse.mean(),
            "CV RMSE std": cv_rmse.std(),
        }

    def train_and_tune(self, method: str, n_iter=100):
        """Tune hyperparameters"""
        if self.X_train is None or self.y_train is None:  # Check if there is some data
            raise ValueError("No data available for training.")

        if method == "random":
            search = OptimisationMethodsLin.random_search(
                self.model, n_iter, k_folds=5, primary_scoring=self.primary_scoring
            )
            search.fit(self.X_train_scaled, self.y_train)  # training
            print("Random search optimisation completed.")

        elif method == "bayesian":
            search = OptimisationMethodsLin.bayesian_search(
                self.model, n_iter, k_folds=5, primary_scoring=self.primary_scoring
            )
            search.fit(self.X_train_scaled, self.y_train)  # training
            print("Byesian Search optimisation completed.")

        elif method == "bayesian_optim":
            search = OptimisationMethodsLin.bayesian_optim(self.model, self.X_train_scaled, self.y_train)
            print("Bayesian optimisation completed.")

        else:
            raise ValueError("❌ Unknown optimisation method. Choose 'random', 'bayesian' or 'bayesian_optim'.")

        self.model = search.best_estimator_  # recover the best model
        self.best_params = search.best_params_  # recover the best hp

        # Evaluate
        preds = self.model.predict(self.X_train_scaled)  # quick check to see if model OK (no overfitting)
        r2 = r2_score(self.y_train, preds)  # IDEM
        mse = mean_squared_error(self.y_train, preds)  # IDEM
        print(f"Best Params: {self.best_params}")
        print(f"R²: {r2:.4f}, MSE: {mse:.4f}")

        # Evaluate with K-Fold CV for stability
        # K-Fold CV setup
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_r2 = cross_val_score(self.model, self.X_train_scaled, self.y_train, cv=cv_splitter, scoring="r2")
        cv_rmse = np.sqrt(
            -cross_val_score(
                self.model, self.X_train_scaled, self.y_train, cv=cv_splitter, scoring="neg_mean_squared_error"
            )
        )
        print(f"📊 CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        print(f"📊 CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

        return {"R²": r2, "MSE": mse, "CV R²": cv_r2.mean(), "CV RMSE": cv_rmse.mean()}

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
        joblib.dump(
            {
                "name": self.name,
                "model": self.model,
                "X_train": self.X_train,
                "X_train_scaled": self.X_train_scaled,
                "y_train": self.y_train,
                "X_test": self.X_test,
                "X_test_scaled": self.X_test_scaled,
                "y_test": self.y_test,
                "best_params": self.best_params,
                "shap_analysis": self.shap_analysis,
                "primary_scoring": self.primary_scoring,
                "secondary_scoring": self.secondary_scoring,
            },
            path,
        )
        print(f"💾 Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load a saved model."""
        data = joblib.load(path)
        obj = cls()
        obj.name = data["name"]
        obj.model = data["model"]
        obj.X_train = data["X_train"]
        obj.X_train_scaled = data["X_train_scaled"]
        obj.y_train = data["y_train"]
        obj.X_test = data["X_test"]
        obj.X_test_scaled = data["X_test_scaled"]
        obj.y_test = data["y_test"]
        obj.best_params = data["best_params"]
        obj.shap_analysis = data["shap_analysis"]
        obj.primary_scoring = data["primary_scoring"]
        obj.secondary_scoring = data["secondary_scoring"]
        print(f"📂 Model loaded from {path}")
        return obj
