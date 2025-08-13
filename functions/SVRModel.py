import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import uniform
import joblib

# Class that deals with the model
class SVRModel:
    def __init__(self):    
        # self.name = name          # can name the model to call them then (i.e.SVRModel("Model A")), or can only initiate then such as model_A = SVRModel()
        self.model = None           # will store the best SVR model, updated each time 
        self.X = None               # features of training dataset, start with nothing, but will be completed each time w/ a new sample
        self.y = None               # labels of training dataset, IDEM
        self.best_params_ = None    # stores the best parameters, and updates it everytime the addition of a sample allows better results

    def add_data(self, X, y):
        """Function that will add new samples to the training set."""
        X = pd.DataFrame(X)         # pandas conversion
        y = pd.Series(y)

        if self.X is None:          # if nothing, will just take it
            self.X = X
            self.y = y
        else:                       # if already with something in, will append the new sample
            self.X = pd.concat([self.X, X], ignore_index=True)
            self.y = pd.concat([self.y, y], ignore_index=True)

    def train_and_tune(self, n_iter=100):
        """Tune hyperparameters with RandomizedSearchCV + LOO CV."""
        if self.X is None or self.y is None:        # Check if there is some data
            raise ValueError("No data available for training.")

        loo = LeaveOneOut()     # method used to cross-validate in the optimisation

        # Define search space
        pbounds = {
            "svr__C": uniform(1, 800),
            "svr__epsilon": uniform(0.01, 1),
            "svr__kernel": ["linear", "poly", "rbf"],
            "svr__gamma": ["scale", "auto"]
        }

        # Create a pipeline for the model: scaling + SVR - everydata will pas through that order
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR())
            ])
    
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=pbounds,
            n_iter=n_iter,
            scoring="neg_mean_squared_error",               # will try to maximise r2
            cv=loo,
            random_state=72,
            n_jobs=-1
        )

        search.fit(self.X, self.y)      # training

        self.model = search.best_estimator_         # recover the best model
        self.best_params_ = search.best_params_     # recover the best hp

        # Evaluate
        preds = self.model.predict(self.X)          # quick check to see if model OK (no overfitting)
        r2 = r2_score(self.y, preds)                # IDEM
        mse = mean_squared_error(self.y, preds)     # IDEM

        print("✅ Training complete.")
        print(f"Best Params: {self.best_params_}")
        print(f"R²: {r2:.4f}, MSE: {mse:.4f}")

    def predict(self, X):
        """Make predictions with the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def save(self, path):
        """Save model and training data."""
        joblib.dump({
            "model": self.model,
            "X": self.X,
            "y": self.y,
            "best_params": self.best_params_
        }, path)
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

#%% RUNNING THE CODE

# # ====== Later: load & use ======
# model = SVRModel.load("svr_model.pkl")

# # Add new sample
# model.add_data(new_X, new_y)

# # Re-train if necessary
# model.train_and_tune(n_iter=100)

# # Save updated model
# model.save("svr_model.pkl")

# # Predict on new data
# preds = model.predict(X_test)
