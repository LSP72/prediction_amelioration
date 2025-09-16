import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, LeaveOneOut, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import uniform
import joblib

#%% SVC
class SVCModel:
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
            
    def perf_estimate(self, n_iter):
        """Check for the overall perf of the model with nested CV method"""
        
        if self.X is None or self.y is None:        # Check if there is some data
            raise ValueError("No data available for training.")
            
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=72)
        
        print(f"🔍 Starting hyperparameter search...")

        # Define search space
        pbounds = {
            "svc__C": uniform(1, 500),
            "svc__degree": [2,3,4],
            "svc__kernel": ["linear", "poly", "rbf"],
            "svc__gamma": ["scale", "auto"]     # "scale" = 1/(n_features * X.var())
                                                # "auto" = 1/n_features 
        }

        # Create a pipeline for the model: scaling + SVR - everydata will pas through that order
        pipeline_svc = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC())
            ])
    
        # Creating the optimisation loop 
        search = RandomizedSearchCV(
            pipeline_svc,
            param_distributions=pbounds,
            n_iter=n_iter,
            scoring="accuracy",               # will try to maximise r2
            cv=inner_cv,
            random_state=72,
            verbose=2,
            n_jobs=-1
        )
        
        cv_acc = cross_val_score(search, self.X, self.y, cv=outer_cv, scoring="accuracy")
        cv_f1 = cross_val_score(search, self.X, self.y, cv=outer_cv, scoring="f1")
        print(f"📊 CV accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
        print(f"📊 CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
        
        return {
            "CV accuracies": cv_acc.tolist(),
            "CV accuracy mean": cv_acc.mean(),
            "CV accuracy std": cv_acc.std(),
            "CV F1s": cv_f1.tolist(),
            "CV F1 mean": cv_f1.mean(),
            "CV F1 std": cv_f1.std()
        }


    def train_and_tune(self, n_iter=100):
        """Tune hyperparameters with RandomizedSearchCV + LOO CV."""
        if self.X is None or self.y is None:        # Check if there is some data
            raise ValueError("No data available for training.")

        cv=KFold(n_splits=5, shuffle=True, random_state=72)
        
        print(f"🔍 Starting hyperparameter search...")

        # Define search space
        pbounds = {
            "svr__C": uniform(1, 500),
            "svr__epsilon": uniform(0.01, 1),
            "svr__kernel": ["linear", "poly", "rbf"],
            "svr__gamma": ["scale", "auto"]     # "scale" = 1/(n_features * X.var())
                                                # "auto" = 1/n_features 
        }

        # Create a pipeline for the model: scaling + SVR - everydata will pas through that order
        pipeline_svr = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR())
            ])
    
        # Creating the optimisation loop 
        search = RandomizedSearchCV(
            pipeline_svr,
            param_distributions=pbounds,
            n_iter=n_iter,
            scoring="neg_mean_squared_error",               # will try to maximise r2
            cv=cv,
            random_state=72,
            verbose=2,
            n_jobs=1
        )

        search.fit(self.X, self.y)      # training

        self.model = search.best_estimator_         # recover the best model
        self.best_params_ = search.best_params_     # recover the best hp
        print("✅ Optimisation completed and model trained.")

        # Evaluate
        preds = self.model.predict(self.X)          # quick check to see if model OK (no overfitting)
        r2 = r2_score(self.y, preds)                # IDEM
        mse = mean_squared_error(self.y, preds)     # IDEM
        print(f"Best Params: {self.best_params_}")
        print(f"R²: {r2:.4f}, MSE: {mse:.4f}")
        
        # Evaluate with K-Fold CV for stability
        # K-Fold CV setup
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=72)
        cv_r2 = cross_val_score(self.model, self.X, self.y, cv=cv_splitter, scoring="r2")
        cv_rmse = np.sqrt(-cross_val_score(self.model, self.X, self.y, cv=cv_splitter, scoring="neg_mean_squared_error"))
        print(f"📊 CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        print(f"📊 CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

        
        return {'R²': r2, 
            'MSE': mse,
            'CV R²': cv_r2.mean(),
            'CV RMSE': cv_rmse.mean()
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
    
  