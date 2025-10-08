import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, LeaveOneOut, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import uniform
from .linear_model import LinearModel

#%% Linear Regression

class LRModel(LinearModel):
    def __init__(self): 
        super().__init__()

        self.model = LinearRegression()

    def train_and_tune(self, n_iter=100):
        """Tune hyperparameters"""
        if self.X is None or self.y is None:        # Check if there is some data
            raise ValueError("No data available for training.")

        cv=KFold(n_splits=5, shuffle=True, random_state=72)
        
        print(f"🔍 Starting hyperparameter search...")
        
        model_pipeline = self.pipeline
    
        model_pipeline.fit(self.X, self.y)      # training

        #TODO: understanding how lr works
        self.model = model_pipeline.best_estimator_         # recover the best model
        self.best_params = model_pipeline.best_params_     # recover the best hp
        print("✅ Optimisation completed and model trained.")

        # Evaluate
        preds = self.model.predict(self.X)          # quick check to see if model OK (no overfitting)
        r2 = r2_score(self.y, preds)                # IDEM
        mse = mean_squared_error(self.y, preds)     # IDEM
        print(f"Best Params: {self.best_params}")
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