import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import uniform
from .linear_model import LinearModel

# %% Linear Regression


class LRModel(LinearModel):
    def __init__(self):
        super().__init__()

        self.model = LinearRegression()

    def train(self, n_iter=100):
        """Tune hyperparameters"""
        if self.X_train is None or self.y_train is None:  # Check if there is some data
            raise ValueError("No data available for training.")

        print(f"Starting training...")
        self.model.fit(self.X_train, self.y_train)  # training

        print("Model trained.")

        # Evaluate
        preds = self.model.predict(self.X_train)  # quick check to see if model OK (no overfitting)
        r2 = r2_score(self.y_train, preds)  # IDEM
        mse = mean_squared_error(self.y_train, preds)  # IDEM
        print(f"Coeeficient: {self.model.coef_}")
        print(f"R²: {r2:.4f}, MSE: {mse:.4f}")

        # Evaluate with K-Fold CV for stability
        # K-Fold CV setup
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_r2 = cross_val_score(self.model, self.X_train, self.y_train, cv=cv_splitter, scoring=self.secondary_scoring)
        cv_rmse = np.sqrt(
            -cross_val_score(self.model, self.X_train, self.y_train, cv=cv_splitter, scoring=self.primary_scoring)
        )
        print(f"📊 CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        print(f"📊 CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

        return {"R²": r2, "MSE": mse, "CV R²": cv_r2.mean(), "CV RMSE": cv_rmse.mean()}
