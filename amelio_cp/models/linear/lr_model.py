import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, LeaveOneOut, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import uniform
from .linear_model import LinearModel
import joblib

# %% Linear Regression


class LRModel(LinearModel):
    def __init__(self):
        super().__init__()

        self.pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
