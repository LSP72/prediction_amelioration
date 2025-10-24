from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from .linear_model import LinearModel

# %% Random Forest Regression


class RFRModel(LinearModel):
    def __init__(self):
        super().__init__()

        self.model = RandomForestRegressor()

        self.param_distributions = {
            "rfr__n_estimators": randint(50, 200),
            "rfr__max_depth": [None, 5, 10, 20],
            "rfr__min_samples_split": randint(2, 5),
        }
