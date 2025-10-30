from sklearn.svm import SVR
from scipy.stats import uniform
from .linear_model import LinearModel


# %% SVR
class SVRModel(LinearModel):
    def __init__(self):
        super().__init__()

        self.name = "svr"
        self.model = SVR()

        self.param_distributions = {
            "svr__C": uniform(1, 500),
            "svr__epsilon": uniform(0.01, 1),
            "svr__kernel": ["linear", "poly", "rbf"],
            "svr__gamma": ["scale", "auto"],
        }
