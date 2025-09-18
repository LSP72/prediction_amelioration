from sklearn.svm import SVC
from .classifier_model import ClassifierModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform

#%% SVC
class SVCModel(ClassifierModel):
    def __init__(self):    
        super().__init__()

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC())
            ])
        self.param_distributions = {
            "svc__C": uniform(1, 500),
            "svc__degree": [2,3,4],
            "svc__kernel": ["linear", "poly", "rbf"],
            "svc__gamma": ["scale", "auto"]     # "scale" = 1/(n_features * X.var())
                                                # "auto" = 1/n_features 
        }
        self.primary_scoring = "accuracy"
        self.secondary_scoring = "f1"