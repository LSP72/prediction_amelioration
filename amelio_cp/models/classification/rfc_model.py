from sklearn.ensemble import RandomForestClassifier
from .classifier_model import ClassifierModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# %% SVC
class RFCModel(ClassifierModel):
    def __init__(self):
        super().__init__()

        self.pipeline = Pipeline([("scaler", StandardScaler()), ("svc", RandomForestClassifier())])
        self.param_distributions = {}
        self.primary_scoring = "accuracy"
        self.secondary_scoring = "f1"
