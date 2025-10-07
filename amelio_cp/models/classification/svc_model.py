from sklearn.svm import SVC
from .classifier_model import ClassifierModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform


# %% SVC
class SVCModel(ClassifierModel):
    def __init__(self):
        super().__init__()

        self.name = 'svc'
        self.model = SVC(probability=True)
        self.search_spaces = {
            "C": [1, 1000], 
            "gamma": [0.001, 0.1],
            "degree": [2, 5],
            "kernel": ["linear", "poly", "rbf"]
            }
        
        self.primary_scoring = "accuracy"
        self.secondary_scoring = "f1"
