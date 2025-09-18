from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import randint
from .linear_model import LinearModel

#%% Random Forest Regression

class RFRModel(LinearModel):
    def __init__(self):    
        super().__init__()

        self.pipeline = Pipeline([
            ("rfr", RandomForestRegressor(random_state=72))
            ])

        self.param_distributions ={
            "rfr__n_estimators": randint(50, 200),
            "rfr__max_depth": [None, 5, 10, 20],
            "rfr__min_samples_split": randint(2,5)
            }
        
    