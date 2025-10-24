import numpy as np
import pandas as pd

from amelio_cp.models.linear.linear_model import LinearModel
from amelio_cp.models.linear.svr_model import SVRModel
from amelio_cp.models.classification.classifier_model import ClassifierModel
from amelio_cp.models.classification.svc_model import SVCModel


data_path = "examples/sandbox/fake_data_for_test.xlsx"
data = pd.read_excel(data_path)

def test_linear_model():
    model = LinearModel()

    np.testing.assert_almost_equal(
        np.array(
            [
                model.random_state,
                model.params_distributions["C"][0],
                model.params_distributions["C"][1],
                model.params_distributions["gamma"][0],
                model.params_distributions["gamma"][1],
                model.params_distributions["epsilon"][0],
                model.params_distributions["epsilon"][1],
                model.params_distributions["degree"][0],
                model.params_distributions["degree"][1],
                len(model.params_distributions["kernel"])
            ]
        ),
        np.array([42, 1, 1000, 0.001, 0.1, 0.01, 1, 2, 5, 3]),
    )

def test_classifier_model():
    model = ClassifierModel()

    np.testing.assert_almost_equal(
        np.array(
            [
                model.random_state,
                model.params_distributions["C"][0],
                model.params_distributions["C"][1],
                model.params_distributions["gamma"][0],
                model.params_distributions["gamma"][1],
                model.params_distributions["degree"][0],
                model.params_distributions["degree"][1],
                len(model.params_distributions["kernel"])
            ]
        ),
        np.array([42, 1, 1000, 0.001, 0.1, 2, 5, 3]),
    )

def test_svr_model():
    model = SVRModel()

    model.best_params = {
        "C": 50,
        "gamma": 0.01,
        "epsilon": 0.1,
        "degree": 4,
        "kernel": "rbf"
    }
    model.model.set_params(**model.best_params)

    np.testing.assert_almost_equal(
        np.array(
            [
                model.model.C,
                model.model.gamma,
                model.model.epsilon,
                model.model.degree
            ]
        ),
        np.array([50, 0.01, 0.1, 4]),
    )
    

def test_svc_model():
    model = SVCModel()

    model.best_params = {
        "C": 50,
        "gamma": 0.01,
        "degree": 4,
        "kernel": "rbf"
    }
    model.model.set_params(**model.best_params)

    np.testing.assert_almost_equal(
        np.array(
            [
                model.model.C,
                model.model.gamma,
                model.model.degree
            ]
        ),
        np.array([50, 0.01, 4]),
    )



# TODO: model.add_train_data and add_test_data tests w/ fake data and compare
