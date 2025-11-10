import numpy as np

from amelio_cp import LinearModel
from amelio_cp import SVRModel
from amelio_cp import ClassifierModel
from amelio_cp import SVCModel
from amelio_cp import Process

data_path = "examples/sandbox/fake_data_for_test.csv"
data = Process.load_csv(data_path)
features_path = "amelio_cp/processing/Features.xlsx"


def test_optimisation_svc_model():
    model = SVCModel()

    X, y, _ = Process.prepare_data(data_path, features_path, "VIT", model_name=model.name)
    model.add_data(X, y, test_size=0.2)

    # model.train_and_tune("bayesian_optim", n_iter=50)
    # y_pred = model.model.predict(model.X_test_scaled)

    # np.testing.assert_equal(
    #     np.array([model.model.C, float(model.model.gamma), model.model.degree]),
    #     np.array([169.63021952277953, 0.10992366904399258, 3]),
    # )

    # np.testing.assert_equal(np.array(y_pred), np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1]))

    model.train_and_tune("bayesian_search", n_iter=10)
    np.testing.assert_equal(
        np.array([model.model.C, model.model.gamma, model.model.degree]),
        np.array([410.6938548944605, 0.09335393188593556, 4]),
    )
    y_pred = model.model.predict(model.X_test_scaled)

    np.testing.assert_equal(np.array(y_pred), np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]))
