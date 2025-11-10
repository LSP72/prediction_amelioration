import numpy as np

from amelio_cp import LinearModel
from amelio_cp import SVRModel
from amelio_cp import ClassifierModel
from amelio_cp import SVCModel
from amelio_cp import Process


data_path = "examples/sandbox/fake_data_for_test.csv"
data = Process.load_csv(data_path)
features_path = "amelio_cp/processing/Features.xlsx"


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
                len(model.params_distributions["kernel"]),
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
                len(model.params_distributions["kernel"]),
            ]
        ),
        np.array([42, 1, 1000, 0.001, 0.1, 2, 5, 3]),
    )


def test_svr_model():
    model = SVRModel()

    model.best_params = {"C": 50, "gamma": 0.01, "epsilon": 0.1, "degree": 4, "kernel": "rbf"}
    model.model.set_params(**model.best_params)

    np.testing.assert_almost_equal(
        np.array([model.model.C, model.model.gamma, model.model.epsilon, model.model.degree]),
        np.array([50, 0.01, 0.1, 4]),
    )

    X, y, _ = Process.prepare_data(data_path, features_path, "VIT", model_name=model.name)
    model.add_data(X, y, test_size=0.2)

    np.testing.assert_equal(np.array(model.X_train.shape), np.array((80, 19)))
    np.testing.assert_equal(
        np.array(model.X_train.index),
        np.array(
            [
                89,
                26,
                42,
                70,
                15,
                40,
                72,
                9,
                96,
                11,
                91,
                64,
                28,
                83,
                5,
                47,
                53,
                35,
                16,
                81,
                34,
                7,
                43,
                73,
                27,
                19,
                94,
                25,
                62,
                49,
                13,
                24,
                3,
                17,
                38,
                8,
                79,
                6,
                65,
                36,
                88,
                56,
                100,
                54,
                50,
                68,
                46,
                69,
                61,
                98,
                80,
                41,
                58,
                48,
                90,
                57,
                75,
                32,
                95,
                59,
                63,
                85,
                37,
                29,
                1,
                52,
                21,
                2,
                23,
                87,
                99,
                74,
                86,
                82,
                20,
                60,
                71,
                14,
                92,
                51,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        model.X_train_scaled[0],
        np.array(
            [
                -1.50658960e00,
                -1.08876334e00,
                -6.53742618e-01,
                -1.88901776e00,
                9.65911502e-01,
                -1.03916133e00,
                1.27214018e00,
                -1.48061604e-01,
                4.55436226e-04,
                -5.05783029e-01,
                -1.07370965e-01,
                -2.80553179e-01,
                6.64556184e-01,
                1.29468029e-01,
                -1.77317741e00,
                -2.04696561e00,
                -4.34434708e-01,
                -9.10231720e-02,
                7.50639657e-01,
            ]
        ),
    )

    # TODO: X_test_scaled[i]


def test_svc_model():
    model = SVCModel()

    model.best_params = {"C": 50, "gamma": 0.01, "degree": 4, "kernel": "rbf"}
    model.model.set_params(**model.best_params)

    np.testing.assert_almost_equal(
        np.array([model.model.C, model.model.gamma, model.model.degree]),
        np.array([50, 0.01, 4]),
    )


# TODO: model.add_train_data and add_test_data tests w/ fake data and compare
