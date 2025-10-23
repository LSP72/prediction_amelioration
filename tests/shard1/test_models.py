import numpy as np

from amelio_cp.models.linear.linear_model import LinearModel
#%%
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
                model.params_distributions["degree"][0],
                model.params_distributions["degree"][1],
            ]
        ),
        np.array([42, 1, 1000, 0.001, 0.1, 2, 5]),
    )

    #TODO: model.add_train_data and add_test_data tests w/ fake data and compare
# %%
