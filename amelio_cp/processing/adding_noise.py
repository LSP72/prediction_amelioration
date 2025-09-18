import numpy as np
import pandas as pd


def adding_noise(X, y, noise_factor):

    # Adding some noise
    X_aug = X + noise_factor * np.random.normal(size=X.shape)
    y_aug = y + noise_factor * np.random.normal(size=y.shape)

    # Concatenating the original and the noised one
    X_new = pd.concat([X, X_aug], ignore_index=True)
    y_new = pd.concat([y, y_aug], ignore_index=True)

    return X_new, y_new
