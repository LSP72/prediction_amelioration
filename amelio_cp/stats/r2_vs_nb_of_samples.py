#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 16:56:36 2025

@author: mathildetardif
"""

nb_samples = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
r2_scores = dict()
rmse_scores = dict()

for i in nb_samples:
    SVR = SVRModel()
    SVR.add_data(x_train1[:i], y_train1[:i])
    training = SVR.train_and_tune(100)
    R2 = training["R²"]
    rmse = np.sqrt(training["MSE"])
    it = str(i)
    r2_scores[it] = R2
    rmse_scores[it] = rmse
