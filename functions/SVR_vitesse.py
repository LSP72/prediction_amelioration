#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:22:17 2025

@author: mathildetardif
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
from scipy.stats import skewtest, kurtosistest
import sys

def prediction_vitesse(all_data):
    
    VIT_POST = all_data['VIT_POST']
    all_data.drop(['VIT_POST'], axis = 1, inplace = True)
    
    # --------------- Feature analysis ---------------


    # ----- Check the normality of features -----
    
    sktst = skewtest(all_data.values)
    print("Skewness values:\n", sktst.statistic)
    print("Skewness test p-values:\n ", sktst.pvalue)
    kurtst = kurtosistest(all_data.values)
    print("Kurtosis values:\n", kurtst.statistic)
    print("Kurtosis test p-values:\n", kurtst.pvalue)
    
    abnormally_distributed_indx = [-2 > x or x > 2 for x in sktst.statistic]
    abnormal_data_columns = [all_data.columns[i] for i in range(len(abnormally_distributed_indx)) if abnormally_distributed_indx[i]]
    abnormal_data = all_data[abnormal_data_columns]
    transformer = PowerTransformer(method='yeo-johnson')
    transformed_data = transformer.fit_transform(abnormal_data.values)
    all_data[abnormal_data_columns] = transformed_data
# 

# sktst = skewtest(all_data.values)
# print("Skewness values:\n", sktst.statistic)
# print("Skewness test p-values:\n ", sktst.pvalue)
# kurtst = kurtosistest(all_data.values)
# print("Kurtosis values:\n", kurtst.statistic)
# print("Kurtosis test p-values:\n", kurtst.pvalue)
