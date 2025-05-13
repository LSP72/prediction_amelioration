#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:22:17 2025

@author: mathildetardif
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
from scipy.stats import skewtest, kurtosistest
from bayes_opt import BayesianOptimization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn import metrics, svm
import sys

def prediction_vitesse(all_data):
    # 'all_data' is a big matrix with all features in columns and all sample in rows
    # from this matrix can be extracted the labels (i.e., VIT_POST)
    
    VIT_POST = all_data['VIT_POST']
    all_data.drop(['VIT_POST'], axis = 1, inplace = True)
    
    # --------------- Feature analysis ---------------


    # ----- Check the normality of features -----
    # print('¨***************')
    # print('Checking the normality of the features for speed')
    # sktst = skewtest(all_data.values)
    # print("Skewness values:\n", sktst.statistic)
    # print("Skewness test p-values:\n ", sktst.pvalue)
    # kurtst = kurtosistest(all_data.values)
    # print("Kurtosis values:\n", kurtst.statistic)
    # print("Kurtosis test p-values:\n", kurtst.pvalue)
    
    # abnormally_distributed_indx = [-2 > x or x > 2 for x in sktst.statistic]
    # abnormal_data_columns = [all_data.columns[i] for i in range(len(abnormally_distributed_indx)) if abnormally_distributed_indx[i]]
    # abnormal_data = all_data[abnormal_data_columns]
    # transformer = PowerTransformer(method='yeo-johnson')
    # transformed_data = transformer.fit_transform(abnormal_data.values)
    # all_data[abnormal_data_columns] = transformed_data

    # --------------- ML algorithm: Support Vector Regresson (SVR) ---------------
    # ----- Selecting features based on earlier choisice for now may be by PCA then ------
    # https://chatgpt.com/share/68234db6-78bc-8013-a689-8b8f146283a2
    features = pd.read_excel(r'/Users/mathildetardif/Documents/Python/Biomarkers/responder_prediction/functions/Features.xlsx')
    selectedFeatures = features['18']
    selectedFeatures = selectedFeatures.dropna()
    selected_data_df = all_data[selectedFeatures]
    selected_data = selected_data_df.values
    
    # ----- Train / test split -----
    # Going to split the data for training/testing
    # Creating bins of VIT_POST
    bins = pd.qcut(VIT_POST, q=5, labels=False)  # Quantile binning into 5 groups

    x_train, x_test, y_train, y_test = train_test_split(selected_data, VIT_POST, test_size=0.25, random_state=72, stratify=bins)    
    #                                                                                                               Using the bins for stratification
    # ----- Scaling the data -----
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train_scaled = scaler_x.fit_transform(x_train) # standardise/normalise the data based in the input by calculatG mean & SD of x_train & then, stdise x_train
    y_train_scaled = scaler_y.fit_transform(y_train.array.reshape(-1,1))    # standardise y_train
    x_test_scaled = scaler_x.transform(x_test)                              # standardise x_test

    # ----- Applying SMOTE (to balance data sets) -----
    # IGNORED HERE bcs used when doing classification
    #smote_train = SMOTE(sampling_strategy='auto',   random_state=42,                n_jobs=-1)
    #                                   'auto' will adjust to have the same amount 
    #ratio of the min class to the maj after resampling # ctrtls random°/ensurG reprodY #nb of // jobs for computation
    #x_train, y_train = smote_train.fit_resample(x_train, y_train) # resample the data sets to have new 'balanced' data sets

    # ----- ML -----
    # Finding an optimised ALPHA w/ Bayesian Optim
 
    errors = []
    BP = {}
    SVR_acc = {} 
 
    pbounds = {
                'C': (1,1000),
                'gamma': (0.001, 1),
                'degree': (2,5),
                'epsilon': (0.01, 1)
                }
     
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    
    for kernel in kernels:
        
        def svr_model(C, gamma, degree, epsilon):
            SVR = svm.SVR(kernel=kernel, C=C, gamma=gamma, degree=int(degree), epsilon=epsilon)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=72)
            scores = cross_val_score(SVR, x_train_scaled, y_train_scaled, cv=cv, scoring = 'neg_mean_squared_error')
            return scores.mean()
        
        print('************************************')
        print('Bayesian Optimization initiated for', kernel, '.')
        
        optimizer = BayesianOptimization(f = svr_model, pbounds = pbounds,                random_state=72, verbose=3)
                                  # the objve f° #a dict specifiG range for each param #for reprodY   #ctrls amount of info displayed 
        optimizer.maximize(init_points=10, n_iter=100) # For 10 features accuracy:91
        best_parameters = optimizer.max['params']
        best_parameters['C'] = float(best_parameters['C'])
        best_parameters['degree'] = int(best_parameters['degree'])
        best_parameters['epsilon'] = float(best_parameters['epsilon'])
        print("For", kernel, "the best hyperparameters are:", best_parameters, ".")
        BP[kernel] = optimizer.max  # Storing best parameters
       
        SVR_best = svm.SVR(kernel=kernel, C=float(best_parameters['C']), gamma=best_parameters['gamma'], degree=int(best_parameters['degree']), epsilon=float(best_parameters['epsilon']))
        SVR_best.fit(x_train_scaled, y_train_scaled)
        
        y_hat = SVR_best.predict(x_test_scaled)
        # Reversing because data has been scaled
        y_hat_rev = scaler_y.inverse_transform(y_hat.reshape(-1, 1))
        
        
        # Evalutation of the model on the test set
        mse = mean_squared_error(y_test, y_hat_rev)
        errors.append({kernel: mse})
        # r2_train = r2_score(y_train, y_hat_train)
        r2 = r2_score(y_test, y_hat_rev)
        SVR_acc[kernel] = r2
        print(f"Final Test MSE for {kernel}: {mse}")
        # print(f"Train R2 score for {kernel}: {r2_train}")
        print(f"Test R2 score for {kernel}: {r2}")
        
        # Plotting the figure
        fig, ax = plt.subplots()
        ax.plot(y_test.index.to_list(), y_test, '.', color='r')
        ax.plot(y_test.index.to_list(), y_hat_rev, '.', color='b')
        ax.legend(['True values','Predictions'])
        plt.title(f"True and predicted speed for {kernel}")
        plt.show()
        
    # Creating a readable matrix with all the info
    BP = pd.DataFrame(BP).T
    params = BP['params'].apply(pd.Series)
    BP = pd.concat([BP[['target']], params], axis=1)
    BP['degree'] =  BP['degree'].apply(lambda x : int(x))
    BP['accuracy'] = SVR_acc
    BP['RMSE'] = errors
    
    return BP