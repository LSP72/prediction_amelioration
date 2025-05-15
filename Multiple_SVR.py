import shutil
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics, svm
from bayes_opt import BayesianOptimization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from scipy.stats import skewtest, kurtosistest
import sys
import shap
sys.path.append('/Users/mathildetardif/Documents/Python/Biomarkers/prediction_amelioration/functions/')
import MyfeaturExtractor, featureSelection
import data_preprocessing_matrix, interface
from imblearn.over_sampling import SMOTE # Synthetic Minority Oversampling TEchnique
                                         # Use to address imbalanced datasets for classification tasks (like here) => oversample the minority class by creating synth sample

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
        
## **** PARAMETERS INITIATION *****
# model = ''
# NbPps = 0
# model, NbPps = interface.window1(model, NbPps)
# Number of participants
nb_participants = 26
# Model used (in capital letters): 'SVM' or 'SVR'
model = 'SVM'
# Number of features
# Directory of the sample's folder
file_directory = r'/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/Data/Lokomat Data (matfiles)/SampleV/'
# Directory of the outputs
output_dir = r'/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/Data/outputsV/'


# ----- Extract variables from .mat files -----
# The output data will be in the order of all pre intervention files and then all post intervention files. 
# Since the folder only contains Pre data so we are gonna have the pre data only. 

all_data = data_preprocessing_matrix.kin_var_fct(file_directory = file_directory, 
                                         output_dir = output_dir,
                                         separate_legs = True,
                                         nb_participants = nb_participants)

# --------------- Feature analysis ---------------

# ----- Correlation -----
# correlation_matrix = all_data.corr()
# sns.heatmap(correlation_matrix, annot= True, cmap = 'coolwarm')
# plt.show()

# ----- Check the normality of features -----
# all_data_numeric = all_data.apply(pd.to_numeric, errors='coerce')
sktst = skewtest(all_data.values)
print("Skewness values:\n", sktst.statistic)
print("Skewness test p-values:\n ", sktst.pvalue)
kurtst = kurtosistest(all_data.values)
print("Kurtosis values:\n", kurtst.statistic)
print("Kurtosis test p-values:\n", kurtst.pvalue)

# abnormally_distributed_indx = [-2 > x or x > 2 for x in sktst.statistic]
# abnormal_data_columns = [all_data.columns[i] for i in range(len(abnormally_distributed_indx)) if abnormally_distributed_indx[i]]
# abnormal_data = all_data[abnormal_data_columns]
# transformer = PowerTransformer(method='yeo-johnson')
# transformed_data = transformer.fit_transform(abnormal_data.values)
# all_data[abnormal_data_columns] = transformed_data

# sktst = skewtest(all_data.values)
# print("Skewness values:\n", sktst.statistic)
# print("Skewness test p-values:\n ", sktst.pvalue)
# kurtst = kurtosistest(all_data.values)
# print("Kurtosis values:\n", kurtst.statistic)
# print("Kurtosis test p-values:\n", kurtst.pvalue)

## ----- Looking for the different predictions -----

# X_pp = row with all the info of one participant

vitesse_pred = SVR_vitesse.prediction_vitesse(X_pp, all_data)
_6MWT_pred = SVR_6MWT.prediction_6MWT(X_pp, all_data)

## Getting the pre-values
pre_vitesse = X_pp['VIT_PRE']
pre_6MWT = X_pp['6MWT_PRE']

temp = ['PRE', 'POST'] 
vitesse = [pre_vitesse, vitesse_pred]
vitesse_progression = 100*(vitesse_pred-pre_vitesse)/pre_vitesse
_6MWT = [pre_6MWT, _6MWT_pred]
_6MWT_progression = 100*(_6MWT_pred-pre_6MWT)/pre_6MWT

# Plotting the evolution
fig, axs = plt.subplots(2)
fig.suptitle('Evolution pre/post')
axs[0].plot(temp, vitesse)
for x, y in zip(temp, vitesse):
    axs[0].text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)
axs[0].text(0.5, 0.1, f"Evolution of {vitesse_progression:.2f} %", transform=axs[0].transAxes, fontsize=10, color='gray', ha='center')
axs[0].set_title('Vitesse')
axs[1].plot(temp, _6MWT)
for x, y in zip(temp, _6MWT):
    axs[1].text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)
axs[1].text(0.5, 0.1, f"Evolution of {_6MWT_progression:.2f} %", transform=axs[1].transAxes, fontsize=10, color='gray', ha='center')
axs[1].set_title('6MWT')
plt.show()
