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
import MyfeaturExtractor, featureSelection


# ** /!\ ATTENTION à l'ordre R/L (dans les autres fonctitons R PUIS L, donc doit être dans le même sens dans le Excel !) **


def kin_var_fct(file_directory, output_dir, separate_legs, nb_participants):
    measurements = ['pctSimpleAppuie', 'distFoulee', 'vitCadencePasParMinute']
    joint_names = ['Hip', 'Knee', 'Ankle']
    side = ['Right', 'Left']
    # Reinitiating the output directory
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Extracting variables from .mat files
    kin_var = MyfeaturExtractor.MinMax_feature_extractor(directory = file_directory,
                                                     measurements = measurements,
                                                     output_dir = output_dir,
                                                     separate_legs = separate_legs,
                                                     output_shape = pd.DataFrame,
                                                     joint_names = joint_names)
    
    kin_var['ROM_Hip_Sag'] = kin_var['Max_Hip_flx/ext'] - kin_var['Min_Hip_flx/ext']
    kin_var['ROM_Hip_Frontal'] = kin_var['Max_Hip_abd/add'] - kin_var['Min_Hip_abd/add']
    kin_var['ROM_Hip_Trns'] = kin_var['Max_Hip_int/ext rot'] - kin_var['Min_Hip_int/ext rot']
    kin_var['ROM_Knee_Sag'] = kin_var['Max_Knee_flx/ext'] - kin_var['Min_Knee_flx/ext']
    kin_var['ROM_Ankle_Sag'] = kin_var['Max_Ankle_flx/ext'] - kin_var['Min_Ankle_flx/ext']
    
    # kin_var.drop(['Max_Hip_flx/ext', 'Min_Hip_flx/ext', 'Max_Hip_abd/add', 'Min_Hip_abd/add',
    #               'Max_Hip_int/ext rot', 'Min_Hip_int/ext rot', 'Max_Knee_flx/ext', 'Min_Knee_flx/ext',
    #               'Max_Ankle_flx/ext', 'Min_Ankle_flx/ext',
    #               'Min_Knee_abd/add', 'Min_Knee_int/ext rot', 'Min_Ankle_abd/add', 'Min_Ankle_int/ext rot',
    #               'Max_Knee_abd/add', 'Max_Knee_int/ext rot', 'Max_Ankle_abd/add', 'Max_Ankle_int/ext rot'],
    #               axis=1,
    #               inplace=True)
    
    kin_var.drop(['Min_Knee_abd/add', 'Min_Knee_int/ext rot', 'Min_Ankle_abd/add', 'Min_Ankle_int/ext rot',
                  'Max_Knee_abd/add', 'Max_Knee_int/ext rot', 'Max_Ankle_abd/add', 'Max_Ankle_int/ext rot'],
                  axis=1,
                  inplace=True)
    
    # ----- fixing the values of cadance -----
    kin_var['vitCadencePasParMinute'] *= 2
    
    # ----- Add GPS to the features -----
    #       Gait Profile Score    
    gps = pd.read_csv(r'/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/MyScripts/Functions/GPS_output%d.csv' %(nb_participants), index_col=False)
    gps.drop(columns=['Unnamed: 0', 'Post'], inplace=True)
    gps.rename(columns={"Pre": "GPS"}, inplace=True)
    all_data = pd.concat((kin_var,gps), axis=1)
    
    # ----- Add participants demographic variables -----
    demo_var = pd.read_excel(r'/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/Data/Lokomat Data (matfiles)/Sample%d/MyParticipants_caracteristics_ML_vitesse%d.xlsx' %(nb_participants, nb_participants))    
    demo_var['BMI'] = demo_var['masse']/((demo_var['taille']/100)**2)
    demo_var.replace(['walker', 'cane', 'none'], [1,2,3], inplace=True) # The numbers are in the order so 1 is walker.
    # ********* ACTION REQUIRED *********:
    #           IF RUNNING WITH separate_legs = False, DEACTIVATE THE FOLLOWING LINE.
    #           OTHERWISE, KEEP IT ACTIVATED.
    # demo_var = demo_var.loc[demo_var.index.repeat(2)].reset_index(drop=True)
    
    # ----- Add Label -----
    demo_var.drop(['delta'], axis = 1, inplace= True)
    demo_var.drop(['Patient', 'masse', 'taille', 'sex','Diagnostique'], axis = 1, inplace= True) #This line excludes some of the features that are in the demographic excel file.
    demo_var.drop(['delta'], axis = 1, inplace= True)
    all_data = pd.concat((all_data, demo_var), axis=1)
    
    return all_data
