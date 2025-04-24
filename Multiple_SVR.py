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

# --------------- ML algorithm: Support Vector Machine (SVM) ---------------
# ----- Select features based on MRMR algorithm ------
features = pd.read_excel(r'/Users/mathildetardif/Documents/Python/Biomarkers/responder_prediction/functions/Features.xlsx')
selectedFeatures = features['18']
selectedFeatures = selectedFeatures.dropna()
# selectedFeatures = featureSelection.selector(allfeatures=all_data, label=MCID, number_of_important_features= len(all_data.columns))
selected_data_df = all_data[selectedFeatures]

# correlation_matrix = selected_data.corr()
# sns.heatmap(correlation_matrix, annot= True, cmap = 'coolwarm')
# plt.show()
selected_data = selected_data_df.values

# ----- Train / test Split -----
# Going to split the data for training/testing
x_train, x_test, y_train, y_test = train_test_split(selected_data, MCID, test_size = 0.25, random_state=72, stratify=MCID)
                                                    #inputs     #outputs #size of test set #making the split reproductible #ensure proportions are same for train/test sets
# ----- Scale the data -----
# normalizer = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # standardise/normalise the data based in the input by calculatG mean & SD of x_train & then, stdise x_train
x_test = scaler.transform(x_test)       # standardise x_test

# Applying SMOTE
smote_train = SMOTE(sampling_strategy='auto',   random_state=42,                n_jobs=-1)
#                                was = {1:27,0:20}
#ratio of the min class to the maj after resampling # ctrtls random°/ensurG reprodY #nb of // jobs for computation
# maj needs to be changed regarding the size of the sample
# 21p => 24 /// 23p => 27
x_train, y_train = smote_train.fit_resample(x_train, y_train) # resample the data sets to have new 'balanced' data sets


# ----- Bayesian Optimization -----
# Method to find the best paramaters for a model (minG/maxG an objve f°)

# pbounds = {
#             'C': (1,1000),
#             'gamma': (0.001, 1),
#             'degree': (2,5),
#             'kernel_idx':(0,3)
#             }
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# errors = []
# BP = {}
# SVM_acc = {} 

# def svm_model(C, gamma, degree, kernel_idx):
#     kernel = kernels[int(round(kernel_idx))]
#     SVM = svm.SVC(kernel=kernel, C=C, gamma=gamma, degree=int(degree), random_state=42)
    
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     # cv_LOO = LeaveOneOut()
#     acc = cross_val_score(SVM, x_train, y_train, cv=cv, scoring = 'accuracy')
#     return acc.mean()
    
# print('************************************')
# print('Bayesian Optimization initiated.')

# optimizer = BayesianOptimization(f = svm_model, pbounds = pbounds,                random_state=42, verbose=3)
#                           # the objve f° #a dict specifiG range for each param #for reprodY   #ctrls amount of info displayed 
# optimizer.fit(x_train, y_train)
# optimizer.maximize(init_points=10, n_iter=100) # For 10 features accuracy:91
# results = pd.DataFrame(optimizer.res)

param_grid = {
    'C' :(1,1000),
    'gamma': (0.001, 1),
    'degree': (2,5),
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid']
    }
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(x_train, y_train)
best_svm = grid_search.best_estimator
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# best_parameters = optimizer.max['params']
# best_parameters['C'] = float(best_parameters['C'])
# best_parameters['degree'] = int(best_parameters['degree'])
# kernel = kernels[int(round(best_parameters['kernel_idx']))]
# print("Best hyperparameters found:", best_parameters," \n and the kernel is: ", kernel, ".")
# BP[kernel] = optimizer.max 

   # Train the final SVM model with the best parameters
SVM_best = svm.SVC(kernel=kernel, C=best_parameters['C'], gamma=best_parameters['gamma'], degree=best_parameters['degree'], random_state=42)
SVM_best.fit(x_train, y_train)
    
y_hat = SVM_best.predict(x_test)

# Evalutation of the model on the test set
SVM_acc[kernel] = accuracy_score(y_test, y_hat)
mse = mean_squared_error(y_test, y_hat)
errors.append({kernel: mse})
print(f"Final Test Accuracy for {kernel}: {SVM_acc[kernel]}")
print("Test set score: ", SVM_best.score(x_test, y_test))
print(metrics.classification_report(y_test, y_hat))
    # y_probs = SVM_best.decision_function(x_test)
    # print("AUC value: ", roc_auc_score(y_test, y_probs))

# ----- Parameter testing to CSV -----
# # If want to recover all the parameters that have been tested during the 5CV 
# # for each kernel 
# # TO ADD IN THE LOOP
# params = pd.DataFrame(results['params'].to_list())
# results = pd.concat([results['target'], params, pd.DataFrame([best_parameters])], axis = 1)
# results.to_csv('/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/MyScripts/Bayesian_Optim_%s.csv' %(kernel))



# Recovery of the best parameters
only_BP = {}
for kernel in kernels:
    only_BP[kernel] = BP[kernel]['params']

only_BP_df = pd.DataFrame(only_BP, index = ['C', 'degree', 'gamma'])
only_BP_df.to_csv('/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/MyScripts/BestParameters_%s.csv' %(smote_train.sampling_strategy))

# mean_error = np.mean(errors)
# print(f'Average Mean Squared Error: {mean_error}')


# ----- Confusion Matrix -----
labels = ['Non-responder', 'Responder']
kernel = 'rbf'
confusion_mat = confusion_matrix(y_true=y_test, y_pred=y_hat)
disp = ConfusionMatrixDisplay(confusion_mat, display_labels=labels)
disp.plot()
plt.title('Confusion Matrix for %s Kernel' %(kernel))
plt.show()
TN, FP, FN, TP = confusion_mat.ravel()
print(f"Vrais positifs (TP) : {TP}")  # Bien classés comme positifs
print(f"Faux positifs (FP) : {FP}")  # Négatifs classés à tort comme positifs
print(f"Faux négatifs (FN) : {FN}")  # Positifs classés à tort comme négatifs
print(f"Vrais négatifs (TN) : {TN}")  # Bien classés comme négatifs


# ----- Calculate SHAP values for the best non-linear model -----
explainer = shap.KernelExplainer(SVM_best.predict, x_train)
shap_values = explainer.shap_values(x_test)
base_value = explainer.expected_value  
# if not isinstance(shap_values, shap.Explanation):
#     shap_values = shap.Explanation(shap_values, feature_names=selectedFeatures)

# Plot SHAP values
shap.summary_plot(shap_values, x_test, feature_names=selectedFeatures)



## Graphs
# ----- Waterfall plots -----
for i in range(len(shap_values)):
    shap_exp = shap.Explanation(        # will transform numpy.arry in explanation obj
        values=shap_values,          # SHAP values for the first instance
        base_values=base_value,        # Baseline prediction
        data=x_test,                    # Feature values for the instance
        feature_names=selectedFeatures) # Custom feature names
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_exp[i], show = False)
    plt.title("Waterfall plot for patient %s" %(y_test.index[i]), fontsize=16, pad=20)  # Customize title
    plt.show()

# ----- Bar plot-----
# To use if shap_values has been transformed in an Explanation Object
if not isinstance(shap_values, shap.Explanation):
     shap_values = shap.Explanation(shap_values, feature_names=selectedFeatures)
shap_values.feature_names = selectedFeatures
shap.plots.bar(shap_values, max_display=len(shap_values[0]))
plt.show()                              # IF shap_values is an Explanation Object
                                        # => no need of ".data"

# ----- Beeswarm plot -----
# summary one

# ----- Dependance plot -----
shap.plots.scatter(shap_values)                         # for all features
shap.plots.scatter(shap_values[:,'pctSimpleAppuie'])    # for specific features

# ----- Decision plot highlighting misclassified -----
select = range(len(y_test))
y_pred = (shap_values.sum(1) + base_value) > 0
misclassified = y_pred != y_test.iloc[select]
shap.decision_plot(base_value, shap_values, selectedFeatures, link="logit", highlight=misclassified)

