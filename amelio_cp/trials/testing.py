import sys
sys.path.append("/Users/mathildetardif/Documents/Python/Biomarkers/prediction_amelioration/functions")
import pandas as pd
import time
from model_functions.adding_noise import adding_noise
from model_functions.LinearModels import SVRModel, RFRModel, LRModel

nb_participants = 25
# Number of features
# Directory of the sample's folder
file_directory = r'/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/Data/Lokomat Data - SVR models/Sample%d/' %(nb_participants)
# Directory of the outputs
output_dir = r'/Users/mathildetardif/Library/CloudStorage/OneDrive-UniversitedeMontreal/Mathilde Tardif - PhD - Biomarkers CP/PhD projects/Training responders/Data/Lokomat Data - SVR models/outputs/'

all_data = pd.read_csv(output_dir+'all_data.csv')
all_data.drop('Unnamed: 0', axis=1, inplace = True)

VIT_POST = all_data['VIT_POST']
END_POST = all_data['6MWT_POST']

data = all_data.drop(['VIT_POST','6MWT_POST'], axis=1)

#%% Different datasets
# Dataset 1 - 'normal'
dataset_vit_1 = ['normal', data, VIT_POST]
dataset_end_1 = ['normal', data, END_POST]

# Dataset 2 - bruité
data_noised_vit, VIT_POST_noised = adding_noise(data, VIT_POST, 0.05)
data_noised_end, VIT_END_noised = adding_noise(data, END_POST, 0.05)

dataset_vit_2_noised = ['noised', data_noised_vit, VIT_POST_noised]
dataset_end_2_noised = ['noised', data_noised_end, VIT_END_noised]

# Dataset 3 - fortement bruité
data_noised_vit_2, VIT_POST_noised_2 = adding_noise(data, VIT_POST, 0.1)
data_noised_end_2, VIT_END_noised_2 = adding_noise(data, END_POST, 0.1)

dataset_vit_3_noised = ['more_noised', data_noised_vit_2, VIT_POST_noised_2]
dataset_end_3_noised = ['more_noised', data_noised_end_2, VIT_END_noised_2]

# Dataset 4 - difference in outcome
delta_VIT = all_data['VIT_POST'] - all_data['VIT_PRE']
dataset_vit_4_diff = ['diff', data, delta_VIT]

delta_END = all_data['6MWT_POST'] - all_data['6MWT_PRE']
dataset_end_4_diff = ['diff', data, delta_END]


#%% Lists of all the dataset that'll be tested
datasets_vit = [dataset_vit_1, dataset_vit_2_noised, dataset_vit_3_noised]
datasets_end = [dataset_end_1, dataset_end_2_noised, dataset_end_3_noised]


#%% SCRIPT - testing all configurations
results_vit_x100 = []

start = time.time()

for outcome in ['6MWT', 'VIT']:
    
    if outcome == '6MWT':
        datasets = datasets_end
    else: datasets = datasets_vit
    
    for dataset_id, dataset in enumerate(datasets):
        # initiating the models
        model_svr = SVRModel()
        model_rf = RFRModel()
        model_lr = LRModel()
        
       
        for model_name, model in {
            "SVR": model_svr,
            "RandomForest": model_rf,
            "LinearRegression": model_lr
        }.items():
            
            # filling the model with the dataset
            model.add_data(dataset[1], dataset[2])
        
            # Testing the perf of the models - check the overall perf of the model w/ nested CV
            metrics = model.perf_estimate(100)
            
            # adding info in the df for analysing after
            dataset_names = ['normal', 'noised', 'more_noised', 'diff']
            metrics.update({
                "outcome": outcome,
                "dataset_id": dataset_id,   # to know which dataset was used
                "dataset_name": dataset_names[dataset_id],
                "model": model_name
            })
            
            results_vit_x100.append(metrics)

end = time.time()
print(f'Time to complete the tests: {(end-start):.2f} seconds')
            
# mettre tous les résultats dans un DataFrame
df_results_vit_x100 = pd.DataFrame(results_vit_x100)

# affichage rapide
print(df_results_vit_x100.head())