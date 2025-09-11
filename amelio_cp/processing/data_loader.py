import pandas as pd
import numpy as np
import os
import re
from scipy import interpolate
from sklearn import preprocessing
from MyfeaturExtractor import natural_sort_key


def matrix_reader(directory_str):

    all_data = pd.DataFrame()
    file_list = os.listdir(directory_str)
    file_list = [f for f in file_list if f.endswith('.csv')]
    csv_files = sorted(file_list, key = natural_sort_key)

    for file in csv_files:
        
        file_path = os.path.join(directory_str, file)
        csv = pd.read_csv(file_path)
        dependent_variables = csv
        independent_variable = pd.Series(range(len(dependent_variables)))     
        interpolation_function = pd.DataFrame()
        interpolated_dependent_variable = pd.DataFrame()

        for i in range(len(dependent_variables.columns)):
            scaled_independent_variable = np.linspace(0, len(dependent_variables) - 1, num=100)
            interpolation_function = interpolate.interp1d(independent_variable, dependent_variables.iloc[:, i], kind='cubic')
            interpolated_dependent_variable[f'{i + 1}'] = interpolation_function(scaled_independent_variable)
        
        scaler = preprocessing.StandardScaler()
        scaled_dependent_variables = pd.DataFrame(scaler.fit_transform(interpolated_dependent_variable))
        scaled_dependent_variables.columns = scaled_dependent_variables.columns.astype(str)
        all_data = pd.concat([all_data, scaled_dependent_variables], ignore_index=True)
        all_data.to_csv(directory_str+"allfiles.csv")
    
    return all_data

    print("--------------------------------")
    print("The shape of data is: {}".format(all_data.shape))
    print("--------------------------------")
    print("The data is now ready to be analyzed!")
