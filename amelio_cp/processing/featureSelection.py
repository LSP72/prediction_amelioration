# Feature Selection
from mrmr import mrmr_classif
from pandas import DataFrame, Series

def selector(allfeatures, label, number_of_important_features, *feature_names):
    # =======================================
    # Input:    features: pandas dataframe, indicating all features from which we want to select.
    #           labels: pandas dataframce, indicating the class of the data.
    #           number of important features: int, showing the number of important features to be selected. 
    #           feature names (not activated in this version): a list or tuple
    # -----------
    # Output:   The important features to be added to train model based on.
    # =======================================
    

    # for the time that feaetures and label are not pandas dataframes.
    # allfeatures = DataFrame(allfeatures, columns=feature_names)
    # label = Series(label)

    selectedFeatures= mrmr_classif(X=allfeatures, y=label, K=number_of_important_features)
    print("The selected features are:", selectedFeatures)
    return selectedFeatures