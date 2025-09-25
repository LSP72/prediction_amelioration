import shap
from pandas import DataFrame
import matplotlib.pyplot as plt

class SHAPPlots:
    def __init__(self):
        pass

    @staticmethod
    def shap_values_calculation(trained_model, model_name, X_train, X_test):

        model = trained_model.model.named_steps[model_name]
        scaler = trained_model.model.named_steps['scaler']
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        explainer = shap.KernelExplainer(model.predict, X_train_scaled)
        shap_values = explainer.shap_values(X_test_scaled)

        return {'explainer': explainer,
                'shap_values': shap_values}
    
    @staticmethod
    def plot_shap_summary(train_model, model_name, features_names:list, X_train:DataFrame, X_test:DataFrame):
        
        # model = train_model.named_steps[model_name]
        scaler = train_model.model.named_steps['scaler']
        # X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # explainer = shap.KernelExplainer(model.predict, X_train_scaled)
        # shap_values = explainer.shap_values(X_test_scaled)
        
        shap_values = train_model.shap_analysis['shap_values']

        shap.summary_plot(
            shap_values, 
            X_test_scaled,
            feature_names=features_names,
            max_display=len(features_names),
            plot_size=(10, 12), 
            show=False  # Prevent SHAP from auto-displaying
            )
        
        plt.gca().set_yticklabels(plt.gca().get_yticklabels(), fontsize=26)
        for collection in plt.gca().collections:
            collection.set_sizes([100])

            # Increase x-label font size
        plt.xlabel("SHAP value (impact on model output)", fontsize=18)
        # Increase color bar label font size
        cbar = plt.gcf().axes[-1]  # The color bar is usually the last axis
        cbar.set_ylabel("Feature value", fontsize=18)  # Adjust the size as needed
        cbar.tick_params(labelsize=18)  # Adjust the size of the ticks (i.e., High/Low)
        plt.title("Weight of each feature on the ML's decision making", fontsize=20)
        plt.show()

    @staticmethod
    def plot_shap_bar(trained_model, features_names:list):
        
        shap_values = trained_model.shap_analysis['shap_values']

        if not isinstance(shap_values, shap.Explanation):
            shap_values_bar = shap.Explanation(shap_values, feature_names=features_names)
        
        shap.plots.bar(shap_values_bar, max_display=len(shap_values[0]), show= False)
        fig = plt.gcf()
        fig.set_size_inches(10, 20)
        plt.title("Weight of each feature on the ML's decision making", fontsize=25)
        plt.gca().tick_params(axis='y', labelsize=35)
        plt.show()