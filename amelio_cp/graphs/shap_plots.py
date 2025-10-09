import shap
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_is_fitted

# KEY = ['vitesse', ']


class SHAPPlots:
    def __init__(self):
        pass

    @staticmethod
    def shap_values_calculation(trained_model):

        try:
            check_is_fitted(trained_model.model)
        except:
            raise ValueError("❌ Model is not fitted yet!")

        explainer = shap.KernelExplainer(trained_model.model.predict, trained_model.X_train_scaled)  # bug potentiel
        shap_values = explainer.shap_values(trained_model.X_test_scaled)

        return {"explainer": explainer, "shap_values": shap_values}

    @staticmethod
    def plot_shap_summary(trained_model, features_names: list, output_path: str):

        shap_values = trained_model.shap_analysis["shap_values"]

        shap.summary_plot(
            shap_values,
            trained_model.X_test_scaled,
            feature_names=features_names,  # model.feature_keys
            max_display=len(features_names),
            plot_size=(8, 10),
            show=False,  # Prevent SHAP from auto-displaying
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
        plt.title(
            f"Weight of each feature on the ML's decision making \n(random state = {trained_model.random_state})",
            fontsize=20,
        )

        # Saving the figure if a path is provided
        if output_path:
            plt.savefig(f"{output_path}shap_fig_{trained_model.random_state}.png", dpi=300, bbox_inches="tight")
            print(f"SHAP plot saved to: {output_path}")

        plt.show()

    @staticmethod
    def plot_shap_bar(trained_model, features_names: list):

        shap_values = trained_model.shap_analysis["shap_values"]

        if not isinstance(shap_values, shap.Explanation):
            shap_values_bar = shap.Explanation(shap_values, feature_names=features_names)

        shap.plots.bar(shap_values_bar, max_display=len(shap_values[0]), show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 20)
        plt.title("Weight of each feature on the ML's decision making", fontsize=25)
        plt.gca().tick_params(axis="y", labelsize=35)
        plt.show()
