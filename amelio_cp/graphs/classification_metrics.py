from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class ClassifierMetrics:
    def __init__(self):
        pass

    @staticmethod
    def conf_matrix(model, y_true, y_pred, class_names: list, title: str, output_path=None):

        labels = class_names
        confusion_mat = confusion_matrix(y_true, y_pred)
        print(f"Confusion Matrix for:\n", confusion_mat)
        disp = ConfusionMatrixDisplay(confusion_mat, display_labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the size if necessary
        disp.plot(ax=ax)
        # Adjust font sizes for title, labels, and ticks
        ax.set_xlabel("Predicted Labels", fontsize=16)
        ax.set_ylabel("True Labels", fontsize=16)
        ax.set_title(title, fontsize=16)
        # Change font size for tick labels
        ax.tick_params(axis="both", labelsize=16)
        plt.yticks(rotation=90, va="center")
        # Adjust the font size of the numbers inside the matrix squares
        for text in ax.texts:
            text.set_fontsize(20)  # Change this value to your desired font size
            # Increase the font size of the color scale (colorbar)
        colorbar = disp.im_.colorbar
        colorbar.ax.tick_params(labelsize=16)  # Change 14 to your desired font size for color scale labels

        if output_path:
            plt.savefig(
                f"{output_path}confusion_matrix_{model.name}_{model.random_state}.svg", dpi=300, bbox_inches="tight"
            )
            print(f"Confusion matrix saved to: {output_path}")

        plt.show()
