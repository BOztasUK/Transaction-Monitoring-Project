import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


def threshold_predictions(predictions, threshold):
    """
    Apply threshold to predictions to get binary class labels.
    """
    return (predictions >= threshold).astype(int)


def calculate_confusion_matrix_metrics(labels, pred_classes):
    """
    Calculate confusion matrix and derived metrics.
    """
    cm = confusion_matrix(labels, pred_classes)
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)

    return cm, tpr, fpr, fnr, tnr


def find_threshold_for_tpr(labels, predictions, desired_tpr):
    """
    Find the threshold that gives the desired TPR.
    """
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    # Find the threshold closest to the desired TPR
    idx = np.argmax(tpr >= desired_tpr)
    threshold = thresholds[idx]
    return threshold


def plot_confusion_matrix(cm, model_name, desired_tpr):
    """
    Plot confusion matrix using seaborn heatmap.
    """
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(f"Confusion Matrix for {model_name} at {desired_tpr * 100:.2f}% TPR")
    plt.show()


def plot_roc_curve(labels, predictions, model_name):
    """
    Plot ROC curve.
    """
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    thinner_lw = 1
    plt.plot(fpr, tpr, lw=thinner_lw, label=f"{model_name} (AUC = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="grey", lw=thinner_lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.legend(loc="lower right")
    plt.title(f"ROC Curve for {model_name}")
    plt.show()


def evaluate_model_performance(models, labels, desired_tpr):
    """
    Evaluate the performance of multiple models, find thresholds, plot ROC curves, and print confusion matrix metrics.
    """
    for model_name, predictions in models.items():
        print(f"Evaluating model: {model_name}")

        # Find the threshold that gives the desired TPR
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        idx = np.argmax(tpr >= desired_tpr)
        threshold = thresholds[idx]

        # Apply threshold to get binary class labels
        pred_classes = threshold_predictions(predictions, threshold)

        # Calculate confusion matrix metrics
        cm, tpr_value, fpr_value, fnr, tnr = calculate_confusion_matrix_metrics(
            labels, pred_classes
        )

        # Print confusion matrix and metrics
        print(f"Model: {model_name}")
        print(f"TPR (Sensitivity, Recall): {tpr_value:.4f}")
        print(f"FPR (Fall-out): {fpr_value:.4f}")
        print(f"FNR: {fnr:.4f}")
        print(f"TNR (Specificity): {tnr:.4f}")

        # Plot ROC curve
        plot_roc_curve(labels, predictions, model_name)

        # Plot confusion matrix
        plot_confusion_matrix(cm, model_name, desired_tpr)


def plot_multiple_roc_curves(model_predictions, labels, save_path=None):
    """
    Plot ROC curves for multiple models on the same graph.

    Args:
        labels (array-like): True labels.
        model_predictions (dict): Dictionary where keys are model names and values are predicted probabilities.
    """
    plt.figure()
    lw = 2
    thinner_lw = 1

    for model_name, predictions in model_predictions.items():
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=thinner_lw, label=f"{model_name} (AUC = %0.2f)" % roc_auc)

    plt.plot([0, 1], [0, 1], color="grey", lw=thinner_lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.legend(loc="lower right")
    plt.title("ROC Curve for Multiple Models")
    # plt.show()
