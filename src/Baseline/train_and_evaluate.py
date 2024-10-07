from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np


import shap  # Import SHAP library


def train_and_evaluate_xgb(
    model_name,
    model_func,
    X_train,
    y_train,
    X_validation,
    y_validation,
    gridsearch=True,
):
    print(f"\tTraining {model_name}")
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y, model = (
        model_func(X_train, y_train, X_validation, gridsearch=gridsearch)
    )
    test_probabilities = model.predict_proba(X_validation)[:, 1]
    test_auc = roc_auc_score(y_validation, test_probabilities)

    # Calculate SHAP values using the Tree Explainer if model is tree-based
    if "XGB" in model_name or "LGBM" in model_name or "RF" in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train, plot_type="bar")

        # Optionally, create a DataFrame for detailed feature importance
        shap_df = pd.DataFrame(
            {
                "Feature": X_train.columns,
                "SHAP Importance": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values(by="SHAP Importance", ascending=False)
        print(f"SHAP Feature Importance for {model_name}:\n{shap_df}\n")

    return test_probabilities, test_auc, model, shap_df  # Return the model as well


def train_and_evaluate_model(
    model_name,
    model_func,
    X_train,
    y_train,
    X_validation,
    y_validation,
    gridsearch=True,
):
    print(f"\tTraining {model_name}")
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y, model = (
        model_func(X_train, y_train, X_validation, gridsearch=gridsearch)
    )
    test_probabilities = model.predict_proba(X_validation)[:, 1]
    test_auc = roc_auc_score(y_validation, test_probabilities)

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame(
            {"Feature": X_train.columns, "Importance": importance}
        ).sort_values(by="Importance", ascending=False)
        print(f"Feature Importance for {model_name}:\n{feature_importance_df}\n")

    return test_probabilities, test_auc, model  # Return the model as well


def evaluate_models_test(models, X_test, y_test):
    """
    Evaluate a list of models on the test dataset and print their performance metrics.

    Args:
        models (list): A list of tuples containing (model_name, model_object).
        X_test (DataFrame): Test features.
        y_test (Series): True labels for the test data.

    Returns:
        dict: A dictionary containing model predictions and performance metrics.
    """
    test_results = {}
    models_predictions_test = {}

    # Iterate through each model and evaluate it
    for model_name, model in models:
        # Predict labels and probabilities
        test_predictions = model.predict(X_test)
        test_probabilities = model.predict_proba(X_test)[:, 1]

        # Store predictions for possible further evaluations
        models_predictions_test[model_name] = test_probabilities

        # Compute metrics
        test_auc = roc_auc_score(y_test, test_probabilities)
        test_accuracy = accuracy_score(y_test, test_predictions)

        # Store results
        test_results[model_name] = {"AUC": test_auc, "Accuracy": test_accuracy}

    # Optionally print results
    for model_name, results in test_results.items():
        print(
            f"{model_name} - AUC: {results['AUC']:.4f}, Accuracy: {results['Accuracy']:.4f}"
        )

    return models_predictions_test, test_results
