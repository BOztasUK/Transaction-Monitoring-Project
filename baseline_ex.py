import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models.Baseline_algos import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    roc_curve,
)
from src.Baseline.Split_data_baseline import split_data
from src.Transformer_Models.evaluation import (
    evaluate_model_performance,
    plot_multiple_roc_curves,
)
from src.Baseline.train_and_evaluate import (
    train_and_evaluate_model,
    evaluate_models_test,
)
from src.utils import set_seeds

set_seeds(seed=28)

# Load and prepare the dataset
df = pd.read_pickle("/Users/berkan_oztas/TM-Project/data/processed/Processed_SAMLD.pkl")

iterations = 1
auc_df = pd.DataFrame()
learner = ClassificationAlgorithms()

target_column = "Is_laundering"

# Split the data into train, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    df, target_column, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1
)

# Define the models to be used
models_data = [
    ("RF", learner.random_forest),
    ("KNN", learner.k_nearest_neighbor),
    ("Decision Tree", learner.decision_tree),
    ("Naive Bayes", learner.naive_bayes),
    ("XGBoost", learner.xgboost_classifier),
    ("Logistic Regression", learner.logistic_regression),
]

val_predictions = {}
val_auc_results = []
trained_models = []

# Train and evaluate each model on the validation set
for model_name, model_func in models_data:
    val_probabilities, val_auc, trained_model = train_and_evaluate_model(
        model_name, model_func, X_train, y_train, X_val, y_val
    )
    val_predictions[model_name] = val_probabilities
    val_auc_results.append({"model": model_name, "auc": val_auc})
    trained_models.append((model_name, trained_model))

val_auc_df = pd.DataFrame(val_auc_results)

# Plot ROC curves for validation results
plot_multiple_roc_curves(val_predictions, y_val)

# Evaluate models on the test set and plot ROC curves
test_predictions, test_results = evaluate_models_test(trained_models, X_test, y_test)
plot_multiple_roc_curves(test_predictions, y_test)
