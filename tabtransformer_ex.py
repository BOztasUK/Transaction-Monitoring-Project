import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
import random
from src.Transformer_Models.dataloaders import prepare_data
from models.TabTransformer_Model import TabTransformer
from src.Transformer_Models.train_val import train_model
from src.Transformer_Models.test import evaluate_model
from src.Transformer_Models.evaluation import (
    evaluate_model_performance,
    plot_multiple_roc_curves,
)
from src.utils import set_seeds


# Set random seed for reproducibility
set_seeds(seed=28)


# Load and preprocess data
df = pd.read_csv(
    "/Users/berkan_oztas/TM-Project/data/processed/FINALDATASET.csv (1).csv_"
)

# Define column types
categorical_columns = [
    "Sender_account",
    "Receiver_account",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
    "Date_Year",
    "Date_Month",
    "Date_Day",
    "Hour",
]
numerical_columns = ["Amount"]
label_column = "Is_laundering"

# Prepare data loaders
train_dataloader, val_dataloader, test_dataloader = prepare_data(
    df,
    categorical_columns,
    numerical_columns,
    label_column,
    batch_size=128,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)


# Define model configurations for different dimensions
model_configs = {
    16: {"dim": 16},
    32: {"dim": 32},
    64: {"dim": 64},
    128: {"dim": 128},
}


# Function to create and train a model
def create_and_train_model(dim):
    model = TabTransformer(
        categories=[len(df[col].unique()) for col in categorical_columns],
        num_continuous=len(numerical_columns),
        depth=2,
        heads=4,
        dim_out=1,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_act=nn.GELU(),
        **model_configs[dim],
    )

    return train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=4,
        patience=1,
        improvement_threshold=1e-4,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        optimizer_class=AdamW,
        learning_rate=0.001,
        loss_function_class=BCEWithLogitsLoss,
    )


# Train models for each dimension
models = {}
test_results = {}
device = "mps" if torch.backends.mps.is_available() else "cpu"

for dim in model_configs.keys():
    model_name = f"TabTransformer_{dim}"
    models[model_name], *_ = create_and_train_model(dim)
    test_results[model_name] = evaluate_model(
        model=models[model_name], test_dataloader=test_dataloader, device=device
    )


# Plot ROC curves for all models
model_predictions = {name: results[2] for name, results in test_results.items()}
plot_multiple_roc_curves(
    model_predictions=model_predictions, labels=test_results["TabTransformer_16"][1]
)

# Evaluate model performance at a specific True Positive Rate
desired_tpr = 0.98
models = {
    f"TabTransformer_{d_model}": test_results[f"TabTransformer_{d_model}"][2]
    for d_model in model_configs.keys()
}

evaluate_model_performance(
    models=models,
    labels=test_results["TabTransformer_32"][1],
    desired_tpr=desired_tpr,
)
