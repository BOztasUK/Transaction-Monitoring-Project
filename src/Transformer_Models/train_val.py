import copy
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import roc_auc_score, log_loss, roc_curve, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random


# Function to train the model for one epoch
def train_one_epoch(model, train_dataloader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    train_labels = []
    train_predictions = []

    for batch_categ, batch_numer, batch_labels in tqdm(
        train_dataloader, desc="Training"
    ):
        batch_categ, batch_numer, batch_labels = (
            batch_categ.to(device),
            batch_numer.to(device),
            batch_labels.view(-1, 1).to(device),
        )

        optimizer.zero_grad()
        predictions = model(batch_categ, batch_numer)
        loss = loss_function(predictions, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_labels.append(batch_labels.cpu().numpy())
        train_predictions.append(torch.sigmoid(predictions).cpu().detach().numpy())

    train_loss = total_loss / len(train_dataloader)
    train_auc = roc_auc_score(np.vstack(train_labels), np.vstack(train_predictions))

    return train_loss, train_auc, train_labels, train_predictions


def validate(model, val_dataloader, loss_function, device):
    model.eval()
    val_total_loss = 0
    val_labels = []
    val_predictions = []

    with torch.no_grad():
        for batch_categ, batch_numer, batch_labels in tqdm(
            val_dataloader, desc="Validation"
        ):
            batch_categ, batch_numer, batch_labels = (
                batch_categ.to(device),
                batch_numer.to(device),
                batch_labels.view(-1, 1).to(device),
            )

            predictions = model(batch_categ, batch_numer)
            loss = loss_function(predictions, batch_labels)

            val_total_loss += loss.item()
            val_labels.append(batch_labels.cpu().numpy())
            val_predictions.append(torch.sigmoid(predictions).cpu().detach().numpy())

    val_loss = val_total_loss / len(val_dataloader)
    val_auc = roc_auc_score(np.vstack(val_labels), np.vstack(val_predictions))
    val_log_loss = log_loss(np.vstack(val_labels), np.vstack(val_predictions))

    return val_loss, val_auc, val_log_loss, val_labels, val_predictions


def check_early_stopping(
    val_auc,
    best_val_auc,
    patience,
    wait,
    model,
    best_model_params,
    improvement_threshold,
):
    if val_auc - best_val_auc > improvement_threshold:
        best_val_auc = val_auc
        best_model_params = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            return True, best_val_auc, best_model_params, wait
    return False, best_val_auc, best_model_params, wait


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    patience,
    improvement_threshold,
    device,
    optimizer_class=Adam,
    learning_rate=0.0001,
    loss_function_class=BCEWithLogitsLoss,
):
    best_val_auc = 0.0
    wait = 0
    best_model_params = copy.deepcopy(model.state_dict())
    train_losses, val_losses, train_aucs, val_aucs = [], [], [], []
    all_train_labels, all_train_predictions = [], []
    all_val_labels, all_val_predictions = [], []

    model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    loss_function = loss_function_class().to(device)
    # optimizer = Adam(model.parameters(), lr=0.0001)
    # loss_function = BCEWithLogitsLoss().to(device)

    for epoch in range(num_epochs):
        train_loss, train_auc, train_labels, train_predictions = train_one_epoch(
            model, train_dataloader, optimizer, loss_function, device
        )
        val_loss, val_auc, val_log_loss, val_labels, val_predictions = validate(
            model, val_dataloader, loss_function, device
        )

        train_losses.append(train_loss)
        train_aucs.append(train_auc)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        all_train_labels.append(train_labels)
        all_train_predictions.append(train_predictions)
        all_val_labels.append(val_labels)
        all_val_predictions.append(val_predictions)

        print(
            f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}"
        )

        stop, best_val_auc, best_model_params, wait = check_early_stopping(
            val_auc,
            best_val_auc,
            patience,
            wait,
            model,
            best_model_params,
            improvement_threshold,
        )

        if stop:
            print(
                f"Stopping at epoch {epoch+1} due to no improvement. Final Training Loss: {train_loss:.4f}, Final Validation Loss: {val_loss:.4f}, Final Train AUC: {train_auc:.4f}, Final Val AUC: {val_auc:.4f}"
            )
            model.load_state_dict(best_model_params)
            break

    return (
        model,
        train_losses,
        val_losses,
        train_aucs,
        val_aucs,
        all_train_labels,
        all_train_predictions,
        all_val_labels,
        all_val_predictions,
    )
