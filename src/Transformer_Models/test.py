import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random


def evaluate_model(model, test_dataloader, device):
    model.eval()
    test_labels = []
    test_predictions = []

    with torch.no_grad():
        for batch_categ, batch_numer, batch_labels in tqdm(
            test_dataloader, desc="Testing"
        ):
            batch_categ, batch_numer, batch_labels = (
                batch_categ.to(device),
                batch_numer.to(device),
                batch_labels.view(-1, 1).to(device),
            )

            predictions = model(batch_categ, batch_numer)

            test_labels.append(batch_labels.cpu().numpy())
            test_predictions.append(torch.sigmoid(predictions).cpu().detach().numpy())

    test_labels = np.vstack(test_labels)
    test_predictions = np.vstack(test_predictions)

    test_auc = roc_auc_score(test_labels, test_predictions)

    return test_auc, test_labels, test_predictions
