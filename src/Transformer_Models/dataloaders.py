import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class TabularDataset(Dataset):
    def __init__(self, categorical_data, numerical_data, labels):
        self.categorical_data = categorical_data
        self.numerical_data = numerical_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.categorical_data[idx], self.numerical_data[idx], self.labels[idx]


def prepare_data(
    df,
    categorical_columns,
    numerical_columns,
    label_column,
    batch_size=64,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
):
    assert (
        train_ratio + val_ratio + test_ratio == 1
    ), "Train, validation, and test ratios must sum to 1."

    categorical_data = df[categorical_columns].to_numpy(dtype=np.int64)
    numerical_data = df[numerical_columns].to_numpy(dtype=np.float32)
    labels = df[label_column].to_numpy(dtype=np.float32)

    train_size = int(train_ratio * len(df))
    val_size = int(val_ratio * len(df))
    test_size = len(df) - train_size - val_size

    train_dataset = TabularDataset(
        categorical_data[:train_size], numerical_data[:train_size], labels[:train_size]
    )
    val_dataset = TabularDataset(
        categorical_data[train_size : train_size + val_size],
        numerical_data[train_size : train_size + val_size],
        labels[train_size : train_size + val_size],
    )
    test_dataset = TabularDataset(
        categorical_data[train_size + val_size :],
        numerical_data[train_size + val_size :],
        labels[train_size + val_size :],
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
