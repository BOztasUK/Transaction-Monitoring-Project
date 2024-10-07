import pandas as pd


def split_data(
    df, target_column, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1
):
    """
    Split the dataframe into train, validation, and test sets.

    Args:
        df (pd.DataFrame): The dataframe to split.
        target_column (str): The name of the target column.
        train_ratio (float): The proportion of the data to use for training.
        validation_ratio (float): The proportion of the data to use for validation.
        test_ratio (float): The proportion of the data to use for testing.

    Returns:
        tuple: Splits of the feature and target data (X_train, X_validation, X_test, y_train, y_validation, y_test).
    """
    assert train_ratio + validation_ratio + test_ratio == 1, "Ratios must sum to 1."

    X = df.drop(columns=[target_column])
    y = df[target_column]

    total_size = len(df)
    train_size = int(train_ratio * total_size)
    test_and_validation_size = total_size - train_size
    test_size = int(test_ratio * test_and_validation_size)
    validation_size = test_and_validation_size - test_size

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_validation = X[train_size : train_size + validation_size]
    y_validation = y[train_size : train_size + validation_size]

    X_test = X[train_size + validation_size :]
    y_test = y[train_size + validation_size :]

    return X_train, X_validation, X_test, y_train, y_validation, y_test
