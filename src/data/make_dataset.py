import pandas as pd
import numpy as np
from sklearn import preprocessing
import math
from scipy.stats import skew


raw_dataset = pd.read_csv("../../data/raw/SAML-D.csv")
df = raw_dataset

df["Hour"] = pd.to_datetime(df["Time"]).dt.hour

df["Date_Year"] = pd.to_datetime(df["Date"]).dt.year
df["Date_Month"] = pd.to_datetime(df["Date"]).dt.month
df["Date_Day"] = pd.to_datetime(df["Date"]).dt.day

df.drop(columns=["Laundering_type"], inplace=True)
df.drop(columns=["Time", "Date"], inplace=True)


skewed_data = df["Amount"]
original_skewness = skew(skewed_data)

log_transformed_data = np.log1p(skewed_data)
transformed_skewness = skew(log_transformed_data)

df["Amount"] = log_transformed_data


categorical_cols = [
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
]

for col in categorical_cols:
    encoder = preprocessing.LabelEncoder()
    df[col] = encoder.fit_transform(df[col])


numerical_cols = ["Hour", "Amount"]

scaler = preprocessing.StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

data_processed = df
data_processed.to_pickle("../../data/processed/01_data_processed.pkl")
