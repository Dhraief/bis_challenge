
import polars as pl
import torch
import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
import pickle
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GENERAL VARIABLES
ACCOUNT_DATASET_PATH = "/dccstor/aml_datasets/bis/data/raw/train_target_dataset.parquet"
OUTPUT_PATH = "/dccstor/aml_datasets/bis"
N_TRANSACTIONS = 288785789  

# LOADING DATA
logging.info("Loading datasets...")
train_target = pl.read_parquet(ACCOUNT_DATASET_PATH)#.head(10000)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# load mapping:
logging.info("Loading mapping for transaction_id...")
with open(os.path.join(OUTPUT_PATH, "mapping_transaction.pkl"), "rb") as f:
    mapping_transaction = pickle.load(f)


# TRANSOFMRATIONS
train_target = train_target.with_columns(
    pl.col("transaction_id").replace(mapping_transaction).alias("transaction_id_mapped")
)

logging.info("Mapping transaction_id to integers...")
train_target = train_target.with_columns(
    train_target["laundering_schema_type"]
    .str.split("_")  # Split by underscore
    .list.get(-1)    # Get the last element
    .cast(pl.Int8)  # Convert to integer
    .alias("laundering_schema_type")
)

def encode_categorical_features(df: pl.DataFrame, categorical_columns: list):
    label_encoders = {}

    for column in categorical_columns:
        logging.info(f"Encoding column: {column}")
        unique_vals = df[column].unique().to_list()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}  # Faster than LabelEncoder
        label_encoders[column] = mapping  # Store the mapping
        df = df.with_columns(df[column].cast(pl.Utf8).replace(mapping).alias(column).cast(pl.Int32))

    return df, label_encoders

train_target, label_encoders = encode_categorical_features(train_target, ["laundering_schema_id"])


# Initialize the array with zeros
y = np.zeros((N_TRANSACTIONS, 3), dtype=np.int16)
# Fill the array with the values from the DataFrame
transaction_ids = train_target['transaction_id_mapped'].to_numpy()
scheme_types = train_target['laundering_schema_type'].to_numpy()
scheme_id = train_target['laundering_schema_id'].to_numpy()


y[transaction_ids, 0] = 1
y[transaction_ids, 1] = scheme_types
y[transaction_ids, 2] = scheme_id

# dump y
np.save(os.path.join(OUTPUT_PATH, "y.npy"), y)
# dump label_encoders
with open(os.path.join(OUTPUT_PATH, "label_encoders_node.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)