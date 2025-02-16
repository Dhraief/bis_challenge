import polars as pl
import torch
import logging
import numpy as np
import os
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
ACCOUNT_DATASET_PATH = "/dccstor/aml_datasets/bis/data/raw/account_dataset.parquet"
OUTPUT_PATH = "/dccstor/aml_datasets/bis"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load dataset
logging.info("Loading account dataset...")
account_dataset = pl.read_parquet(ACCOUNT_DATASET_PATH)

# Function to encode categorical features
def encode_categorical_features(df: pl.DataFrame, categorical_columns: list):
    label_encoders = {}
    for column in categorical_columns:
        logging.info(f"Encoding column: {column}")
        unique_vals = df[column].unique().to_list()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        label_encoders[column] = mapping
        df = df.with_columns(df[column].cast(pl.Utf8).replace(mapping).alias(column).cast(pl.Int32))
    return df, label_encoders

# Encode categorical columns
categorical_columns = ["assigned_bank_type", "assigned_bank"]
logging.info("Encoding categorical features...")
account_dataset, label_encoders = encode_categorical_features(account_dataset, categorical_columns)

# Save label encoders
with open(os.path.join(OUTPUT_PATH, "label_encoders_node.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)

# Round initial_balance and convert age to int
account_dataset = account_dataset.with_columns(
    pl.col("initial_balance").round(0).cast(pl.Int32),
    pl.col("age").cast(pl.Float32).round(0).cast(pl.Int32)
)

# Load ID mapping
logging.info("Loading ID mapping...")
with open(os.path.join(OUTPUT_PATH, "id_mapping.pkl"), "rb") as f:
    id_mapping = pickle.load(f)

account_dataset = account_dataset.with_columns(
    pl.col("account_id").replace(id_mapping).alias("account_id_mapped").cast(pl.Int32),
    pl.col("account_id").cast(pl.Int32)
)

# Include missing accounts in dataset
logging.info("Ensuring all accounts are included...")
all_accounts_df = pl.DataFrame({"account_id_mapped": list(id_mapping.values())})
all_accounts_df = all_accounts_df.join(account_dataset, on="account_id_mapped", how="left")

# Fill missing values and sort
all_accounts_df = all_accounts_df.fill_null(-1).sort("account_id_mapped")

# Select required columns
cols = ["age", "initial_balance", "assigned_bank_type", "assigned_bank"]
all_accounts_df = all_accounts_df.select(cols).with_columns(
    pl.col("assigned_bank_type").cast(pl.Int32),
    pl.col("assigned_bank").cast(pl.Int32)
)

# Convert to tensor and save
logging.info("Converting to tensor and saving...")
node_feature = torch.tensor(np.array(all_accounts_df.to_numpy(), dtype=np.int32), dtype=torch.int32)
# show shape of node_feature
logging.info(f"Node feature shape: {node_feature.shape}")
torch.save(node_feature, os.path.join(OUTPUT_PATH, "node_feature.pt"))

logging.info("Processing complete! ✅")
