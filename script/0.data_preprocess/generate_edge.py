
import polars as pl
import torch
import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pickle
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
TRANSACTION_DATASET_PATH = "/dccstor/aml_datasets/bis/data/raw/transaction_dataset.parquet"
OUTPUT_PATH = "/dccstor/aml_datasets/bis"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load datasets
logging.info("Loading datasets...")

transaction_dataset = pl.read_parquet(TRANSACTION_DATASET_PATH)#.head(100000)
# drop duplicates in transaction_id
print("before",transaction_dataset.shape)
transaction_dataset = transaction_dataset.unique(subset=["transaction_id"])
print("after",transaction_dataset.shape)

logging.info("Loaded transaction dataset.")
# Create mapping for transaction_id
logging.info("Creating mapping for transaction_id...")

transaction_ids = transaction_dataset["transaction_id"].unique()  # Get unique IDs
logging.info(f"Unique transaction IDs: {len(transaction_ids)}")
mapping_transaction = dict(zip(transaction_ids, range(len(transaction_ids))))  # Faster mapping

# save mapping_transaction
with open(os.path.join(OUTPUT_PATH, "mapping_transaction.pkl"), "wb") as f:
    pickle.dump(mapping_transaction, f)

def encode_categorical_features(df: pl.DataFrame, categorical_columns: list):
    label_encoders = {}

    for column in categorical_columns:
        logging.info(f"Encoding column: {column}")
        unique_vals = df[column].unique().to_list()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}  # Faster than LabelEncoder
        label_encoders[column] = mapping  # Store the mapping
        df = df.with_columns(df[column].cast(pl.Utf8).replace(mapping).alias(column).cast(pl.Int32))

    return df, label_encoders

# Define categorical columns
categorical_columns = [
    "weekday", "channel", "payment_system",
    "category_0", "category_1", "category_2", "amount"
]

logging.info("Encoding categorical features...")
transaction_dataset, label_encoders = encode_categorical_features(transaction_dataset, categorical_columns)


# Function to convert time to seconds of the day
def convert_to_seconds_of_day(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (df["hour"] * 3600 + df["min"] * 60 + df["sec"]).alias("seconds_of_day")
    ).drop(["hour", "min", "sec"])

logging.info("Converting time to seconds of the day...")
transaction_dataset = convert_to_seconds_of_day(transaction_dataset)

# Function to convert date to days of the year
def convert_to_days_of_year(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (df["day"] + (df["month"] - 1) * 30).alias("days_of_year")
    ).drop(["day", "month"])

logging.info("Converting date to days of the year...")
transaction_dataset = convert_to_days_of_year(transaction_dataset)

# Function to encode categorical features
# def encode_categorical_features(df: pl.DataFrame, categorical_columns: list):
#     label_encoders = {}
#     df_dict = df.to_dict(as_series=False)
    
#     for column in categorical_columns:
#         le = LabelEncoder()
#         df_dict[column] = le.fit_transform(df_dict[column])
#         label_encoders[column] = le
    
#     return pl.DataFrame(df_dict), label_encoders


# Round and convert 'amount' to integer
logging.info("Rounding and converting amount column...")
transaction_dataset = transaction_dataset.with_columns(
    transaction_dataset["amount"].round().cast(pl.Int32)
)

# Map account_id and counterparty_id to unique integers
logging.info("Creating ID mappings...")
account_ids = transaction_dataset["account_id"].unique()
counterparty_ids = transaction_dataset["counterpart_id"].unique()
all_ids = np.concatenate((account_ids.to_numpy(), counterparty_ids.to_numpy()))
all_ids = np.unique(all_ids)
id_mapping = {id_: i for i, id_ in enumerate(all_ids)}
# save id_mapping 
with open(os.path.join(OUTPUT_PATH, "id_mapping.pkl"), "wb") as f:
    pickle.dump(id_mapping, f)


# Apply ID mapping using `.replace()`
transaction_dataset = transaction_dataset.with_columns([
    transaction_dataset["account_id"].replace(id_mapping).alias("account_id_mapped"),
    transaction_dataset["counterpart_id"].replace(id_mapping).alias("counterpart_id_mapped")
])

# Generate edge list efficiently

logging.info("Generating edges...")

# Efficient vectorized edge generation in Polars
transaction_dataset = transaction_dataset.with_columns([
    pl.when(transaction_dataset["transaction_direction"] == "inbound")
      .then(transaction_dataset["counterpart_id_mapped"])
      .otherwise(transaction_dataset["account_id_mapped"])
      .alias("edge_source"),

    pl.when(transaction_dataset["transaction_direction"] == "inbound")
      .then(transaction_dataset["account_id_mapped"])
      .otherwise(transaction_dataset["counterpart_id_mapped"])
      .alias("edge_target")
])

# Convert edge list to tensor
logging.info("Converting edges to tensor...")
edges = transaction_dataset.select(["edge_source", "edge_target"]).to_numpy()
edges = np.array(edges, dtype=np.int32)
edges_tensor = torch.tensor(edges, dtype=torch.int32)
# save
edges_tensor_path = os.path.join(OUTPUT_PATH, "edges_tensor.pt")
torch.save(edges_tensor, edges_tensor_path)

# Create edge feature tensor
logging.info("Generating edge feature tensor...")
cols_to_ignore = [
    "transaction_id", "account_id", "transaction_direction",
    "counterpart_id", "counterpart_id_mapped", "account_id_mapped", "edge","__index_level_0__",
     'edge_source', 'edge_target'
]
cols_to_keep = [col for col in transaction_dataset.columns if col not in cols_to_ignore]
print(cols_to_keep)

edge_feature = transaction_dataset.select(cols_to_keep).to_numpy()
edge_feature = np.array(edge_feature, dtype=np.int32)
edge_features = torch.tensor(edge_feature, dtype=torch.int32)
edge_features_path = os.path.join(OUTPUT_PATH, "edge_features.pt")
torch.save(edge_features, edge_features_path)

# Save label encoders
logging.info("Saving label encoders...")

with open(os.path.join(OUTPUT_PATH, "label_encoders.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)

logging.info(f"Edge feature tensor shape: {edge_features.shape}")
logging.info(f"Edges tensor shape: {edges_tensor.shape}")

