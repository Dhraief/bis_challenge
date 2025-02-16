
import polars as pl
import torch
import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pickle
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import numpy as np
import torch
from sklearn.model_selection import train_test_split


INPUT_PATH = "/dccstor/aml_datasets/bis/"
N_TEST = 1000000

if N_TEST is not None:
    OUTPUT_PATH = "/dccstor/aml_datasets/bis_{}k/".format(str(N_TEST))
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    logging.info("Output path set to: {}".format(OUTPUT_PATH))

else: 
    OUTPUT_PATH = INPUT_PATH

def load_data(edge_feature_path, edge_index_path, node_feature_path, y_path):
    """Loads edge features, edge index, node features, and labels."""
    # load data
    logging.info("Loading edge features...")
    edge_features = torch.load(edge_feature_path)  # Shape: [N_edges, feature_dim]
    logging.info("Loading edge index...")
    edge_index = torch.load(edge_index_path).T    # Shape: [2, N_edges]
    logging.info("Loading node features...")
    node_features = torch.load(node_feature_path) # Shape: [N_nodes, feature_dim]
    logging.info("Loading labels...")
    y = np.load(y_path)        # Shape: [N_edges, num_classes] (or N_edges, 1 for binary)
    if N_TEST is not None:
        y = y[:N_TEST]
        edge_features = edge_features[:N_TEST]
        edge_index = edge_index[:,:N_TEST]
    return edge_features, edge_index, node_features, y

def enhance_edge_features(edge_features, edge_index, node_features):
    """Enhances edge features by adding source & target node features."""
    source_nodes = edge_index[0]  # Extract source node indices
    target_nodes = edge_index[1]  # Extract target node indices
    

    # Pull source and target node features
    logging.info("Extracting source and target node features...")
    source_features = node_features[source_nodes]  # Shape: [N_edges, node_feature_dim]
    target_features = node_features[target_nodes]  # Shape: [N_edges, node_feature_dim]

    # Concatenate source, edge, and target features
    logging.info("Concatenating features...")
    enhanced_edge_features = torch.cat([source_features, edge_features, target_features], dim=1)
    
    return enhanced_edge_features

def preprocess_data(edge_feature_path, edge_index_path, node_feature_path, y_path):
    """Loads, processes, and returns X and y for training."""
    # Load raw data
    logging.info("Loading data...")
    edge_features, edge_index, node_features, y = load_data(edge_feature_path, edge_index_path, node_feature_path, y_path)

    # Enhance edge features
    logging.info("Enhancing edge features...")
    X = enhance_edge_features(edge_features, edge_index, node_features)

    return X, y

def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Splits data into train and test sets while:
    1. Ensuring similar proportions of illicit transactions in both sets.
    2. Keeping all rows with the same scheme_id in both train and test.

    Parameters:
    - X (np.array or pd.DataFrame): Features
    - y (np.array or pd.DataFrame): Labels (where column index 0 is illicit label and column index 2 is scheme_id)
    - test_size (float): Proportion of data to use as test set
    - random_state (int): Random seed for reproducibility

    Returns:
    - X_train, X_test, y_train, y_test (arrays or DataFrames)
    """

    # Extract labels and scheme IDs
    illicit_labels = y[:, 0]  # Assuming illicit label is in the first column
    scheme_ids = y[:, 2]  # Assuming scheme_id is in the third column

    # Create a mask for illicit transactions (where scheme_id exists)
    illicit_mask = illicit_labels == 1  # Assuming -1 or NaN means no scheme_id

    # Get illicit and non-illicit indices
    illicit_indices = np.where(illicit_mask)[0]
    non_illicit_indices = np.where(~illicit_mask)[0]

    # Split non-illicit transactions normally
    non_illicit_train, non_illicit_test = train_test_split(
        non_illicit_indices, 
        test_size=test_size, random_state=random_state
    )

    # get all of the scheme_ids for illicit labels, split them on train and test
    scheme_ids_illicit = scheme_ids[illicit_indices]

    unique_scheme_ids = np.unique(scheme_ids_illicit)
    train_scheme_ids, test_scheme_ids = train_test_split(
        unique_scheme_ids, test_size=test_size, random_state=random_state
    )
    # Get indices for illicit transactions based on scheme_ids
    illicit_train = illicit_indices[np.isin(scheme_ids_illicit, train_scheme_ids)]
    illicit_test = illicit_indices[np.isin(scheme_ids_illicit, test_scheme_ids)]

    # Combine train and test sets
    train_indices = np.concatenate([non_illicit_train, illicit_train])
    test_indices = np.concatenate([non_illicit_test, illicit_test])

    # Shuffle the indices to ensure randomness
    np.random.seed(random_state)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Subset the data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    print(f"Train X Shape: {X_train.shape}, Train y Shape: {y_train.shape}")
    print(f"Test X Shape: {X_test.shape}, Test y Shape: {y_test.shape}")

    print(f"Train Illicit Count: {np.sum(y_train[:, 0] == 1)}, Test Illicit Count: {np.sum(y_test[:, 0] == 1)}")
    print(f"Train Scheme IDs: {len(np.unique(y_train[:, 2]))}, Test Scheme IDs: {len(np.unique(y_test[:, 2]))}")
    return X_train, X_test, y_train, y_test

# Example usage:
# X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2)


# Example Usage
if __name__ == "__main__":
    edge_feature_path = INPUT_PATH+"edge_features.pt"
    edge_index_path = INPUT_PATH+"edges_tensor.pt"
    node_feature_path = INPUT_PATH+"node_feature.pt"
    y_path = INPUT_PATH+"y.npy"

    logging.info("Loading data...")
    X, y = preprocess_data(edge_feature_path, edge_index_path, node_feature_path, y_path)
    print(f"Processed X Shape: {X.shape}, Processed y Shape: {y.shape}")
    logging.info("Data loaded and processed.")
    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2)
    logging.info("Data split completed.")

    # dump X_train, X_test, y_train, y_test
    torch.save(X_train, OUTPUT_PATH+"X_train_{}.pt".format(N_TEST if N_TEST is not None else ""))
    torch.save(X_test, OUTPUT_PATH+"X_test_{}.pt".format(N_TEST if N_TEST is not None else ""))
    torch.save(y_train, OUTPUT_PATH+"y_train_{}.pt".format(N_TEST if N_TEST is not None else ""))
    torch.save(y_test, OUTPUT_PATH+"y_test_{}.pt".format(N_TEST if N_TEST is not None else ""))
    logging.info("Data saved to disk.")