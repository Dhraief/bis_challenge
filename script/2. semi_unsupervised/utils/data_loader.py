import torch
import os
import numpy  as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_PATH = "/dccstor/aml_datasets/bis/"

(edge_feature_path, edge_index_path, node_feature_path, y_path) = (os.path.join(INPUT_PATH, "edge_features.pt"),
    os.path.join(INPUT_PATH, "edges_tensor.pt"),
    os.path.join(INPUT_PATH, "node_feature.pt"),
    os.path.join(INPUT_PATH, "y.npy"))

def load_all_data(edge_feature_path=edge_feature_path,
                 edge_index_path=edge_index_path,
                 node_feature_path=node_feature_path,
                 y_path=y_path,N_TEST=None):
    """Loads edge features, edge index, node features, and labels."""
    # load data
    logging.info("Loading edge features...")
    edge_features = torch.load(edge_feature_path)  # Shape: [N_edges, feature_dim]
    logging.info("Loading edge index...")
    edge_index = torch.load(edge_index_path).T    # Shape: [2, N_edges]
    logging.info("Loading node features...")
    node_features = torch.load(node_feature_path) # Shape: [N_nodes, feature_dim]
    logging.info("Loading labels...")
    y = np.load(y_path)      

    if N_TEST is not None:
        y = y[:N_TEST]
        edge_features = edge_features[:N_TEST]
        edge_index = edge_index[:,:N_TEST]

    return edge_features, edge_index, node_features, y