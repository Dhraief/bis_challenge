import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
INPUT_PATH = "/dccstor/aml_datasets/bis/"

(edge_feature_path, edge_index_path, node_feature_path, y_path) = (os.path.join(INPUT_PATH, "edge_features.pt"),
    os.path.join(INPUT_PATH, "edges_tensor.pt"),
    os.path.join(INPUT_PATH, "node_feature.pt"),
    os.path.join(INPUT_PATH, "y.npy"))


class BISDataset(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        
        # Load preprocessed data
        self.load_data()
        
    def load_data(self):
        """Loads preprocessed edge features, node features, edge indices, and labels."""
        logging.info("Loading edge features...")
        self.edge_features = torch.load(edge_feature_path).long()  # Shape: [N_edges, feature_dim]
        logging.info("Loading edge index...")
        self.edge_index = torch.load(edge_index_path).T.long()    # Shape: [2, N_edges]
        logging.info("Loading node features...")
        self.node_features = torch.load(node_feature_path).float() # Shape: [N_nodes, feature_dim]
        logging.info("Loading labels...")
        self.y = torch.tensor(np.load(y_path),dtype= torch.long)

        self._data = Data(
            x=self.node_features,       # Node features
            edge_index=self.edge_index,  # Edge indices
            edge_attr=self.edge_features,  # Edge features
            y=self.y  # Edge labels
        )
        logging.info("Data loaded successfully.")
        logging.info("Shapes are:")
        logging.info(f"Node features: {self.node_features.shape}")
        logging.info(f"Edge features: {self.edge_features.shape}")
        logging.info(f"Edge index: {self.edge_index.shape}")
        logging.info(f"Labels: {self.y.shape}")
        
        self.slices = None  # Not required for a single graph dataset
    
    @property
    def processed_file_names(self):
        """Define expected preprocessed file names."""
        return ['node_feature.pt', 'edge_feature.pt', 'edge_index.pt', 'y.pt']
    
    def process(self):
        """No processing needed since we load preprocessed data."""
        pass

    def __getitem__(self, idx):
        """Return the dataset (only one graph in this case)."""
        return self.data
    
    def __len__(self):
        """Returns the number of graphs (only 1 here)."""
        return 1
