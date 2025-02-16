import numpy as np
import torch
from utils import data_loader
from clustering import *
import logging
from anomaly_detection import isolation_forest, one_class_svm, autoencoder
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # Load preprocessed data
    logging.info("Loading data...")
    edge_features, edge_index, node_features, y = data_loader.load_all_data()

    # For clustering and anomaly detection, you might work on transaction features.
    # Here we use edge_features as an example.
    X = edge_features.numpy()  # Convert to NumPy array if it's a torch tensor
    y = y[:,0]

    print("=== Clustering Methods ===")
    # Run K-Means clustering
    logging.info("Running K-Means clustering...")
    kmeans_labels = run_kmeans(X, n_clusters=2)
    print("K-Means labels:", np.unique(kmeans_labels))

    # Run DBSCAN clustering
    logging.info("Running DBSCAN clustering...")
    dbscan_labels = run_dbscan(X, eps=1, min_samples=5)
    print("DBSCAN labels (with -1 as outliers):", np.unique(dbscan_labels))

    print("\n=== Anomaly Detection ===")
    # Isolation Forest
    logging.info("Running Isolation Forest...")
    iso_preds, iso_scores = isolation_forest.run_isolation_forest(X, contamination=0.05)
    logging.info(f"Isolation Forest predictions: {iso_preds}")
    logging.info(f"Isolation Forest scores: {iso_scores}")
    print("Isolation Forest predictions (1 normal, -1 anomaly)")


    # One-Class SVM
    svm_preds, svm_scores = one_class_svm.run_one_class_svm(X, nu=0.05)
    logging.info(f"One-Class SVM predictions: {svm_preds}")
    logging.info(f"One-Class SVM scores: {svm_scores}")
    print("One-Class SVM predictions (1 normal, -1 anomaly)")


    # Autoencoder (using a subset if data is large)
    input_dim = X.shape[1]
    print("Training Autoencoder...")
    ae_model, ae_errors = autoencoder.train_autoencoder(X, input_dim, encoding_dim=32, epochs=1, batch_size=128)



if __name__ == "__main__":
    main()
