import numpy as np
from sklearn.cluster import KMeans, DBSCAN

def run_kmeans(data, n_clusters=2,max_iter=1):
    """
    Run K-Means clustering on data.
    :param data: NumPy array of shape (n_samples, n_features)
    :param n_clusters: Number of clusters
    :return: cluster labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,max_iter=max_iter)
    labels = kmeans.fit_predict(data)
    return labels

def run_dbscan(data, eps=1, min_samples=2):
    """
    Run DBSCAN clustering on data.
    :param data: NumPy array of shape (n_samples, n_features)
    :param eps: Maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    :return: cluster labels (-1 for outliers)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples,max_iter=1)
    labels = dbscan.fit_predict(data)
    return labels

