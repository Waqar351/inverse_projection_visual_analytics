import numpy as np
from sklearn.metrics import pairwise_distances

def neighborhood_preservation_precision(high_dim_data, low_dim_data, n_neighbors=10):
    """
    Comparing the overlap of the nearest neighbors in both high- and low-dimensional spaces.

    Parameters:
        high_dim_data (numpy.ndarray): High-dimensional data.
        low_dim_data (numpy.ndarray): Low-dimensional embedding.
        n_neighbors (int): Number of neighbors to consider.

    Returns:
        float: "Neighborhood Preservation Precision (NPP).
    """
    n_samples = high_dim_data.shape[0]

    # Compute pairwise distances
    dist_high = pairwise_distances(high_dim_data)
    dist_low = pairwise_distances(low_dim_data)

    # Find the k-nearest neighbors
    neighbors_high = np.argsort(dist_high, axis=1)[:, 1:n_neighbors+1]
    neighbors_low = np.argsort(dist_low, axis=1)[:, 1:n_neighbors+1]

    # Compute the Neighborhood Preservation Precision for each point
    quality_score = []
    for i in range(n_samples):
        overlap = len(set(neighbors_high[i]).intersection(set(neighbors_low[i])))
        quality_score.append(overlap / n_neighbors)

    # Return the average precision score
    return {
        'new_metric': np.mean(quality_score)
    }
