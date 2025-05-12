import numpy as np
from sklearn.metrics import pairwise_distances

def projection_precision_score_v1(O, P, n_neighbors):
    """
    Compute the Projection Precision Score (PPS) for a dataset.
    
    Parameters:
        O (numpy.ndarray): Original high-dimensional data (shape: N x D1).
        P (numpy.ndarray): Low-dimensional projection (shape: N x D2).
        n_neighbors (int): Number of nearest neighbors to consider (1 < n < N).
    
    Returns:
        float: Average Projection Precision Score (PPS) over all points.
    """
    N = O.shape[0]  # Number of data points
    
    # Compute pairwise distances in the high-dimensional and low-dimensional spaces
    distances_O = pairwise_distances(O)  # High-dimensional distances
    distances_P = pairwise_distances(P)  # Low-dimensional distances

    # Initialize a list to store PPS for each point
    pps_scores = []

    for i in range(N):
        # Get the indices of the n nearest neighbors in the high-dimensional space (excluding the point itself)
        nearest_neighbors_indices = np.argsort(distances_O[i])[:n_neighbors+1][1:]

        # Compute distance vectors for high-dimensional and low-dimensional spaces
        d_O = distances_O[i, nearest_neighbors_indices]
        d_P = distances_P[i, nearest_neighbors_indices]

        # Normalize the distance vectors to unit length
        d_O_normalized = d_O / np.linalg.norm(d_O)
        d_P_normalized = d_P / np.linalg.norm(d_P)

        # Compute the PPS for the current point
        pps = np.linalg.norm(d_O_normalized - d_P_normalized)
        pps_scores.append(pps)

    # Return the average PPS across all points
    return {
        'projection_precision_score': np.mean(pps_scores)
    }
