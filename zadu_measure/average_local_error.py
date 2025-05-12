import numpy as np
from sklearn.metrics import pairwise_distances

def average_local_error(high_dim_data, low_dim_data):
    """
    Compute the Average Local Error (ALE) for each point in the dataset.

    Parameters:
        high_dim_data (numpy.ndarray): High-dimensional data (N x D).
        low_dim_data (numpy.ndarray): Low-dimensional data (N x d).

    Returns:
        numpy.ndarray: ALE for each point (length N).
    """
    n_samples = high_dim_data.shape[0]

    # Compute pairwise distances in high-dimensional and low-dimensional spaces
    dist_high = pairwise_distances(high_dim_data)
    dist_low = pairwise_distances(low_dim_data)

    # Normalize distances
    norm_high = dist_high / np.sum(dist_high, axis=1, keepdims=True)
    norm_low = dist_low / np.sum(dist_low, axis=1, keepdims=True)

    # Calculate ALE for each point
    ale_scores = np.zeros(n_samples)
    for i in range(n_samples):
        ale_scores[i] = np.mean(np.abs(norm_high[i] - norm_low[i]))

    ale_scores = np.mean(ale_scores)
    return {
		"average_local_error": ale_scores
	    }