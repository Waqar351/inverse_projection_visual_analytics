from sklearn import manifold
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
# from pydrmetrics import continuity
# from pyDRMetrics.pyDRMetrics import continuity


def trustworthiness(high_dim_dt, low_dim_dt, n_neighbors, metric = 'euclidean'):
    # Calculate trustworthiness between the original high-dimensional data and the reduced 2D data
    trust = manifold.trustworthiness(high_dim_dt, low_dim_dt, n_neighbors=n_neighbors)
    return trust

# def calculate_continuity_buitin(high_dim_dt, low_dim_dt, n_neighbors):
#     """
#     Calculate the Continuity metric for dimensionality reduction.
    
#     Parameters:
#     - X_high: Original high-dimensional data (n_samples, n_features).
#     - X_low: Reduced low-dimensional data (n_samples, n_components).
#     - k: Number of nearest neighbors to consider.
    
#     Returns:
#     - Continuity score (float).
#     """
#     continuity_score = continuity(high_dim_dt, low_dim_dt, n_neighbors=n_neighbors)
#     return continuity_score


# def calculate_continuity(high_dim_dt, low_dim_dt, n_neighbors=7):
#     """
#     Calculate the Continuity metric for the given high-dimensional data (X)
#     and its low-dimensional projection (Y).
#     """
#     n = high_dim_dt.shape[0]
    
#     # Compute pairwise distances and find k-nearest neighbors in both spaces
#     dist_X = pairwise_distances(high_dim_dt)
#     dist_Y = pairwise_distances(low_dim_dt)
    
#     # Get sorted indices based on distances
#     neighbors_X = np.argsort(dist_X, axis=1)[:, 1:n_neighbors+1]
#     neighbors_Y = np.argsort(dist_Y, axis=1)[:, 1:]
    
#     # Calculate rank of each neighbor in the low-dimensional space
#     ranks_Y = np.argsort(np.argsort(dist_Y, axis=1), axis=1)
    
#     # Compute the continuity penalty
#     penalty = 0
#     for i in range(n):
#         for j in neighbors_X[i]:
#             if j not in neighbors_Y[i, :n_neighbors]:
#                 penalty += max(0, n_neighbors + 1 - ranks_Y[i, j])
    
#     # Normalize the penalty
#     C = 1 - (2 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1))) * penalty
#     return C


import numpy as np
from sklearn.metrics import pairwise_distances

def calculate_continuity(high_dim_dt, low_dim_dt, n_neighbors=7):
    """
    Calculate the Continuity metric for the given high-dimensional data
    and its low-dimensional projection.
    """
    n = high_dim_dt.shape[0]
    
    # Compute pairwise distances and find k-nearest neighbors in both spaces
    dist_X = pairwise_distances(high_dim_dt)
    dist_Y = pairwise_distances(low_dim_dt)
    
    # Get sorted indices based on distances
    neighbors_X = np.argsort(dist_X, axis=1)[:, 1:n_neighbors+1]
    neighbors_Y = np.argsort(dist_Y, axis=1)[:, 1:n_neighbors+1]
    
    # Calculate the continuity penalty
    penalty = 0
    for i in range(n):
        missing_neighbors = set(neighbors_X[i]) - set(neighbors_Y[i])
        for j in missing_neighbors:
            rank_in_Y = np.where(np.argsort(dist_Y[i]) == j)[0][0]
            penalty += max(0, n_neighbors + 1 - rank_in_Y)
    
    # Normalize the penalty
    normalization_factor = n * n_neighbors * (2 * n - 3 * n_neighbors - 1)
    C = 1 - (2 / normalization_factor) * penalty
    return C


def average_local_error(X, Y, k=5):
    """
    Compute the Average Local Error (ALE) between the original space and the projected space.

    Parameters:
    - X: np.ndarray of shape (N, D), the original high-dimensional dataset.
    - Y: np.ndarray of shape (N, d), the projected low-dimensional dataset.
    - k: int, the number of nearest neighbors to consider.

    Returns:
    - ale: float, the Average Local Error.
    """
    # Ensure X and Y have the same number of samples
    assert X.shape[0] == Y.shape[0], "X and Y must have the same number of samples."

    # Number of points
    N = X.shape[0]

    # Compute pairwise distances in both spaces
    dist_orig = euclidean_distances(X)
    dist_proj = euclidean_distances(Y)

    # Find k-nearest neighbors in the original space
    nbrs_orig = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(X)
    neighbors_orig = nbrs_orig.kneighbors(X, return_distance=False)

    # Compute local errors
    local_errors = []
    for i in range(N):
        errors = []
        for j in neighbors_orig[i][1:]:  # Exclude self-distance
            errors.append(abs(dist_orig[i, j] - dist_proj[i, j]))
        local_errors.append(np.mean(errors))

    # Compute and return ALE
    return np.mean(local_errors)
