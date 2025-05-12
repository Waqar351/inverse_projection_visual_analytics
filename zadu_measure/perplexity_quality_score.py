import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jensenshannon

# Function to compute perplexity-sensitive probability distributions
def compute_perplexity_probabilities(distances, perplexity=30, eps=1e-10):
    sigma = np.mean(np.sort(distances, axis=1)[:, 1:int(perplexity)]) + eps  # Adaptive bandwidth
    similarities = np.exp(-distances ** 2 / (2 * sigma ** 2))  # Gaussian kernel
    np.fill_diagonal(similarities, 0)  # Remove self-similarity
    probabilities = similarities / (np.sum(similarities, axis=1, keepdims=True) + eps)  # Normalize
    
    # Ensure no probability is exactly zero to avoid log(0)
    return np.clip(probabilities, eps, 1)

# Function to compute quality score using Jensen-Shannon Divergence
def perplexity_quality_score(high_dim_data, low_dim_data, perplexity=30):
    n_samples = high_dim_data.shape[0]
    dist_high = pairwise_distances(high_dim_data)
    dist_low = pairwise_distances(low_dim_data)

    P_high = compute_perplexity_probabilities(dist_high, perplexity)
    P_low = compute_perplexity_probabilities(dist_low, perplexity)

    # Compute average JSD between P_high and P_low
    quality_scores = [jensenshannon(P_high[i], P_low[i]) for i in range(n_samples)]
    return {
		"quality_score": np.mean(quality_scores),  # Lower is better,
	}