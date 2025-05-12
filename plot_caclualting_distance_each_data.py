import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.model_selection import train_test_split
import argparse
from inver_project_model import model_train, model_test
from datasets import *
from projections_methods import get_reducer
from temp_projection_quality_metrics import trustworthiness, calculate_continuity, average_local_error
# from plots import trustworthiness_plot, continuity_plot, average_local_error_plot
from plots import *
from projection_metrics import calculate_projection_metrics, ProjectionMetrics
from utility import *
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
from zadu_measure.local_continuity_meta_criteria import measure_lcmc
from zadu_measure.neighbor_dissimilarity import measure_nd
from zadu_measure.neighborhood_preservation_precision import neighborhood_preservation_precision
from zadu_measure.perplexity_quality_score import perplexity_quality_score
from scipy.ndimage import sobel
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler


cluster_spacing = 1.0

bTetrahedron = True
spread_factor = 0.01
# spread_factor = 0.10
distance_factor= 1.0    #  0.5, far 3.0 more_far =6.0
distance_factor_2 = 1.0  
move_cluster_index=0 
num_grid_points = 200  # 200
bNormFlag = False
norm_type = 'global'

datasets = ['equidistant', 'far', 'close', '2_close']

all_distance_matrices = []
all_distance_matrices_mean = []

for dataset in datasets:

    if dataset == 'equidistant':
        distance_factor= 1.0    #  0.5, far 3.0 more_far =6.0
        distance_factor_2 = 1.0
        fig_name_last=  f'_df_{distance_factor}_df_2_{distance_factor_2}'
    if dataset == 'far':
        distance_factor= 3.0    #  0.5, far 3.0 more_far =6.0
        distance_factor_2 = 1.0
        fig_name_last=  f'_df_{distance_factor}_df_2_{distance_factor_2}'
    if dataset == 'close':
        distance_factor= 0.0    #  0.5, far 3.0 more_far =6.0
        distance_factor_2 = 1.0
        fig_name_last=  f'_df_{distance_factor}_df_2_{distance_factor_2}'
    if dataset == '2_close':
        distance_factor= 0.5    #  0.5, far 3.0 more_far =6.0
        distance_factor_2 = 0.5
        fig_name_last=  f'_df_{distance_factor}_df_2_{distance_factor_2}'

    tetra =  'tetrahedron_eq'
    output_folder = f"thesis_reproduced/testing_new/perplexity_analysis/HD_distance_plots"
    
    os.makedirs(output_folder, exist_ok=True) 

    centers, overlap_factor = cluster_position(cluster_spacing, mode = tetra)
    D, c, centers = generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=200, base_tetrahedron = centers, spread_factor= spread_factor, distance_factor=distance_factor, distance_factor_2= distance_factor_2, move_cluster_index=move_cluster_index)

    orig_label = c
    orig_label = np.array(orig_label)

    # Unique class labels
    unique_labels = np.sort(np.unique(orig_label))
    ###########______________Plot gaussian______________________________________________

    colors = ['#FF0000', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF', '#0000FF', '#000000']

    distance_matrix_hd, mean_cluster_distance__hd = inter_intra_cluster_pairwise_distance(D, orig_label, unique_labels, metric = 'euclidean', norm_distance = bNormFlag, norm_type=norm_type)

    all_distance_matrices.append(distance_matrix_hd)
    all_distance_matrices_mean.append(mean_cluster_distance__hd)


# Convert list to a single NumPy array
all_distance_matrices = np.array(all_distance_matrices, dtype=object)  # Using dtype=object for variable-sized matrices
all_distance_matrices_mean = np.array(all_distance_matrices_mean, dtype=object)  # Using dtype=object for variable-sized matrices

# Compute global min and max
global_min = min(np.min(matrix) for matrix in all_distance_matrices)
global_max = max(np.max(matrix) for matrix in all_distance_matrices)
# Compute global min and max
global_min_mean = min(np.min(matrix) for matrix in all_distance_matrices_mean)
global_max_mean = max(np.max(matrix) for matrix in all_distance_matrices_mean)

# Normalize each distance matrix using global min-max normalization
normalized_matrices = [(matrix - global_min) / (global_max - global_min) for matrix in all_distance_matrices]
normalized_mean = [(matrix - global_min_mean) / (global_max_mean - global_min_mean) for matrix in all_distance_matrices_mean]



# breakpoint()
for i, dataset in enumerate(datasets):
    dist_mat = normalized_matrices[i]
    dist_mat_mean = normalized_mean[i]
    dist_mat = np.array(dist_mat, dtype=float)
    dist_mat_mean = np.array(dist_mat_mean, dtype=float)

    output_path = os.path.join(output_folder, f"HD_clust_distance_{dataset}")
    fig_title = 'Distances across clusters in high dimension'
    plot_pairwise_cluster_distance_v2(dist_mat, dist_mat_mean, orig_label, unique_labels, colors, fig_title, output_path=output_path)


