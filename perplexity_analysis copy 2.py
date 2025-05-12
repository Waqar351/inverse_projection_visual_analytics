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
from scipy.spatial import Delaunay
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
import itertools
from matplotlib.collections import LineCollection
import itertools

# Argument Parser
parser = argparse.ArgumentParser(description="Select dimensionality reduction technique: t-SNE or UMAP.")
parser.add_argument(
    "--method",
    type=str,
    choices=["tsne", "umap", "pca"],
    required=True,
    help="Choose 'tsne' or 'umap' or 'pca' for dimensionality reduction."
)
parser.add_argument(
    "--dataset",
    type=str,
    # choices=["iris", "gaussian", "digits"],
    required=True,
    help="Choose dataset for dimensionality reduction."
)
parser.add_argument(
    "--num_dim",
    type=int,
    # choices=["iris", "gaussian", "digits"],
    # required=True,
    help="Choose dataset for dimensionality reduction."
)
args = parser.parse_args()

# Customize Parameters
# n_gauss = 4 
n_pts_per_gauss = 500  
cluster_spacing = 1.0

spread_factor = 0.01
# spread_factor = 0.10
distance_factor= 0.5    #  0.5, far 3.0 more_far =6.0
distance_factor_2 = 0.5  
move_cluster_index=0 
num_grid_points = 200  # 200
bNormFlag = True
norm_type = 'global'

# perplexities = [2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
# perplexities = [2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, ] #40, 45, 50, 55, 60, 70, 80, 90, 100]
# perplexities = [5, 10, 15, 20, 25] 
perplexities = [5] 
n_neihbors_metrics = [5]

##########______ NN model ___________________________________________________
input_size = 2
batch_size = 64
num_epochs = 1000
# output_size = dim = 3
np.random.seed(5)
###______________________________________________________

if args.dataset == "gaussian":
    centers, overlap_factor = cluster_position(cluster_spacing, mode = args.dataset)
    dim = 3
    output_size = dim
    n_gauss = 6
    D, c, centers = gaussian_dt(n_gauss, n_pts_per_gauss, dim)
elif args.dataset == "tetrahedron_eq":
    centers, overlap_factor = cluster_position(cluster_spacing, mode = args.dataset)
    D, c, centers = generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=200, base_tetrahedron = centers, spread_factor= spread_factor, distance_factor=distance_factor, distance_factor_2= distance_factor_2, move_cluster_index=move_cluster_index)
    dim = D.shape[1]
    output_size = dim
    n_gauss = len(np.unique(c))


elif args.dataset == "iris":
    dim = 4
    output_size = dim
    n_gauss = 3  # number of classes
    D, c = iris_dt()

elif args.dataset == 'digits':

    D, c = digits_dt()
    dim = D.shape[1]
    output_size = dim
    n_gauss = len(np.unique(c))
elif args.dataset == 'har':

    D, c = har_dt()
    dim = D.shape[1]
    output_size = dim
    n_gauss = len(np.unique(c))
elif args.dataset == 'covariance':

    D, c = covariance_type()
    dim = D.shape[1]
    output_size = dim
    n_gauss = len(np.unique(c))
elif args.dataset == 'wine':

    D, c = wine_dt()
    dim = D.shape[1]
    output_size = dim
    n_gauss = len(np.unique(c))
elif args.dataset == 'breast':

    D, c = breast_cancer_dt()
    dim = D.shape[1]
    output_size = dim
    n_gauss = len(np.unique(c))
elif args.dataset == 'cifar':

    D, c = cifar_10()
    dim = D.shape[1]
    output_size = dim
    n_gauss = len(np.unique(c))
elif args.dataset == 'high_dim':
    D, c, centers = generate_high_dimension_gaussians(num_dim = args.num_dim, n_pts_per_gauss=200, spread_factor= spread_factor, distance_factor=distance_factor, distance_factor_2= distance_factor_2, move_cluster_index=move_cluster_index, random_seed = 5)
    dim = D.shape[1]
    output_size = dim
    n_gauss = len(np.unique(c))

else:
    raise ValueError("Invalid dataset name. Choose 'iris' or 'gaussian'.")

# breakpoint()
###
# Specify the output folder path
# dataset = 'tetrahedron_eq'    #dron_1_far_1_cl','tetrahedron_eq', 'tetrahedron_close', 'tetrahedron_far', 'tetrahedron_more_far', 1_close_pairs_1_pair_far, 'cluster_1_far_other_close, 'equidistant', '2_close_pairs', '2_close_pairs_far','2_10_points_far', 'non_symmetric','irregular', 'sparse'
dataset = args.dataset    #dron_1_far_1_cl','tetrahedron_eq', 'tetrahedron_close', 'tetrahedron_far', 'tetrahedron_more_far', 1_close_pairs_1_pair_far, 'cluster_1_far_other_close, 'equidistant', '2_close_pairs', '2_close_pairs_far','2_10_points_far', 'non_symmetric','irregular', 'sparse'
method = args.method
# breakpoint()
if dataset == 'tetrahedron_eq':
    output_folder = f"thesis_reproduced/testing_new/perplexity_analysis/{dataset}_cl_spac_{cluster_spacing}_spred_{spread_factor}_df_{distance_factor}_df_2_{distance_factor_2}/{method}_plots_new_model"
    fig_name_last=  f'_df_{distance_factor}_df_2_{distance_factor_2}'
elif dataset == 'gaussian':
    output_folder = f"thesis_reproduced/testing_new/perplexity_analysis/{dataset}_cl_spac_{cluster_spacing}/{method}_plots_new_model"
    fig_name_last = f'_cl_spac_{cluster_spacing}'
elif dataset == 'high_dim':
    # breakpoint()
    output_folder = f"thesis_reproduced/testing_new/perplexity_analysis/{dataset}_{dim}/{method}_plots_new_model"
    fig_name_last = f'_high_dim_{dim}'
    
else:
    output_folder = f"thesis_reproduced/testing_new/perplexity_analysis/{dataset}/{method}_plots_new_model"
    fig_name_last = f'{dataset}'

# breakpoint()   
os.makedirs(output_folder, exist_ok=True) 

# Generate Data
## Dynamic gaussian creation
# D, c, centers = gaussian_dt(n_gauss, n_pts_per_gauss, dim)

# # ##Manual gaussian creation
# centers, overlap_factor = cluster_position(cluster_spacing, mode = dataset)

# D, c, centers = generate_gaussian_clusters(n_gauss, n_pts_per_gauss, centers= centers, cluster_spacing=cluster_spacing, overlap_factors=overlap_factor)
# D, c, centers = generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=200, base_tetrahedron = centers, spread_factor= spread_factor, distance_factor=distance_factor, distance_factor_2= distance_factor_2, move_cluster_index=move_cluster_index)

# # ###__________________________ HAR dataset________________________
# D, c = har_dt()
# dim = D.shape[1]
# output_size= dim
# n_gauss = len(np.unique(c))
# # # ###______________________________________________________________________

orig_label = c
orig_label = np.array(orig_label)

# Unique class labels
unique_labels = np.sort(np.unique(orig_label))
###########______________Plot gaussian______________________________________________

# colors = ['#FF0000', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF', '#0000FF', '#000000']
colors = ['#FF0000', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF', '#0000FF', '#000000', 
          '#FFA500', '#8000FF', '#FF1493']

# colors = ['#FFEB3B', '#1E88E5', '#D32F2F', '#8E24AA']

# output_path = os.path.join(output_folder, f'original_hd_data_{dataset}{fig_name_last}.html')
# plot_3D_gaussian(D, c,n_gauss, colors, dataset, output_path)

# breakpoint()
##########################################################################################################

######___________mean_inter-intra Cluster distance of High dimensional (original) data_______________________________________

# mean_distance_matrix_hd = inter_intra_cluster_mean_distance(D, orig_label, unique_labels)
# output_path = os.path.join(output_folder, f"high_dimension_cluster_mean_distance")
# plot_mean_cluster_distance(mean_distance_matrix_hd,centers, unique_labels, output_path=output_path)
# breakpoint()
###__________pairwise_inter_intra_cluster distance of High dimensional (original) data______________________________________
distance_matrix_hd, mean_cluster_distance__hd = inter_intra_cluster_pairwise_distance(D, orig_label, unique_labels, metric = 'euclidean', norm_distance = bNormFlag, norm_type=norm_type)
# distance_matrix_hd_new, mean_cluster_distance__hd_new = inter_intra_cluster_pairwise_distance_fixed(D, orig_label, unique_labels, metric = 'euclidean', norm_distance = bNormFlag, norm_type=norm_type)
# breakpoint()
# output_path = os.path.join(output_folder, f"HD_clust_distance_{dataset}{fig_name_last}")
fig_title = 'Distances across clusters in high dimension'
# plot_pairwise_cluster_distance(distance_matrix_hd, mean_cluster_distance__hd, orig_label, unique_labels, colors, fig_title, output_path=output_path)

output_path = os.path.join(output_folder, f"HD_clust_distance_Version_2{dataset}{fig_name_last}")

plot_pairwise_cluster_distance_v2(distance_matrix_hd, mean_cluster_distance__hd, orig_label, unique_labels, colors, fig_title, output_path=output_path)

# output_path = os.path.join(output_folder, f"HD_clust_mean_distance{dataset}{fig_name_last}")
# plot_mean_cluster_distance(mean_cluster_distance__hd, unique_labels, colors, output_path=output_path)

# output_path = os.path.join(output_folder, f"HD_NEW_clust_distance_{dataset}")
# plot_pairwise_cluster_distance_v2(distance_matrix_hd_new, mean_cluster_distance__hd_new, orig_label, unique_labels, colors, output_path=output_path)

# output_path = os.path.join(output_folder, f"HD_NEW_clust_mean_distance{dataset}")
# plot_mean_cluster_distance(mean_cluster_distance__hd_new, unique_labels, colors, output_path=output_path)
# breakpoint()



########################################################


prj_metrics_hd_to_ld = ProjectionMetrics()
prj_metrics_hd_to_hd = ProjectionMetrics()
prj_inverse_proj_diff = []
jacobian_norm_results = {}
# projection_metrics_results = {}
# projection_metrics_resutls_hd_hd = {}
low_dm_emb_per_perplexity = {}
mantel_results = {}

print('Start perplexity loop ...')

for i, perplexity in enumerate(perplexities):

    reducer, method_name, title_var = get_reducer(method, perplexity)

    

    low_dm_emb = reducer.fit_transform(D)
    low_dm_emb_per_perplexity[perplexity] = low_dm_emb

    ####_________ Ground truth_______________________________________________________________
    output_path = os.path.join(output_folder, f"Fully_connected_clusters_point_edges_{method_name}_{perplexity}_{dataset}_{fig_name_last}")

    # plot_fully_connected_points_edges(low_dm_emb, D, c,n_gauss, point_color=colors, edge_sort = 'normal', output_path=output_path)
    # plot_fully_connected_points_edges(low_dm_emb, D, c,n_gauss, point_color=colors, edge_sort = 'ascending', output_path=output_path)
    # plot_fully_connected_points_edges(low_dm_emb, D, c,n_gauss, point_color=colors, edge_sort = 'descending', output_path=output_path)

    # # breakpoint()

    # output_path = os.path.join(output_folder, f"Fully_connected_clusters_point_blending{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    # plot_fully_connected_points(low_dm_emb, D, c,n_gauss, point_color=colors, blend_method = 'average', output_path=output_path)
    # print('generated fully connected points: averag')
    # plot_fully_connected_points(low_dm_emb, D, c,n_gauss, point_color=colors, blend_method = 'min', output_path=output_path)
    # print('generated fully connected points: min')
    # plot_fully_connected_points(low_dm_emb, D, c,n_gauss, point_color=colors, blend_method = 'max', output_path=output_path)
    # print('generated fully connected points: max')

    # breakpoint()
    ###______________________Delaunay analysis_____________________________________________

    
    # Apply Delaunay triangulation
    tri_delaunay = Delaunay(low_dm_emb)
    
    output_path = os.path.join(output_folder, f"delaunay_triangulation_{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    plot_delaunay_triangulation(tri_delaunay, low_dm_emb, c,n_gauss,  title="Delaunay Triangulation of t-SNE Output", 
                                point_color=colors, tri_color='blue', alpha=0.7, figsize=(8,6), output_path=output_path)
    

    # Extract triangle vertices (indices)
    tri_nodes = tri_delaunay.simplices
    # Create a triangulation object using the t-SNE coordinates
    triang_t_sne = tri.Triangulation(low_dm_emb[:, 0], low_dm_emb[:, 1], tri_nodes)

    # # ####___1._____________Distortion ratio using relative edge length ratio____________________________________

    all_tri_edges_len_hd, all_tri_edges_len_ld = calculate_delanay_edge_length(D, low_dm_emb, tri_nodes)
    
    
    relative_edge_ratio = all_tri_edges_len_hd/all_tri_edges_len_ld
    # breakpoint()
    # # relative_edge_ratio_log_1p = np.log1p(relative_edge_ratio)
    relative_edge_ratio_log = np.log(relative_edge_ratio)

    relative_edge_ratio_log_norm = (relative_edge_ratio_log - relative_edge_ratio_log.min()) / (relative_edge_ratio_log.max() - relative_edge_ratio_log.min())

    # ####__________analysis greater values___________________________________________
    # thr = 0.9
    # mask = relative_edge_ratio_log_norm > thr
    # counts = np.sum(mask, axis=1)
    # rows_with_values_greater_than_thr = np.where(counts > 0)[0]
    # for row in rows_with_values_greater_than_thr:
    #     print(f"Row {row}: {counts[row]} values greater than {thr}")
    
    # print(f'total greater than {thr}:', len(rows_with_values_greater_than_thr))

    # ################################################################################

    # bLogNorm_tri = True
    
    # output_path = os.path.join(output_folder, f"delaunay_tri_max_rel_lenth_colore_edges_{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    # # plot_triangulation_colored_edges_v2(tri_nodes, low_dm_emb, c, n_gauss, relative_edge_ratio, 
    # #                             title="Delaunay Triangulation with Colored Edges", 
    # #                             point_color=colors, cmap="hot", alpha=0.7, bLogNorm = bLogNorm_tri, figsize=(8,6), black_clust_points= False, output_path=output_path)
    
    # plot_triangulation_colored_edges_v3(tri_nodes, low_dm_emb, c, n_gauss, relative_edge_ratio_log_norm, 
    #                                      title="Delaunay Triangulation with Colored Edges", 
    #                                      point_color=colors, cmap="hot", alpha=0.7, figsize=(8,6), 
    #                                      black_clust_points=False, output_path=output_path)
    # output_path = os.path.join(output_folder, f"delaunay_tri_max_rel_colore_edges_black_points_{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    # # plot_triangulation_colored_edges(tri_nodes, low_dm_emb, c, n_gauss, relative_edge_ratio, 
    # #                             title="Delaunay Triangulation with Colored Edges", 
    # #                             point_color=colors, cmap="hot", alpha=0.7, bLogNorm = bLogNorm_tri, figsize=(8,6), black_clust_points= True, output_path=output_path)
    
    # plot_triangulation_colored_edges_v3(tri_nodes, low_dm_emb, c, n_gauss, relative_edge_ratio_log_norm, 
    #                                      title="Delaunay Triangulation with Colored Edges", 
    #                                      point_color=colors, cmap="hot", alpha=0.7, figsize=(8,6), 
    #                                      black_clust_points=True, output_path=output_path)
    # # breakpoint()
    ###__2._______________Distortion using maximum edge length each triangle____________________________________________________________________________________________________
    
    n_grid_points_inter = 1000

    # Select maximum edge ratio
    max_rel_edge_ratio_per_row = np.max(relative_edge_ratio, axis=1)
    max_rel_edge_ratio_per_row_log = np.log(max_rel_edge_ratio_per_row)
    max_rel_edge_ratio_per_row_log_norm = (max_rel_edge_ratio_per_row_log - max_rel_edge_ratio_per_row_log.min()) / (max_rel_edge_ratio_per_row_log.max() - max_rel_edge_ratio_per_row_log.min())

    min_rel_edge_ratio_per_row = np.min(relative_edge_ratio, axis=1)
    min_rel_edge_ratio_per_row_log = np.log(min_rel_edge_ratio_per_row)
    min_rel_edge_ratio_per_row_log_norm = (min_rel_edge_ratio_per_row_log - min_rel_edge_ratio_per_row_log.min()) / (min_rel_edge_ratio_per_row_log.max() - min_rel_edge_ratio_per_row_log.min())

    # breakpoint()

    # max_rel_edge_ratio_per_row_log = np.log1p(max_rel_edge_ratio_per_row)
    # max_rel_edge_ratio_per_row_norm_min_max = (max_rel_edge_ratio_per_row_log - max_rel_edge_ratio_per_row_log.min()) / (max_rel_edge_ratio_per_row_log.max() - max_rel_edge_ratio_per_row_log.min())

    # Get indices where values are greater than 0.5
    # indices_greater_than_05 = np.where(max_rel_edge_ratio_per_row_norm_min_max > 0.5)[0]

    # Get the values at those indices
    # values_greater_than_05 = max_rel_edge_ratio_per_row_norm_min_max[indices_greater_than_05]

    # print(len(indices_greater_than_05))

    output_path = os.path.join(output_folder, f"delaunay_tri_max_rel_lenth_histogram_{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    plot_histogram_values(max_rel_edge_ratio_per_row, bins= 30, output_path=output_path)
    # output_path = os.path.join(output_folder, f"delaunay_tri_max_rel_lenth_histogram_min_max_{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    # plot_histogram_values(max_rel_edge_ratio_per_row, bins= 30, output_path=output_path)
    

    output_path = os.path.join(output_folder, f"delaunay_tri_MAX_rel_lenth_distortion_clamp_{method_name}_{perplexity}_{dataset}_{fig_name_last}_new")
    # plot_triangulation_with_distortion_max_min(low_dm_emb, c,n_gauss, triang_t_sne, max_rel_edge_ratio_per_row_log_norm, bLogNorm = False, figsize=(10, 8),
    #                                    cmap_name="hot", point_color = colors, alpha=0.6, title="Triangles Colored by Distortion Ratio", output_path = output_path)
    
    intensity_interp_max_ratios, x_min, x_max, y_min, y_max = max_min_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, max_rel_edge_ratio_per_row_log_norm, blog = False, bclamping = True)
    
    
    plot_interpolation(n_gauss, low_dm_emb, c, colors, intensity_interp_max_ratios, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'white', output_path=output_path)

    # plot_triangulation_with_distortion_max_min(low_dm_emb, c,n_gauss, triang_t_sne, min_rel_edge_ratio_per_row_log_norm, bLogNorm = False, figsize=(10, 8),
    #                                    cmap_name="hot", point_color = colors, alpha=0.6, title="Triangles Colored by Distortion Ratio", output_path = output_path)
    
    intensity_interp_min_ratios, x_min, x_max, y_min, y_max = max_min_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, min_rel_edge_ratio_per_row_log_norm, blog = False, bclamping = False)
    output_path = os.path.join(output_folder, f"delaunay_tri_MIN_rel_lenth_distortion_{method_name}_{perplexity}_{dataset}_{fig_name_last}_new")
    plot_interpolation(n_gauss, low_dm_emb, c, colors, intensity_interp_min_ratios, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'white', output_path=output_path)

    breakpoint()
    # ####____ distortion ration using Area_________________________________
    # all_triangle_area_hd, all_triangle_area_ld = calculate_area_traingle_hd_ld(D, low_dm_emb, tri_nodes)
    # # breakpoint()
    # all_triangle_area_hd = (all_triangle_area_hd - all_triangle_area_hd.min()) / (all_triangle_area_hd.max() - all_triangle_area_hd.min())
    # all_triangle_area_ld = (all_triangle_area_ld - all_triangle_area_ld.min()) / (all_triangle_area_ld.max() - all_triangle_area_ld.min())
    # # breakpoint()
    # dist_ratio_area = np.array(all_triangle_area_hd)/(np.array(all_triangle_area_ld + 1e-8))

    # print('Dist', dist_ratio_area.min(),dist_ratio_area.max(), dist_ratio_area.mean())

    # check_triangle_data_integrity(triang_t_sne, all_triangle_area_hd, all_triangle_area_ld)

    # output_path = os.path.join(output_folder, f"delaunay_tri_area_distortion_{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    # plot_triangulation_with_distortion(low_dm_emb,  c,n_gauss, triang_t_sne, dist_ratio_area, bLogNorm = True, figsize=(10, 8),
    #                                    cmap_name="hot", point_color = colors, alpha=0.6, title="Triangles Colored by Distortion Ratio", output_path = output_path)
    
    
    # ## output_path = os.path.join(output_folder, f"delaunay_tri_area_histogram_{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    # ## plot_histogram_values(dist_ratio_area_normalized, bins= 30, output_path=output_path)    


    # ######___END____Delanay analysis_____________________________________________________________________________________________


    ###_________________Interpolation__________________________________________________________________________________________
    

    # # 1. ratios --> log --> interpolation --> normalization --> clamping --> normalization
    # intensity_interp_log, x_min, x_max, y_min, y_max = barycentric_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, relative_edge_ratio_log, blog = False, , bclamping = False)
    
    # output_path = os.path.join(output_folder, f"barycentric_interpolation_log_ratio_{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    # plot_interpolation(intensity_interp_log, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'black', output_path=output_path)
      
    # # 2. ratio --> interpolation --> log --> normalization --> clamping --> normalization
    # intensity_interp, x_min, x_max, y_min, y_max = barycentric_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, relative_edge_ratio, blog = True, , bclamping = False)
    
    # output_path = os.path.join(output_folder, f"barycentric_interpolation_ratio_after_log{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    # plot_interpolation(intensity_interp, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'black', output_path=output_path)
    
    # # 3. ratio --> interpolation --> normalization --> clamping --> normalization
    # intensity_interp, x_min, x_max, y_min, y_max = barycentric_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, relative_edge_ratio, blog = False, , bclamping = False)
    # np.count_nonzero(intensity_interp > 0.9)

    # output_path = os.path.join(output_folder, f"barycentric_interpolation_ratio_{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    # plot_interpolation(intensity_interp, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'black', output_path=output_path)

    # # 4. Hd_edge_lengths --> log --> interpolation --> normalization --> clamping --> normalization
    # all_tri_edges_len_hd_log = np.log(all_tri_edges_len_hd)
    # intensity_interp_hd_lengths_log, x_min, x_max, y_min, y_max = barycentric_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, all_tri_edges_len_hd_log, blog = False, , bclamping = True)

    # output_path = os.path.join(output_folder, f"barycentric_interpolation_hd_edge_lengths_log_{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    # plot_interpolation(intensity_interp_hd_lengths_log, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'black', output_path=output_path)

    # 5. Hd_edge_lengths --> interpolation --> normalization  --> clamping --> normalization
    intensity_interp_hd_lengths, x_min, x_max, y_min, y_max = barycentric_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, all_tri_edges_len_hd, blog = False, bclamping = False)

    output_path = os.path.join(output_folder, f"barycentric_interpolation_hd_edge_lengths_{method_name}_{perplexity}_{dataset}_{fig_name_last}")
    plot_interpolation(n_gauss, low_dm_emb, c, colors, intensity_interp_hd_lengths, x_min, x_max, y_min, y_max, bscatter_plot = True, background_color =  'black', output_path=output_path)

    #### ____________________________________________________________________________________________________________________

    ###__________________ Inter-intra cluster analysis of Low dimensioal space 2D__________________________________
    
    ld_distance_matrix, mean_cluster_distance_ld = inter_intra_cluster_pairwise_distance(low_dm_emb, orig_label, unique_labels, norm_distance = bNormFlag, norm_type=norm_type)
    output_path = os.path.join(output_folder, f"LD_clust_distance_{method_name}_{perplexity}_{dataset}{fig_name_last}")
    fig_title = 'Distances across clusters in low dimension'
    plot_pairwise_cluster_distance_v2(ld_distance_matrix, mean_cluster_distance_ld, orig_label, unique_labels, colors, fig_title, perplexity= perplexity, output_path=output_path)
    ## output_path = os.path.join(output_folder, f"LD_clust_mean_distance_{method_name}_{perplexity}")
    ## plot_mean_cluster_distance(mean_cluster_distance_ld, unique_labels, perplexity= perplexity, output_path=output_path)
    
    ##___________________________Compute relative compactness_____________________________________
    absolute_diff = compute_difference_3D_to_2D(mean_cluster_distance__hd, mean_cluster_distance_ld)

    abs_diff_off_diag =extract_off_diagonal(absolute_diff)
    abs_diff_off_diag_norm = normalize_and_sort_pairwise_dict(abs_diff_off_diag)
    output_path = os.path.join(output_folder, f"Absolute_difference_{method_name}_{perplexity}_{dataset}{fig_name_last}")
    plot_normalized_distances_with_custom_colors(abs_diff_off_diag_norm, colors, perplexity, 'HD to LD', output_path)
    
    # ##______________________ Distance matrix similarity______________________________

    # ##output_path = os.path.join(output_folder, f"cluster_difference_{method_name}_{perplexity}")
    # ##plot_hd_to_ld_clust_distance_difference(distance_matrix_hd, ld_distance_matrix, perplexity, output_path)


    ###################################################################################


    ## Data split for Inverse Projection model training
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
                                                            low_dm_emb, D, c, test_size=0.33, random_state=42, stratify=c)

    
    inverse_model = model_train(epochs = num_epochs, input_size= input_size, output_size= output_size, batch_size= batch_size, 
                                X_train= X_train, y_train=y_train,
                                out_folder=None)

    output_hd_emb, loss = model_test(X_test = X_test, y_test = y_test, model = inverse_model)

    hd_emb_inver = output_hd_emb.detach().cpu().numpy()
    
    print(f'Test loss for {method_name} perplexity {perplexity}: {loss:.4f}')

    


    ############################################################

    ###__________________ Inter-intra cluster analysis of Reconstructed High dimensioal space 3D__________________________________
    hd_distance_matrix, mean_cluster_distance_hd = inter_intra_cluster_pairwise_distance(hd_emb_inver, c_test, unique_labels, norm_distance = bNormFlag, norm_type=norm_type)
    output_path = os.path.join(output_folder, f"HD_reconst_clust_distance_{method_name}_{perplexity}{fig_name_last}")

    fig_title = 'Distances across clusters in reconstructed high dimension'
    plot_pairwise_cluster_distance_v2(hd_distance_matrix, mean_cluster_distance_hd, c_test, unique_labels, colors, fig_title, perplexity= perplexity, output_path=output_path)

    # output_path = os.path.join(output_folder, f"cluster_difference_hd_reconst_hd_{method_name}_{perplexity}")
    # plot_hd_to_ld_clust_distance_difference(distance_matrix_hd, hd_distance_matrix, perplexity, output_path)

    ###_____________________________________ Compactness ration 2D to 3D reconstructed______________________________________
    absolute_diff_2D_3D = compute_difference_3D_to_2D(mean_cluster_distance_ld, mean_cluster_distance_hd)

    abs_diff_off_diag_2D_3D =extract_off_diagonal(absolute_diff_2D_3D)
    abs_diff_off_diag_norm_2D_3D = normalize_and_sort_pairwise_dict(abs_diff_off_diag_2D_3D)
    # breakpoint()
    output_path = os.path.join(output_folder, f"Absolute_difference__2D_3D_{method_name}_{perplexity}_{dataset}{fig_name_last}")
    plot_normalized_distances_with_custom_colors(abs_diff_off_diag_norm_2D_3D, colors, perplexity,'LD to HD', output_path)

    # ##___________________________Compute relative compactness 3D_3D_recons_____________________________________

    # absolute_diff_3D_3D_recons = compute_difference_3D_to_2D(mean_cluster_distance__hd, mean_cluster_distance_hd)

    # abs_diff_off_diag_3D_3D_recons =extract_off_diagonal(absolute_diff_3D_3D_recons)
    # abs_diff_off_diag_norm__3D_3D_recons = normalize_and_sort_pairwise_dict(abs_diff_off_diag_3D_3D_recons)
    # # breakpoint()
    # output_path = os.path.join(output_folder, f"Absolute_difference__3D_3D_recons_{method_name}_{perplexity}_{dataset}{fig_name_last}")
    # plot_normalized_distances_with_custom_colors(abs_diff_off_diag_norm__3D_3D_recons, colors, perplexity, 'HD to HD reconstructed', output_path)

    # ## _____________________Total distortion_________________________________________________________________________________
    # distort_3D_2D_plus_2D_3D =add_nested_dicts(abs_diff_off_diag_norm, abs_diff_off_diag_norm_2D_3D )
    # distort_3D_2D_plus_2D_3D_norm = normalize_and_sort_pairwise_dict(distort_3D_2D_plus_2D_3D)
    # output_path = os.path.join(output_folder, f"Absolute_difference__total_proces_distort_{method_name}_{perplexity}_{dataset}{fig_name_last}")
    # plot_normalized_distances_with_custom_colors(distort_3D_2D_plus_2D_3D_norm, colors, perplexity, 'total' , output_path)
    
    # # Compute relative compactness for each cluster
    # relative_compactness_ratios_mean_reconst_hd = compute_relative_compactness(
    #     mean_cluster_distance_hd, mean_cluster_distance_ld, unique_labels
    # )
    
    # # output_path = os.path.join(output_folder, f"relative_compactness{method_name}_{perplexity}_{dataset}")
    # # plot_relative_compactness(relative_compactness_ratios, output_path)

    # output_path = os.path.join(output_folder, f"relative_compactness_mean_reconst_LD_HD_{method_name}_{perplexity}_{dataset}")
    # plot_relative_compactness(relative_compactness_ratios_mean_reconst_hd, output_path)

    ###__________________ Inter-intra cluster analysis of Original High dimensioal space 3D__________________________________

    hd_orig_distance_matrix, mean_orig_cluster_distance_hd = inter_intra_cluster_pairwise_distance(y_test, c_test, unique_labels, norm_distance = bNormFlag, norm_type=norm_type)
    # output_path = os.path.join(output_folder, f"HD_original_subclust_distance_{method_name}_{perplexity}")
    # plot_pairwise_cluster_distance_v2(hd_orig_distance_matrix, mean_orig_cluster_distance_hd, c_test, unique_labels, colors, perplexity= perplexity, output_path=output_path)

    # output_path = os.path.join(output_folder, f"cluster_difference_hd_reconst_hd_{method_name}_{perplexity}")
    # plot_hd_to_ld_clust_distance_difference(hd_orig_distance_matrix, hd_distance_matrix, perplexity, output_path)

    ###_____________________________________ Compactness ration 2D to 3D reconstructed______________________________________

    
    # # Compute relative compactness for each cluster
    # relative_compactness_ratios_mean_hd = compute_relative_compactness(
    #     mean_cluster_distance_hd, mean_orig_cluster_distance_hd, unique_labels
    # )
    
    # # output_path = os.path.join(output_folder, f"relative_compactness{method_name}_{perplexity}_{dataset}")
    # # plot_relative_compactness(relative_compactness_ratios, output_path)

    # output_path = os.path.join(output_folder, f"relative_compactness_mean_HD_HD_{method_name}_{perplexity}_{dataset}")
    # plot_relative_compactness(relative_compactness_ratios_mean_hd, output_path)

    # ###________________________ Jacobian norm______________________________________________________________
    print('start evaluating Jacobian norm ...')
    # spectral_norm_jacobian_heatmap_plot(low_dm_emb, c, n_gauss, num_grid_points, inverse_model, input_size, output_size,
    #                                     perplexity ,method_name, title_var, output_folder)
    try:
        jacob_norm, jacob_norm_log, U_matrix, Vt_matrix, sing_values, x_min, x_max, y_min, y_max, grid_points, xx,yy = jacobian_norm_calculation(low_dm_emb, num_grid_points, inverse_model.eval(), input_size, output_size)
        jacobian_norm_results[perplexity] = jacob_norm
        print("Jacobian is succesfull main loop")
    except Exception as e:
        print("Error occurred during Jacobian calculation:", str(e))

    ####_______________U matrix analysisi________________________________________________________
    # U_1 = U_matrix[:, :, 0, 0]  # First component of first column
    # U_2 = U_matrix[:, :, 0, 1]  # First component of second column

    # Initialize the scaled Vt array
    scaled_Vt = Vt_matrix.copy()  # Create a copy of the Vt array to hold the scaled values

    # Scale the first row of each Vt by the corresponding singular value
    for i in range(Vt_matrix.shape[0]):  # Iterate over the first dimension (200)
        for j in range(Vt_matrix.shape[1]):  # Iterate over the second dimension (200)
            # Scale the first row of the current Vt (i, j) by the first singular value
            scaled_Vt[i, j, 0, :] *= sing_values[i, j, 0]


    output_path = os.path.join(output_folder, f"Vt_vector_quiver_plot_{method_name}_{perplexity}_{dataset}")

    # plot_U_vector_field(U_1, U_2,U_matrix, xx, yy, output_path)
    plot_U_vector_field_updated(scaled_Vt, xx, yy, output_path)

    output_path = os.path.join(output_folder, f"U_vector_quiver_plot_{method_name}_{perplexity}_{dataset}_reduce_density")
    plot_U_vector_field_updated(scaled_Vt, xx, yy, output_path, bdensity_reduce=True)

    # breakpoint()
    
    # # _____________Calculate Projection Quality (Metrics calculation)__________________________________________

    # print('start evaluating projection metrics ...')

    # prj_quality_score_hd_ld = calculate_projection_metrics(D, low_dm_emb, c, n_neihbors_metrics)
    # ## prj_quality_score = calculate_projection_metrics(D, D, c, n_neihbors_metrics)
    # prj_quality_score_test_dt_hd_hd = calculate_projection_metrics(y_test, hd_emb_inver, c_test, n_neihbors_metrics)
    
    # # Dynamically handle metrics
    # prj_metrics_hd_to_ld = process_quality_metrics(prj_quality_score_hd_ld, prj_metrics_hd_to_ld)
    # prj_metrics_hd_to_hd = process_quality_metrics(prj_quality_score_test_dt_hd_hd, prj_metrics_hd_to_hd)

    # ###______________HD to LD & HD to HD metric differences_______________________
    # for metric in prj_metrics_hd_to_ld.metrics[5]:
        
    #     hd_to_ld_value = prj_metrics_hd_to_ld.get_metric(5, metric)[i]
    #     hd_to_hd_value = prj_metrics_hd_to_hd.get_metric(5, metric)[i]
    #     diff = abs(hd_to_ld_value - hd_to_hd_value)
    #     # breakpoint()
    #     prj_inverse_proj_diff.append({"perplexity": perplexity, "metric": metric, "difference": diff}) 
   
    ##################################################################################################################
    
    ######________________New analysis for clarity score_______________
    
    # Compute mean position (centeroid) for each class
    tsne_centroids = np.array([low_dm_emb[orig_label == label].mean(axis=0) for label in unique_labels])
    
    # Calculate the distance from each grid point to each mean position centroid
    distances = np.linalg.norm(grid_points[:, np.newaxis] - tsne_centroids, axis=2)

    scaler = MinMaxScaler()
    normalized_distances = scaler.fit_transform(distances)

    # For each grid point, assign the label of the closest centroid
    nearest_centroid_labels = np.argmin(normalized_distances, axis=1)

    # # ######_____ introducing KDE in decisoin boundary strength_____________________________

    # # grid_lbl_kde = kde_cluster_labeling(low_dm_emb, orig_label, grid_points)

    # # # Compute mean position (centeroid) for each class
    # # kde_centroids = np.array([grid_points[grid_lbl_kde == label].mean(axis=0) for label in unique_labels])
    # # # Calculate the distance from each grid point to each mean position centroid
    # # distances_kde = np.linalg.norm(grid_points[:, np.newaxis] - kde_centroids, axis=2)

    
    # ############################################################################################################

    ## Measureing decision boundary strength
    distances_reshaped = normalized_distances.reshape(num_grid_points, num_grid_points, n_gauss)

    decision_boundary_strength = np.zeros((num_grid_points, num_grid_points))

    ### weighted sum of distance and Jacobian norm
    decision_boundary_strength = (0.6 * jacob_norm) + (0.4 * np.min(distances_reshaped, axis=-1))

    # breakpoint()

    ##____________Clarity Score below________________________________
    ## 1. Boundary sharpness
    ## 2. Neighboorhood preservation
    ## 3. discision boundary strength
    

    ## _______Calculate Boundary Sharpness:___________
    ## 1. Compute gradient magnitude (sharpness)
    grad_x = sobel(decision_boundary_strength, axis=0)
    grad_y = sobel(decision_boundary_strength, axis=1)

    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)     # gradient_magnitude == gradient magnitude
    gradient_magnitude_mean = np.sqrt(grad_x**2 + grad_y**2).mean()

    ## 2. Variance of Gradients
    ## A high variance indicates a sharp boundary, while a low variance suggests a smooth transition. Helps capture local variations in boundary sharpness. Useful in cases where boundaries have mixed sharp and smooth regions.
    sharpness_variance = np.var(gradient_magnitude)
    
    ## 3. Laplacian operator (second order derivative)
    ## Laplacian operator captures intensity changes more aggressively. The Laplacian highlights sharp edges more than Sobel. It is less sensitive to small variations than first-order gradients.

    from scipy.ndimage import laplace

    laplacian_sharpness = np.abs(laplace(decision_boundary_strength)).mean()

    ## 4. Edge Density in Decision Regions
    ## Count the number of high-gradient points (edges) relative to the total region. A higher proportion of edges suggests a sharper boundary. If many pixels have high gradients, it indicates a sharper boundary. Helps normalize across different image sizes.

    ## threshold = np.percentile(gradient_magnitude, 90)  # Define high gradient threshold
    ## edge_density = (gradient_magnitude > threshold).sum() / gradient_magnitude.size

    
    # ## 5. Entropy of the Decision Boundary
    # ## A sharp decision boundary has low entropy because class labels transition abruptly. A smooth boundary has higher entropy due to uncertainty. Sharp boundaries = low entropy (high certainty). Smooth boundaries = high entropy (uncertain regions).
    
    ## from skimage.filters import entropy
    ## from skimage.morphology import disk
    ## entropy_map = entropy(decision_boundary_strength, disk(5))
    ## boundary_entropy = entropy_map.mean()

    boundary_sharpness = gradient_magnitude_mean + sharpness_variance + laplacian_sharpness # + edge_density
    
    ###______________________Calculate Class Separation_______________________________________________________
    ## Calculate Class Separation:
    
    new_metric = neighborhood_preservation_precision(D, low_dm_emb, n_neighbors=5)
    # new_metric = perplexity_quality_score(D, low_dm_emb, perplexity)
    ####_______________
    gamma, delta, epsilon = 0.00, 0.7, 0.3
    clarity = gamma * boundary_sharpness + delta * new_metric['new_metric'] + epsilon * decision_boundary_strength.mean()
    clarity = np.round(clarity, 3)
    
    ### if we choose only decision_boundary_strength: if the ratio of jacob_norm is hgiher than distance then the clarity is too much dependent on jacobian norm and gives high values if the width of decion boundary is high even the boundary is wron. if we make the jacob norm raion less than distance then unnecessary boundaries appear according to linear boundaries of distannce labelling. so we put the same ratio for them. However, still the clarity is not clear and dominant buy the jacobina norm and gives wrong clarity values. Then we introduce class seperation in the metric so it can understand the local structure of the data also and compare it with the original local structure of data.
    ###############################################################################################################################

    ####___________________optimize parameters using contrained optimization___________________
    # # Objective function to maximize clarity
    # from scipy.optimize import minimize
    # def objective(params):
    #     gamma, delta, epsilon = params
    #     clarity_score = gamma * gradient_magnitude_mean + delta * (lcmc_measure['lcmc'] - nd_measure['neighbor_dissimilarity']) + epsilon * decision_boundary_strength.mean()
    #     return -clarity_score  # Minimize the negative clarity to maximize

    # # Constraints: gamma + delta + epsilon = 1
    # constraints = ({'type': 'eq', 'fun': lambda params: sum(params) - 1})

    # # Bounds: 0 ≤ gamma, delta, epsilon ≤ 1
    # bounds = [(0, 1), (0, 1), (0, 1)]

    # # Initial guess (equal weights)
    # initial_guess = [1/3, 1/3, 1/3]

    # # Optimize
    # result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')

    # # Optimal values
    # gamma_opt, delta_opt, epsilon_opt = result.x
    # print("Optimal gamma:", gamma_opt)
    # print("Optimal delta:", delta_opt)
    # print("Optimal epsilon:", epsilon_opt)

    # # Compute final clarity using optimal values
    # clarity_opt = gamma_opt * gradient_magnitude_mean + delta_opt * (lcmc_measure['lcmc'] - nd_measure['neighbor_dissimilarity'])+ epsilon_opt * decision_boundary_strength.mean()
    # print("Optimized Clarity Score:", clarity_opt)
    # clarity = clarity_opt

    # ###_________________________optimizing gamma, delta and epsilon using OPTUNA_____________________
    # import optuna

    # def objective(trial):
    #     gamma = trial.suggest_float("gamma", 0, 1)
    #     delta = trial.suggest_float("delta", 0, 1)
    #     epsilon = trial.suggest_float("epsilon", 0, 1)

    #     # Normalize to sum 1
    #     total = gamma + delta + epsilon
    #     gamma /= total
    #     delta /= total
    #     epsilon /= total

    #     clarity = gamma * gradient_magnitude_mean + delta * lcmc_measure['lcmc'] + epsilon * decision_boundary_strength.mean()
    #     return clarity  # Direct maximization

    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=100)

    # # Best values
    # best_gamma, best_delta, best_epsilon = study.best_params["gamma"], study.best_params["delta"], study.best_params["epsilon"]
    # # print("Best gamma:", best_gamma)
    # # print("Best delta:", best_delta)
    # # print("Best epsilon:", best_epsilon)

    # clarity = best_gamma * gradient_magnitude_mean + best_delta * lcmc_measure['lcmc'] + best_epsilon * decision_boundary_strength.mean()    
    
    # ###############################################################################################################################

    ###________________________Plots_______________________________________________________________________________________
    
    ###_____ Grids projected back to 3D____________________________________________

    output_hd_emb, _ = model_test(grid_points, nearest_centroid_labels, model = inverse_model, bLossFlag= False)
    ## plot_3D_gaussian(output_hd_emb, nearest_centroid_labels, n_gauss, colors, output_folder, dataset)
    ## plot_3D_gaussian_no_label(output_hd_emb.detach().numpy(), nearest_centroid_labels,n_gauss, colors, dataset,  output_folder,  perplexity= perplexity)

    output_path = os.path.join(output_folder, f'grids_projection_{perplexity}_{dataset}{fig_name_last}.html')
    # plot_3D_gaussian(output_hd_emb.detach().numpy(), nearest_centroid_labels, n_gauss, colors, dataset, output_path,  perplexity= perplexity)

    # plot_3D_gaussian_no_label_with_projected_grids(output_hd_emb.detach().numpy(), D, orig_label,n_gauss, colors, dataset,  output_path,  perplexity= perplexity)

    # breakpoint()
    ###########################################################################
    selected_k = 5
    # breakpoint()
    print('start plotting heatmap ...')
    # spectral_norm_jacob_heatmap_vs_quality_metrics_plot_2Q_metrics(low_dm_emb, c, jacob_norm, prj_quality_score_hd_ld, prj_quality_score_test_dt_hd_hd, perplexity, n_gauss, selected_k,
    #                                                     x_min, x_max, y_min, y_max, clarity,
    #  
    #                                                    method_name, title_var, output_folder)

    output_path = os.path.join(output_folder, f"DR_{method_name}_{perplexity}_{dataset}")
    plot_dimensionality_reduction(low_dm_emb, c, n_gauss, method_name, title_var, perplexity, clarity, output_path=output_path)

    output_path = os.path.join(output_folder, f"spectral_norm_{method_name}_{perplexity}_{dataset}")
    plot_jacobian_spectral_norm_heatmap(low_dm_emb, c, jacob_norm, n_gauss, x_min, x_max, y_min, y_max, method_name, title_var, perplexity, clarity, output_path = output_path)

    output_path = os.path.join(output_folder, f"spectral_norm_log_{method_name}_{perplexity}_{dataset}")
    plot_jacobian_spectral_norm_heatmap(low_dm_emb, c, jacob_norm_log, n_gauss, x_min, x_max, y_min, y_max, method_name, title_var, perplexity, clarity, output_path = output_path)
    
    # breakpoint()
    # breakpoint()
    ###________________Plot cluster Distance decision boundary vs Jacobian______________________________________________________

    # output_path = os.path.join(output_folder, f"distance_jacob_{method_name}_{perplexity}")
    # plot_spectral_norm_vs_centroid_decision_boundaries(low_dm_emb, orig_label,n_gauss ,grid_points,  nearest_centroid_labels, tsne_centroids, jacob_norm,
    #                                                x_min, x_max, y_min, y_max,
    #                                                perplexity, clarity, colors, output_path = output_path)
    
    # # Visualize the combined decision boundaries
    # plt.figure(figsize=(10, 8))
    # plt.imshow(decision_boundary_strength, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='seismic', alpha=1.0)
    # # Add colorbar for the heatmap
    # plt.colorbar(label='decision boundary strength')
    # plt.title(f'perplexity:{perplexity} : clarity_score:{clarity}')
    # output_path = os.path.join(output_folder, f"decision_boundary_strength_{method_name}_{perplexity}")

    # plt.savefig(output_path)
    # plt.close()

    ############################################################################################

    
## Save Jacobian Norm Results
# save_metrics(prj_metrics_hd_to_ld, prj_met_hd_ld_filepath)
# save_metrics(jacobian_norm_results, jacobian_norm_filepath)
# save_metrics(low_dm_emb_per_perplexity, low_emb_filepath)
## print(f"Metrics saved to {prj_met_hd_ld_filepath}")
# Quality metric Plots
####################################################################

#_____________Below need to be uncomment________?????????????????????????????????????


# output_path = os.path.join(output_folder, f"projection_inv_proj_metric_differnece_{dataset}{fig_name_last}")
# plot_proj_inverse_proj_metric_differences(prj_inverse_proj_diff, output_path)

# output_path = os.path.join(output_folder, f"projection_quality_metrics_heatmap_{dataset}{fig_name_last}")
# plot_metrics_perplexity_heatmap(prj_metrics_hd_to_ld.metrics, perplexities,output_path, "HD to LD")

# output_path = os.path.join(output_folder, f"projection_Inverse_quality_metrics_heatmap_{dataset}{fig_name_last}")
# plot_metrics_perplexity_heatmap(prj_metrics_hd_to_hd.metrics, perplexities,output_path, "HD to HD")

# output_path = os.path.join(output_folder, f"projection_HD_LD_metrics_vs_rate_change_perplexity_heatmap_{dataset}{fig_name_last}")
# plot_metrics_vs_rate_change_perplexity_heatmap(prj_metrics_hd_to_ld.metrics, perplexities,output_path, "HD to LD")

# output_path = os.path.join(output_folder, f"projection_HD_HDmetrics_vs_rate_change_perplexity_heatmap_{dataset}{fig_name_last}")
# plot_metrics_vs_rate_change_perplexity_heatmap(prj_metrics_hd_to_hd.metrics, perplexities,output_path, "HD to HD")

# output_path = os.path.join(output_folder, f"projection_HD_LD_metrics_vs_rate_change_combined_heatmap_{dataset}{fig_name_last}")
# plot_metrics_perplexity_with_combined(prj_metrics_hd_to_ld.metrics, perplexities,output_path, "HD to LD")

# output_path = os.path.join(output_folder, f"projection_HD_HD_metrics_vs_rate_change_combined_heatmap_{dataset}{fig_name_last}")
# plot_metrics_perplexity_with_combined(prj_metrics_hd_to_hd.metrics, perplexities,output_path, "HD to HD")
# ######################################################################

# plot_metrics_vs_perplexity(perplexities, prj_metrics_hd_to_ld.metrics, method_name,title_var, output_folder)
# ### plot_metrics_vs_perplexity(perplexities, prj_metrics.metrics, method_name, output_folder, n_neihbors_metrics)
# print(f"Plots saved in folder: {output_folder}")



