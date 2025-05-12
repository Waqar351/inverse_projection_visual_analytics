import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse
from inver_project_model import NNinv, model_train, model_test
from datasets import *
from projections_methods import get_reducer
from plots import *
from utility import *
from scipy.spatial import Delaunay
import matplotlib.tri as tri
from pathlib import Path



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
    required=True,
    help="Choose dataset for dimensionality reduction."
)
parser.add_argument(
    "--num_dim",
    type=int,
    help="Choose dataset for dimensionality reduction."
)
args = parser.parse_args()


n_pts_per_gauss = 200  
num_grid_points = 200
bNormFlag = True
norm_type = 'global'

# perplexities = [2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
perplexities = [5] 
n_neihbors_metrics = [5]
figsize= (10, 10)
# colors = ['#FF0000', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF', '#0000FF', '#000000']
colors = ['#FF0000', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF', '#0000FF', '#000000', 
          '#FFA500', '#8000FF', '#FF1493']
##########______ NN model _____________________________________________________
input_size = 2
batch_size = 64
num_epochs = 1000
###____________________________________________________________________________
np.random.seed(5)

# #####################################
dataset = args.dataset
method = args.method
num_dim = args.num_dim
############################################

D,c, dim, output_size, n_gauss = selected_dataset_dt(dataset, num_dim, n_pts_per_gauss, cluster_spacing = 1.0, spread_factor = 0.01)
orig_label = c
orig_label = np.array(orig_label)
unique_labels = np.sort(np.unique(orig_label))   # Unique class labels

# breakpoint()

###_____________ Output folder__________________________________
if dataset == 'high_dim':
    output_folder = f"thesis_reproduced/testing_new/new_final_results/{dataset}_{num_dim}/{method}_plots_new_model"
    dataset = f'{dataset}_{num_dim}'
else:
    output_folder = f"thesis_reproduced/testing_new/new_final_results/{dataset}/{method}_plots_new_model"
os.makedirs(output_folder, exist_ok=True) 
##______________________________________________________________


###############################################################################################################################


###########______________Plot gaussian______________________________________________
output_path = os.path.join(output_folder, f'original_hd_data_{dataset}_{method}.html')
# plot_3D_gaussian(D, c,n_gauss, colors, dataset, output_path)
##########################################################################################################

###__________pairwise_inter_intra_cluster distance of High dimensional (original) data______________________________________
distance_matrix_hd, mean_cluster_distance__hd = inter_intra_cluster_pairwise_distance(D, orig_label, unique_labels, metric = 'euclidean', norm_distance = bNormFlag, norm_type=norm_type)
fig_title = 'Distances across clusters in high dimension'
output_path = os.path.join(output_folder, f"HD_clust_distance_Version_2_{dataset}_{method}")
plot_pairwise_cluster_distance_v2(distance_matrix_hd, mean_cluster_distance__hd, orig_label, unique_labels, colors, fig_title, output_path=output_path, figsize= figsize)


########################################################

jacobian_norm_results = {}
low_dm_emb_per_perplexity = {}

print('Start perplexity loop ...')

for i, perplexity in enumerate(perplexities):

    ### Initialize Dimensionality reductioin methods
    reducer, method_name, title_var = get_reducer(method, perplexity)
    print('run for tsne')
    # output_path = Path(os.path.join(output_folder, f"low_dim_emb_{method_name}_{perplexity}_{dataset}_{method}"))

    # low_dm_emb = load_metrics(output_path)

    low_dm_emb = reducer.fit_transform(D)
    low_dm_emb_per_perplexity[perplexity] = low_dm_emb
    print('projection completed')

    output_path = os.path.join(output_folder, f"low_dim_emb_{method_name}_{perplexity}_{dataset}_{method}")
    save_metrics(low_dm_emb, output_path)

    ### ____ F inverse Model_________________________________________________________________________________________

    print('Data splitting for model training ...')
    ## Data split for Inverse Projection model training
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
                                                            low_dm_emb, D, c, test_size=0.33, random_state=42, stratify=c)

    print('model training starts ...')
    inverse_model = model_train(epochs = num_epochs, input_size= input_size, output_size= output_size, batch_size= batch_size, 
                                X_train= X_train, y_train=y_train,
                                out_folder=None)

    output_hd_emb, loss = model_test(X_test = X_test, y_test = y_test, model = inverse_model)

    hd_emb_inver = output_hd_emb.detach().cpu().numpy()
    
    print(f'Test loss for {method_name} perplexity {perplexity}: {loss:.4f}')

    output_path = os.path.join(output_folder, f"inverse_model_{method_name}_{perplexity}_{dataset}_{method}")
    save_metrics(inverse_model.state_dict(), output_path)

    # inverse_model = NNinv(input_size=input_size, output_size=output_size)
    # model_path = os.path.join(output_folder, f"inverse_model_{method_name}_{perplexity}_{dataset}_{method}")
    # model_stat_dict = load_metrics(model_path)
    # inverse_model.load_state_dict(model_stat_dict)


    ####################################################################################################
    
    ###______________________Delaunay analysis_____________________________________________

    # Apply Delaunay triangulation
    tri_delaunay = Delaunay(low_dm_emb)
    
    print('Plotting Delaunay Traiangulation')
    output_path = os.path.join(output_folder, f"delaunay_triangulation_{method_name}_{perplexity}_{dataset}_{method}")
    plot_delaunay_triangulation(tri_delaunay, low_dm_emb, c,n_gauss,  title="Delaunay Triangulation of t-SNE Output", 
                                point_color=colors, tri_color='blue', alpha=1.0, figsize=(10,10), output_path=output_path)
    
    # Extract triangle vertices (indices)
    tri_nodes = tri_delaunay.simplices
    # Create a triangulation object using the t-SNE coordinates
    triang_t_sne = tri.Triangulation(low_dm_emb[:, 0], low_dm_emb[:, 1], tri_nodes)

    # breakpoint()
    
    ####________________ Discrete FTLE______________________________________________________
    # n_grid_points_inter = 1000
    n_grid_points_inter = 500
    # all_tri_ftle = calculate_FTLE_all_traingle(D, low_dm_emb, tri_nodes)

    # min_ftle , max_ftle= all_tri_ftle.min(), all_tri_ftle.max()
    # all_tri_ftle_normalized = (all_tri_ftle - min_ftle)/(max_ftle - min_ftle)

    # all_tri_ftle_log = np.log(all_tri_ftle)

    # all_tri_ftle_log_norm = (all_tri_ftle_log - all_tri_ftle_log.min()) / (all_tri_ftle_log.max() - all_tri_ftle_log.min())


    # # output_path = os.path.join(output_folder, f"ftle_{method_name}_{perplexity}_{dataset}_{method}")
    # # plot_triangulation_with_distortion_max_min(low_dm_emb, c,n_gauss, triang_t_sne, all_tri_ftle_log_norm, bLogNorm = False, figsize=figsize,
    # #                                     cmap_name="hot", point_color = colors, alpha=1.0, title="Triangles Colored by Distortion Ratio", output_path = output_path)

    # output_path = os.path.join(output_folder, f"ftle_interpolate_{method_name}_{perplexity}_{dataset}_{method}")
    # intensity_interp_ftle, x_min, x_max, y_min, y_max = max_min_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, all_tri_ftle_log_norm, blog = False, bclamping = True)
    # plot_interpolation(n_gauss, low_dm_emb, c, colors, intensity_interp_ftle, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'white', output_path=output_path, figsize=figsize)

    # # # breakpoint()

    # ####___1._____________Distortion ratio using relative edge length ratio____________________________________

    all_tri_edges_len_hd, all_tri_edges_len_ld = calculate_delanay_edge_length(D, low_dm_emb, tri_nodes)


    #####################################################
    print('Barcentirc Coordinate Interpolation starts ...')
    # # intensity_interp_cordinates, x_min, x_max, y_min, y_max = barycentric_interpolation_coordiantes(n_grid_points_inter, D, low_dm_emb, tri_delaunay, inverse_model.eval(), blog = False, bclamping = False)
    intensity_interp_cordinates, x_min, x_max, y_min, y_max = barycentric_interpolation_coordiantes_batch(n_grid_points_inter, D, low_dm_emb, tri_delaunay, inverse_model.eval(), blog = False, bclamping = False)
    # print('Barcentirc Coordinate Interpolation ends ...')
    
    # output_path = os.path.join(output_folder, f"barycentric_coordinates_interpolation_data_{method_name}_{perplexity}_{dataset}_{method}_new")
    # save_metrics(intensity_interp_cordinates, output_path)

    output_path = os.path.join(output_folder, f"barycentric_coordianates_interpolation_{method_name}_{perplexity}_{dataset}_{method}")
    plot_interpolation(n_gauss, low_dm_emb, c, colors, intensity_interp_cordinates, x_min, x_max, y_min, y_max, bscatter_plot = True, background_color =  'white', output_path=output_path, figsize=figsize)

    # breakpoint()


    ########################################################
    
    
    relative_edge_ratio = all_tri_edges_len_hd/all_tri_edges_len_ld

    relative_edge_ratio_log = np.log(relative_edge_ratio)

    relative_edge_ratio_log_norm = (relative_edge_ratio_log - relative_edge_ratio_log.min()) / (relative_edge_ratio_log.max() - relative_edge_ratio_log.min())


    # ################_________COLORED EDGES (NOT USED)_______________################################################

    # bLogNorm_tri = True
    
    # output_path = os.path.join(output_folder, f"delaunay_tri_max_rel_lenth_colore_edges_{method_name}_{perplexity}_{dataset}_{method}")
    
    # plot_triangulation_colored_edges_v3(tri_nodes, low_dm_emb, c, n_gauss, relative_edge_ratio_log_norm, 
    #                                      title="Delaunay Triangulation with Colored Edges", 
    #                                      point_color=colors, cmap="hot", alpha=0.7, figsize=(8,6), 
    #                                      black_clust_points=False, output_path=output_path)
    # output_path = os.path.join(output_folder, f"delaunay_tri_max_rel_colore_edges_black_points_{method_name}_{perplexity}_{dataset}_{method}")
    
    # plot_triangulation_colored_edges_v3(tri_nodes, low_dm_emb, c, n_gauss, relative_edge_ratio_log_norm, 
    #                                      title="Delaunay Triangulation with Colored Edges", 
    #                                      point_color=colors, cmap="hot", alpha=0.7, figsize=(8,6), 
    #                                      black_clust_points=True, output_path=output_path)
    ###__2._______________Distortion using maximum edge length each triangle____________________________________________________________________________________________________
    
    # print('Running Max / Min relative edge ratio')
    # n_grid_points_inter = 500

    # Select maximum edge ratio
    max_rel_edge_ratio_per_row = np.max(relative_edge_ratio, axis=1)
    max_rel_edge_ratio_per_row_log = np.log(max_rel_edge_ratio_per_row)
    max_rel_edge_ratio_per_row_log_norm = (max_rel_edge_ratio_per_row_log - max_rel_edge_ratio_per_row_log.min()) / (max_rel_edge_ratio_per_row_log.max() - max_rel_edge_ratio_per_row_log.min())

    output_path = os.path.join(output_folder, f"delaunay_tri_MAX_rel_lenth_distortion_clamp_{method_name}_{perplexity}_{dataset}_{method}_new")
    intensity_interp_max_ratios, x_min, x_max, y_min, y_max = max_min_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, max_rel_edge_ratio_per_row_log_norm, blog = False, bclamping = True)
    plot_interpolation(n_gauss, low_dm_emb, c, colors, intensity_interp_max_ratios, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'white', output_path=output_path, figsize=figsize)
    # ## plot_triangulation_with_distortion_max_min(low_dm_emb, c,n_gauss, triang_t_sne, max_rel_edge_ratio_per_row_log_norm, bLogNorm = False, figsize=(10, 8),
    # ##                                    cmap_name="hot", point_color = colors, alpha=0.6, title="Triangles Colored by Distortion Ratio", output_path = output_path)


    ##____ Minimum edge ratios________________________________________________
    # # min_rel_edge_ratio_per_row = np.min(relative_edge_ratio, axis=1)
    # # min_rel_edge_ratio_per_row_log = np.log(min_rel_edge_ratio_per_row)
    # # min_rel_edge_ratio_per_row_log_norm = (min_rel_edge_ratio_per_row_log - min_rel_edge_ratio_per_row_log.min()) / (min_rel_edge_ratio_per_row_log.max() - min_rel_edge_ratio_per_row_log.min())

    ## plot_triangulation_with_distortion_max_min(low_dm_emb, c,n_gauss, triang_t_sne, min_rel_edge_ratio_per_row_log_norm, bLogNorm = False, figsize=(10, 8),
    ##                                    cmap_name="hot", point_color = colors, alpha=0.6, title="Triangles Colored by Distortion Ratio", output_path = output_path)
    
    # intensity_interp_min_ratios, x_min, x_max, y_min, y_max = max_min_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, min_rel_edge_ratio_per_row_log_norm, blog = False, bclamping = False)
    # output_path = os.path.join(output_folder, f"delaunay_tri_MIN_rel_lenth_distortion_{method_name}_{perplexity}_{dataset}_{method}_new")
    # plot_interpolation(n_gauss, low_dm_emb, c, colors, intensity_interp_min_ratios, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'white', output_path=output_path,figsize=figsize)

    # ####____ distortion ration using Area_________________________________
    # all_triangle_area_hd, all_triangle_area_ld = calculate_area_traingle_hd_ld(D, low_dm_emb, tri_nodes)
    # # breakpoint()
    # all_triangle_area_hd = (all_triangle_area_hd - all_triangle_area_hd.min()) / (all_triangle_area_hd.max() - all_triangle_area_hd.min())
    # all_triangle_area_ld = (all_triangle_area_ld - all_triangle_area_ld.min()) / (all_triangle_area_ld.max() - all_triangle_area_ld.min())
    # # breakpoint()
    # dist_ratio_area = np.array(all_triangle_area_hd)/(np.array(all_triangle_area_ld + 1e-8))

    # print('Dist', dist_ratio_area.min(),dist_ratio_area.max(), dist_ratio_area.mean())

    # check_triangle_data_integrity(triang_t_sne, all_triangle_area_hd, all_triangle_area_ld)

    # output_path = os.path.join(output_folder, f"delaunay_tri_area_distortion_{method_name}_{perplexity}_{dataset}_{method}")
    # plot_triangulation_with_distortion(low_dm_emb,  c,n_gauss, triang_t_sne, dist_ratio_area, bLogNorm = True, figsize=(10, 8),
    #                                    cmap_name="hot", point_color = colors, alpha=0.6, title="Triangles Colored by Distortion Ratio", output_path = output_path)
    
    
    # ## output_path = os.path.join(output_folder, f"delaunay_tri_area_histogram_{method_name}_{perplexity}_{dataset}_{method}")
    # ## plot_histogram_values(dist_ratio_area_normalized, bins= 30, output_path=output_path)    


    # ######___END____Delanay analysis_____________________________________________________________________________________________


    ###_________________Interpolation__________________________________________________________________________________________
    print('Running Interpolation (Barycentric) ...')

    # # 1. ratios --> log --> interpolation --> normalization --> clamping --> normalization
    # intensity_interp_log, x_min, x_max, y_min, y_max = barycentric_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, relative_edge_ratio_log, blog = False, , bclamping = False)
    
    # output_path = os.path.join(output_folder, f"barycentric_interpolation_log_ratio_{method_name}_{perplexity}_{dataset}_{method}")
    # plot_interpolation(intensity_interp_log, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'black', output_path=output_path)
      
    # # 2. ratio --> interpolation --> log --> normalization --> clamping --> normalization
    # intensity_interp, x_min, x_max, y_min, y_max = barycentric_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, relative_edge_ratio, blog = True, , bclamping = False)
    
    # output_path = os.path.join(output_folder, f"barycentric_interpolation_ratio_after_log{method_name}_{perplexity}_{dataset}_{method}")
    # plot_interpolation(intensity_interp, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'black', output_path=output_path)
    
    # # 3. ratio --> interpolation --> normalization --> clamping --> normalization
    # intensity_interp, x_min, x_max, y_min, y_max = barycentric_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, relative_edge_ratio, blog = False, , bclamping = False)
    # np.count_nonzero(intensity_interp > 0.9)

    # output_path = os.path.join(output_folder, f"barycentric_interpolation_ratio_{method_name}_{perplexity}_{dataset}_{method}")
    # plot_interpolation(intensity_interp, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'black', output_path=output_path)

    # # 4. Hd_edge_lengths --> log --> interpolation --> normalization --> clamping --> normalization
    # all_tri_edges_len_hd_log = np.log(all_tri_edges_len_hd)
    # intensity_interp_hd_lengths_log, x_min, x_max, y_min, y_max = barycentric_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, all_tri_edges_len_hd_log, blog = False, , bclamping = True)

    # output_path = os.path.join(output_folder, f"barycentric_interpolation_hd_edge_lengths_log_{method_name}_{perplexity}_{dataset}_{method}")
    # plot_interpolation(intensity_interp_hd_lengths_log, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color =  'black', output_path=output_path)

    #__________ ABove not used_______________________
    # 5. Hd_edge_lengths --> interpolation --> normalization  --> clamping --> normalization

    intensity_interp_hd_lengths, x_min, x_max, y_min, y_max = barycentric_interpolation(n_grid_points_inter, low_dm_emb, tri_delaunay, all_tri_edges_len_hd, blog = False, bclamping = False)

    output_path = os.path.join(output_folder, f"barycentric_interpolation_hd_edge_lengths_data_{method_name}_{perplexity}_{dataset}_{method}_new")
    save_metrics(intensity_interp_hd_lengths, output_path)
    
    output_path = os.path.join(output_folder, f"barycentric_interpolation_hd_edge_lengths_{method_name}_{perplexity}_{dataset}_{method}")
    plot_interpolation(n_gauss, low_dm_emb, c, colors, intensity_interp_hd_lengths, x_min, x_max, y_min, y_max, bscatter_plot = True, background_color =  'white', output_path=output_path, figsize=figsize)

    # breakpoint()
    #### ____________________________________________________________________________________________________________________

    ###__________________ Inter-intra cluster analysis of Low dimensioal space 2D__________________________________
    
    # ld_distance_matrix, mean_cluster_distance_ld = inter_intra_cluster_pairwise_distance(low_dm_emb, orig_label, unique_labels, norm_distance = bNormFlag, norm_type=norm_type)
    output_path = os.path.join(output_folder, f"LD_clust_distance_{method_name}_{perplexity}_{dataset}_{method}")
    fig_title = 'Distances across clusters in low dimension'
    # plot_pairwise_cluster_distance_v2(ld_distance_matrix, mean_cluster_distance_ld, orig_label, unique_labels, colors, fig_title, perplexity= perplexity, output_path=output_path)
    
    ##___________________________Compute relative compactness (HD to LD)_____________________________________

    # print('Compute relative compactness plot (HD to LD) ...')
    # absolute_diff = compute_difference_3D_to_2D(mean_cluster_distance__hd, mean_cluster_distance_ld)

    # abs_diff_off_diag =extract_off_diagonal(absolute_diff)
    # abs_diff_off_diag_norm = normalize_and_sort_pairwise_dict(abs_diff_off_diag)
    # output_path = os.path.join(output_folder, f"Absolute_difference_{method_name}_{perplexity}_{dataset}_{method}")
    
    # plot_normalized_distances_with_custom_colors(abs_diff_off_diag_norm, colors, perplexity, 'HD to LD', output_path, figsize = figsize)


    # #### ____ F inverse Model_________________________________________________________________________________________

    # ## Data split for Inverse Projection model training
    # X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
    #                                                         low_dm_emb, D, c, test_size=0.33, random_state=42, stratify=c)

    
    # inverse_model = model_train(epochs = num_epochs, input_size= input_size, output_size= output_size, batch_size= batch_size, 
    #                             X_train= X_train, y_train=y_train,
    #                             out_folder=None)

    # output_hd_emb, loss = model_test(X_test = X_test, y_test = y_test, model = inverse_model)

    # hd_emb_inver = output_hd_emb.detach().cpu().numpy()
    
    # print(f'Test loss for {method_name} perplexity {perplexity}: {loss:.4f}')


    ####################################################################################################

    ###________________________ Jacobian norm______________________________________________________________
    print('start evaluating Jacobian norm ...')
    # spectral_norm_jacobian_heatmap_plot(low_dm_emb, c, n_gauss, num_grid_points, inverse_model, input_size, output_size,
    #                                     perplexity ,method_name, title_var, output_folder)
    try:
        jacob_norm, jacob_norm_log, x_min, x_max, y_min, y_max, grid_points, xx,yy = jacobian_norm_calculation(low_dm_emb, num_grid_points, inverse_model.eval(), input_size, output_size)
        # jacob_norm, jacob_norm_log, U_matrix, Vt_matrix, sing_values, x_min, x_max, y_min, y_max, grid_points, xx,yy = jacobian_norm_calculation(low_dm_emb, num_grid_points, inverse_model.eval(), input_size, output_size)
        jacobian_norm_results[perplexity] = jacob_norm
        print("Jacobian is succesfull main loop")
    except Exception as e:
        print("Error occurred during Jacobian calculation:", str(e))
    
    ## Save Jacobian Norm Results
    output_path = os.path.join(output_folder, f"Jacobian_results_{method_name}_{perplexity}_{dataset}")

    save_metrics(jacobian_norm_results, output_path)
    
    # ##########______________Plotting_________________________________________________####
    selected_k = 5
    # breakpoint()
    # print('start plotting heatmap ...')
    clarity = 0.2
    # output_path = os.path.join(output_folder, f"DR_{method_name}_{perplexity}_{dataset}")
    # plot_dimensionality_reduction(low_dm_emb, c, n_gauss, method_name, title_var, perplexity, clarity, output_path=output_path, figsize=figsize)


    output_path = os.path.join(output_folder, f"spectral_norm_{method_name}_{perplexity}_{dataset}")
    plot_jacobian_spectral_norm_heatmap(low_dm_emb, c, jacob_norm, n_gauss, x_min, x_max, y_min, y_max, method_name, title_var, perplexity, clarity , output_path = output_path, figsize=figsize)

    output_path = os.path.join(output_folder, f"spectral_norm_log_{method_name}_{perplexity}_{dataset}")
    plot_jacobian_spectral_norm_heatmap(low_dm_emb, c, jacob_norm_log, n_gauss, x_min, x_max, y_min, y_max, method_name, title_var, perplexity, clarity, output_path = output_path, figsize=figsize)

    ####_________ Ground truth_______________________________________________________________   
    print('Start plotting fully edges graphs with blending ...')
    output_path = os.path.join(output_folder, f"Fully_connected_clusters_point_blending_new_{method_name}_{perplexity}_{dataset}_{method}")
    # plot_fully_connected_points(low_dm_emb, D, c,n_gauss, point_color=colors, blend_methods = ['average', 'max', 'min'], output_path=output_path, figsize = figsize)
    # plot_fully_connected_points(low_dm_emb, D, c,n_gauss, point_color=colors, blend_methods = ['average'], output_path=output_path, figsize = figsize)
    plot_fully_connected_points_optimized(low_dm_emb, D, c,n_gauss, point_color=colors, blend_methods = ['average','max', 'min'], output_path=output_path, figsize = figsize)
    
    ###__________________ Inter-intra cluster analysis of Reconstructed High dimensioal space 3D__________________________________
    print('Compute relative compactness plot (LD to HD) ...')
    
    # # hd_distance_matrix, mean_cluster_distance_hd = inter_intra_cluster_pairwise_distance(hd_emb_inver, c_test, unique_labels, norm_distance = bNormFlag, norm_type=norm_type)
    # output_path = os.path.join(output_folder, f"HD_reconst_clust_distance_{method_name}_{perplexity}_{method}")

    # fig_title = 'Distances across clusters in reconstructed high dimension'
    # # plot_pairwise_cluster_distance_v2(hd_distance_matrix, mean_cluster_distance_hd, c_test, unique_labels, colors, fig_title, perplexity= perplexity, output_path=output_path, figsize = figsize)

    # ###_____________________________________ Compactness ration LD (2D) to HD (3D) reconstructed______________________________________
    # absolute_diff_2D_3D = compute_difference_3D_to_2D(mean_cluster_distance_ld, mean_cluster_distance_hd)

    # abs_diff_off_diag_2D_3D =extract_off_diagonal(absolute_diff_2D_3D)
    # abs_diff_off_diag_norm_2D_3D = normalize_and_sort_pairwise_dict(abs_diff_off_diag_2D_3D)
    # # breakpoint()
    # output_path = os.path.join(output_folder, f"Absolute_difference__2D_3D_{method_name}_{perplexity}_{dataset}_{method}")
    # plot_normalized_distances_with_custom_colors(abs_diff_off_diag_norm_2D_3D, colors, perplexity,'LD to HD', output_path, figsize = figsize)

    # breakpoint()
    # ##___________________________Compute relative compactness 3D_3D_recons_____________________________________

    # absolute_diff_3D_3D_recons = compute_difference_3D_to_2D(mean_cluster_distance__hd, mean_cluster_distance_hd)

    # abs_diff_off_diag_3D_3D_recons =extract_off_diagonal(absolute_diff_3D_3D_recons)
    # abs_diff_off_diag_norm__3D_3D_recons = normalize_and_sort_pairwise_dict(abs_diff_off_diag_3D_3D_recons)
    # # breakpoint()
    # output_path = os.path.join(output_folder, f"Absolute_difference__3D_3D_recons_{method_name}_{perplexity}_{dataset}_{method}")
    # plot_normalized_distances_with_custom_colors(abs_diff_off_diag_norm__3D_3D_recons, colors, perplexity, 'HD to HD reconstructed', output_path, figsize = figsize)

    


# print(f"Plots saved in folder: {output_folder}")



