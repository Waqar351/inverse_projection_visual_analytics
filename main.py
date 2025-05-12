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
    help="Choose 'iris' or 'gaussian' dataset for dimensionality reduction."
)
args = parser.parse_args()

# __________Defining parameters__________________________________
input_size = 2
num_grid_points = 100
batch_size = 64
num_epochs = 200
# perplexities = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 170, 200]
# perplexities = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
perplexities = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]
# perplexities = [1, 2, 3, 4] 
# n_neihbors_metrics = [5, 10, 15, 20, 25, 30, 35]
n_neihbors_metrics = [5]

np.random.seed(5)

if args.dataset == "gaussian":
    dim = 3
    output_size = dim
    n_gauss = 6
    n_pts_per_gauss = 300

    D, c, centers = gaussian_dt(n_gauss, n_pts_per_gauss, dim)

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

else:
    raise ValueError("Invalid dataset name. Choose 'iris' or 'gaussian'.")


#____________ Define Paths _________________________________________________________________________

output_folder = f"thesis_reproduced/testing_new/temp/{args.dataset}/{args.method}_plots_new_model"
os.makedirs(output_folder, exist_ok=True)

prj_met_hd_ld_fold = f"{output_folder}/hd_ld_metrics"
os.makedirs(prj_met_hd_ld_fold, exist_ok=True)
prj_met_hd_ld_filepath = f"{prj_met_hd_ld_fold}/{args.dataset}_{args.method}_prj_metrics_hd_ld.pkl"

jacobian_norm_folder = f"{output_folder}/jacobian_norms"
os.makedirs(jacobian_norm_folder, exist_ok=True)
jacobian_norm_filepath = f"{jacobian_norm_folder}/{args.dataset}_{args.method}_jacobian_norm.pkl"

low_emb_folder = f"{output_folder}/low_emb_space"
os.makedirs(low_emb_folder, exist_ok=True)
low_emb_filepath = f"{low_emb_folder}/{args.dataset}_{args.method}_low_emb_space.pkl"
#_____________________________________________________________________________________________________________



# if os.path.exists(prj_met_hd_ld_filepath):
#     # Load precomputed metrics
#     print(f"Loading precomputed metrics from {prj_met_hd_ld_filepath}...")
#     prj_metrics_hd_to_ld = load_metrics(prj_met_hd_ld_filepath)

# else:
prj_metrics_hd_to_ld = ProjectionMetrics()
jacobian_norm_results = {}
projection_metrics_results = {}
low_dm_emb_per_perplexity = {}

print('Start perplexity loop ...')

for perplexity in perplexities:

    reducer, method_name, title_var = get_reducer(args.method, perplexity)

    

    low_dm_emb = reducer.fit_transform(D)
    low_dm_emb_per_perplexity[perplexity] = low_dm_emb



    ## Data split for Inverse Projection model training
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
                                                            low_dm_emb, D, c, test_size=0.33, random_state=42, stratify=c)

    
    inverse_model = model_train(epochs = num_epochs, input_size= input_size, output_size= output_size, batch_size= batch_size, 
                                X_train= X_train, y_train=y_train,
                                out_folder=output_folder)

    output_hd_emb, loss = model_test(X_test = X_test, y_test = y_test, model = inverse_model)

    hd_emb_inver = output_hd_emb.detach().cpu().numpy()
    
    print(f'Test loss for {method_name} perplexity {perplexity}: {loss:.4f}')

    # _____________Calculate Projection Quality__________________________________________

    print('start evaluating projection metrics ...')
    prj_quality_score_hd_ld = calculate_projection_metrics(D, low_dm_emb, c, n_neihbors_metrics)
    ## prj_quality_score = calculate_projection_metrics(D, D, c, n_neihbors_metrics)
    ## prj_quality_score = calculate_projection_metrics(y_test, hd_emb_inver, c_test, n_neihbors_metrics)
    
    # Dynamically handle metrics
    prj_metrics_hd_to_ld = process_quality_metrics(prj_quality_score_hd_ld, prj_metrics_hd_to_ld)
    ## projection_metrics_results[perplexity] = prj_metrics_hd_to_ld.metrics
    #####################################################################

    print('start evaluating Jacobian norm ...')
    # spectral_norm_jacobian_heatmap_plot(low_dm_emb, c, n_gauss, num_grid_points, inverse_model, input_size, output_size,
    #                                     perplexity ,method_name, title_var, output_folder)
    jacob_norm, x_min, x_max, y_min, y_max, grid_points = jacobian_norm_calculation(low_dm_emb, num_grid_points, inverse_model.eval(), input_size, output_size)
    jacobian_norm_results[perplexity] = jacob_norm

    selected_k = 5

    print('start plotting heatmap ...')
    spectral_norm_jacob_heatmap_vs_quality_metrics_plot(low_dm_emb, c, jacob_norm, prj_quality_score_hd_ld, perplexity, n_gauss, selected_k,
                                                        x_min, x_max, y_min, y_max,
                                                        method_name, title_var, output_folder)
    
    
## Save Jacobian Norm Results
save_metrics(prj_metrics_hd_to_ld, prj_met_hd_ld_filepath)
save_metrics(jacobian_norm_results, jacobian_norm_filepath)
save_metrics(low_dm_emb_per_perplexity, low_emb_filepath)
## print(f"Metrics saved to {prj_met_hd_ld_filepath}")
# Quality metric Plots

plot_metrics_vs_perplexity(perplexities, prj_metrics_hd_to_ld.metrics, method_name,title_var, output_folder)
# plot_metrics_vs_perplexity(perplexities, prj_metrics.metrics, method_name, output_folder, n_neihbors_metrics)


print(f"Plots saved in folder: {output_folder}")
