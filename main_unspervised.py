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
from sklearn.impute import KNNImputer

# Construct the path to the datasets folder
datasets_folder = os.path.join(project_dir, "datasets", "Market_segmentation_kaggle")

dataset = 'market_segmentation'

# File path
data_path = os.path.join(datasets_folder, "Customer_Data.csv")
# Read the CSV file while skipping the first row (column names)

data = pd.read_csv(data_path, header=None, skiprows=1)
data = data.iloc[:, 1:]  # Remove the first column

###############################
input_size = 2
batch_size = 64
num_epochs = 1000
output_size = dim = data.shape[1]
num_grid_points = 200
method = 'tsne'

output_folder = f"thesis_reproduced/testing_new/perplexity_analysis/{dataset}/{method}_plots_new_model"
fig_name_last = f'{dataset}'
os.makedirs(output_folder, exist_ok=True) 
###############################

# null_columns=data.columns[data.isnull().any()]
# cleaner = KNNImputer(n_neighbors=9, weights="distance")
# numerical = data[null_columns].select_dtypes(exclude = "object").columns
# data[numerical] = cleaner.fit_transform(data[numerical])
# breakpoint()
data = data.dropna()  # Remove rows with any NaN values

data = data.sample(n=600, random_state=42)  # Randomly select 10 rows

# Normalize the features
scaler = MinMaxScaler()
D = scaler.fit_transform(data)


print('Data reday')
perplexities = [2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50]
perplexities = [ 40]
perplexity = 40
for perplexity in perplexities:
    reducer, method_name, title_var = get_reducer(method, perplexity)

    print('apply tsne')
    low_dm_emb = reducer.fit_transform(D)


        # low_dm_emb_per_perplexity[perplexity] = low_dm_emb

    ## Data split for Inverse Projection model training
    X_train, X_test, y_train, y_test = train_test_split(low_dm_emb, D, test_size=0.33, random_state=42)

    inverse_model = model_train(epochs = num_epochs, input_size= input_size, output_size= output_size, batch_size= batch_size, 
                                    X_train= X_train, y_train=y_train,
                                    out_folder=None)

    output_hd_emb, loss = model_test(X_test = X_test, y_test = y_test, model = inverse_model)

    hd_emb_inver = output_hd_emb.detach().cpu().numpy()

    print(f'Test loss for {method_name} perplexity {perplexity}: {loss:.4f}')

    # ###________________________ Jacobian norm______________________________________________________________
    print('start evaluating Jacobian norm ...')
    # spectral_norm_jacobian_heatmap_plot(low_dm_emb, c, n_gauss, num_grid_points, inverse_model, input_size, output_size,
    #                                     perplexity ,method_name, title_var, output_folder)
    try:
        jacob_norm, x_min, x_max, y_min, y_max, grid_points = jacobian_norm_calculation(low_dm_emb, num_grid_points, inverse_model.eval(), input_size, output_size)
        # jacobian_norm_results[perplexity] = jacob_norm
        print("Jacobian is succesfull main loop")
    except Exception as e:
        print("Error occurred during Jacobian calculation:", str(e))

    ###########################################################################
    selected_k = 5
    # breakpoint()
    print('start plotting heatmap ...')
    # spectral_norm_jacob_heatmap_vs_quality_metrics_plot_2Q_metrics(low_dm_emb, c, jacob_norm, prj_quality_score_hd_ld, prj_quality_score_test_dt_hd_hd, perplexity, n_gauss, selected_k,
    #                                                     x_min, x_max, y_min, y_max, clarity,
    #  
    #                                                    method_name, title_var, output_folder)
    output_path = os.path.join(output_folder, f"spectral_norm_{method_name}_{perplexity}_{dataset}")
    plot_jacobian_spectral_norm_heatmap_unsupervised(low_dm_emb, jacob_norm, x_min, x_max, y_min, y_max, method_name, title_var, perplexity, output_path = output_path)


# Select seeds interactively
num_seeds = 5  # Adjust as needed

output_path = os.path.join(output_folder, f"seed_generation_{method_name}_{perplexity}_{dataset}")
seeds = select_seeds(jacob_norm, num_seeds, x_min, x_max, y_min, y_max, output_path)

breakpoint()
# Convert seeds into a marker array for the random walker
markers = np.zeros(jacob_norm.shape, dtype=np.uint)
for i, (x, y) in enumerate(seeds):
    markers[y, x] = i + 1  # Assign unique label to each seed

# Run the random walker algorithm
segmented_grid = random_walker(jacob_norm, markers, beta=100, mode='bf')

# breakpoint()

# # Run random walker algorithm
# segmented_grid = segment_with_random_walker(jacob_norm, num_seeds=5, threshold=0.7)

# Plot results
plt.figure(figsize=(8, 6))
plt.imshow(
        # segmented_grid,
        # np.flipud(segmented_grid),
        np.fliplr(segmented_grid),
        extent=(x_min, x_max, y_min, y_max),
        origin='upper',
        cmap='seismic',
        alpha=1.0
    )
plt.colorbar(label="Segment Labels")
plt.scatter(seeds[:, 1], seeds[:, 0], c='blue', marker='o')
# seed_x = x_min + (x_max - x_min) * (seeds[:, 1] / segmented_grid.shape[1])
# seed_y = y_min + (y_max - y_min) * (seeds[:, 0] / segmented_grid.shape[0])

# plt.scatter(seed_x, seed_y, c='blue', marker='o')

plt.title("Random Walk Segmentation Based on Spectral Norm")

output_path = os.path.join(output_folder, f"random_walk{method_name}_{perplexity}_{dataset}")
plt.savefig(f"{output_path}.png", dpi=300, format='png', bbox_inches="tight")
plt.show()
# plt.gca().invert_yaxis()
plt.close()
# # Plot Results
# plt.figure(figsize=(8, 6))
# plt.imshow(labels, cmap='tab10', origin='lower')
# plt.colorbar(label="Segmented Regions")
# plt.title("Random Walk Segmentation of Spectral Norm Grid with Merging")
# plt.show()