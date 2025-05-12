# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import torch
# from sklearn.model_selection import train_test_split
# import argparse
# from inver_project_model import model_train, model_test
# from datasets import *
# from projections_methods import get_reducer
# from temp_projection_quality_metrics import trustworthiness, calculate_continuity, average_local_error
# # from plots import trustworthiness_plot, continuity_plot, average_local_error_plot
# from plots import *
# from projection_metrics import calculate_projection_metrics, ProjectionMetrics
# from utility import *
# from sklearn.manifold import TSNE
# import seaborn as sns
# from sklearn.cluster import KMeans
# from scipy.spatial import distance
# from sklearn.neighbors import NearestNeighbors
# from scipy.stats import mode
# from zadu_measure.local_continuity_meta_criteria import measure_lcmc
# from zadu_measure.neighbor_dissimilarity import measure_nd
# from zadu_measure.neighborhood_preservation_precision import neighborhood_preservation_precision
# from zadu_measure.perplexity_quality_score import perplexity_quality_score
# from scipy.ndimage import sobel
# import numpy as np
# from sklearn.metrics import pairwise_distances
# from scipy.spatial.distance import pdist, squareform
# from scipy.stats import pearsonr
# from sklearn.preprocessing import MinMaxScaler
# from scipy.spatial import Delaunay
# import matplotlib.tri as tri
# from matplotlib.collections import LineCollection
# from sklearn.decomposition import PCA
# import itertools

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import RectangleSelector
# from sklearn.manifold import TSNE
# from scipy.spatial import cKDTree

# np.random.seed(5)
# colors = ['#FF0000', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF', '#0000FF', '#000000', 
#           '#FFA500', '#8000FF', '#FF1493']

# X, y = har_dt_v2()
# dim = X.shape[1]
# output_size = dim
# n_gauss = len(np.unique(y))

# # print('tsne')
# # Apply t-SNE to reduce to 2D
# # low_dm_emb = TSNE(n_components=2, perplexity=5, init="random", random_state=0).fit_transform(X)
# # print('projection completed')

# # # Step 3: Save Embeddings
# # with open("tsne_embeddings.pkl", "wb") as f:
# #     pickle.dump((low_dm_emb, y), f)  # Save both embeddings and labels
# # print("t-SNE embeddings saved!")

# def load_tsne():
#     """Reloads the saved t-SNE embeddings from a file."""
#     with open("tsne_embeddings.pkl", "rb") as f:
#         return pickle.load(f)

# low_dm_emb, y_loaded = load_tsne()  # Reload saved data
# print("t-SNE embeddings loaded!")

import numpy as np
import os
# from sklearn.model_selection import train_test_split
import argparse
# from inver_project_model import model_train, model_test
from datasets import *
from projections_methods import get_reducer
from plots import *
from utility import *
# from scipy.spatial import Delaunay
# import matplotlib.tri as tri
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
# from sklearn.manifold import TSNE
# from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay




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
bNormFlag = False
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
# orig_label = np.array(orig_label)
unique_labels = np.sort(np.unique(orig_label))   # Unique class labels
y_loaded, y = orig_label, orig_label
X = D
###_____________ Output folder__________________________________
if dataset == 'high_dim':
    output_folder = f"thesis_reproduced/testing_new/new_final_results/{dataset}_{num_dim}/{method}_plots_new_model"
    dataset = f'{dataset}_{num_dim}'
else:
    output_folder = f"thesis_reproduced/testing_new/new_final_results/{dataset}/{method}_plots_new_model"
os.makedirs(output_folder, exist_ok=True) 
##______________________________________________________________

perplexity = 5
### Initialize Dimensionality reductioin methods
reducer, method_name, title_var = get_reducer(method, perplexity)
print('run for tsne')
low_dm_emb = reducer.fit_transform(D)

# # Apply Delaunay triangulation
# tri_delaunay = Delaunay(low_dm_emb)
# # Extract triangle vertices (indices)
# tri_nodes = tri_delaunay.simplices

# # Extract edges from Delaunay triangulation (edges are pairs of points)
# edges = []
# for simplex in tri_nodes:
#     edges.extend([(simplex[i], simplex[j]) for i in range(3) for j in range(i + 1, 3)])
#     # breakpoint()

# # Remove duplicate edges (edges are undirected)
# edges = list(set(edges))

# distance_matrix_hd, mean_cluster_distance__hd = inter_intra_cluster_pairwise_distance(D, orig_label, unique_labels, metric = 'euclidean', norm_distance = bNormFlag, norm_type=norm_type)

# # Extract the lengths of the edges from distance matrix
# # edge_lengths = [(i, j, np.linalg.norm(D[i] - D[j])) for i, j in edges]
# edge_lengths_n = [(i, j, distance_matrix_hd[i,j]) for i, j in edges]

# # Create a base pairwise distance matrix with grey color
# highlighted_matrix = np.zeros_like(distance_matrix_hd)

# highlighted_matrix.fill(0.0)  # Grey (0.5) in the colormap

# # breakpoint()

# for i, j, length in edge_lengths_n:
#     # Change the color at the selected edges (for example, set to 1 for bright color)
#     # highlighted_matrix[i, j] = length  # Bright color (1)
#     highlighted_matrix[i, j] = length  # Bright color (1)
#     highlighted_matrix[j, i] = length  # Bright color (1)
#     # highlighted_matrix[j, i] = length  # Symmetric matrix (undirected edges)

# # log_matrix = np.log1p(highlighted_matrix) 
# # log_matrix = (log_matrix - log_matrix.min())/((log_matrix.max() - log_matrix.min()))
# # log_matrix = (highlighted_matrix - highlighted_matrix.min())/((highlighted_matrix.max() - highlighted_matrix.min()))
# log_matrix = highlighted_matrix 
# # breakpoint()

# # Plot the highlighted matrix
# # Create the plot
# plt.figure(figsize=(10, 10))
# plt.imshow(log_matrix, cmap='grey', interpolation='nearest')

# # Add colorbar
# plt.colorbar()

# # Title and Show Plot
# plt.title("Pairwise Distance Matrix with Highlighted Edges")
# plt.xticks([])  # Hide x-axis labels
# plt.yticks([])  # Hide y-axis labels
# plt.show()


# breakpoint()
# #______________________________________

# --- Storage for selection ---
selected_points = {1: [], 2: []}
selected_classes = {1: None, 2: None}  # Track class of each selection
active_boxes = 0  # Track drawn boxes

def onselect(eclick, erelease):
    """Callback function to select points within the drawn box."""
    global active_boxes

    if active_boxes >= 2:
        print("Selection complete! No more boxes allowed.")
        return  

    box_id = active_boxes + 1
    x_min, x_max = sorted([eclick.xdata, erelease.xdata])
    y_min, y_max = sorted([eclick.ydata, erelease.ydata])

    # Find points inside the box
    mask = (low_dm_emb[:, 0] >= x_min) & (low_dm_emb[:, 0] <= x_max) & \
           (low_dm_emb[:, 1] >= y_min) & (low_dm_emb[:, 1] <= y_max)

    selected = np.where(mask)[0]  # Indices of selected points
    selected_points[box_id] = selected.tolist()

    # Identify the class of the selected points
    unique_classes = np.unique(y[selected])
    if len(unique_classes) == 1:
        selected_classes[box_id] = int(unique_classes[0])
        print(f"\n Selected {len(selected)} points from Class {unique_classes[0]} for Box {box_id}")
    else:
        print(f"\n Box {box_id} contains mixed classes: {unique_classes}. Select carefully.")

    active_boxes += 1

    if active_boxes == 2:
        # disable_selector()
        apply_pca_on_selected_classes()
        ask_continue()

# def disable_selector():
#     """Disable further selection after two boxes are drawn."""
#     rect_selector.set_active(False)
#     print("\n Selection process completed! Preparing PCA visualization...")

#     apply_pca_on_selected_classes()
    



def apply_pca_on_selected_classes():
    """Apply PCA to the two selected classes and highlight the selected points."""
    class_1, class_2 = selected_classes[1], selected_classes[2]

    if class_1 is None or class_2 is None:
        print(" Error: One of the selected boxes does not contain a single class.")
        return

    print(f"\n Applying PCA on Class {class_1} and Class {class_2}...")

    # Extract original data for the two classes
    mask_classes = (y == class_1) | (y == class_2)
    X_filtered = X[mask_classes]
    y_filtered = y[mask_classes]

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_filtered)

    # Identify selected points in PCA space
    # selected_indices = np.concatenate([selected_points[1], selected_points[2]])
    # mask_selected = np.isin(np.where(mask_classes)[0], selected_indices)

    mask_selected_1 = np.isin(np.where(mask_classes)[0], selected_points[1])
    mask_selected_2 = np.isin(np.where(mask_classes)[0], selected_points[2])
    
    # Plot PCA
    fig, ax_pca = plt.subplots(figsize=(10, 7))
    for class_label in [class_1, class_2]:
        mask = y_filtered == class_label
        ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       color = colors[class_label],
                       label=f'Class {class_label}', edgecolor='k', alpha=0.8)

    # Highlight selected points class 1
    ax_pca.scatter(X_pca[mask_selected_1, 0], X_pca[mask_selected_1, 1], 
                   color=colors[class_1], edgecolor='yellow', s=150, label=f"Selected Points {class_1}")
    
    # Highlight selected points_class 2
    ax_pca.scatter(X_pca[mask_selected_2, 0], X_pca[mask_selected_2, 1], 
                   color=colors[class_2], edgecolor='yellow', s=150, label=f"Selected Points {class_2}")

    ax_pca.set_title("PCA Visualization with Highlighted Selection")
    ax_pca.legend()
    plt.show()

def ask_continue():
    """Ask the user if they want to select more boxes."""
    global active_boxes

    active_boxes = 0  # Reset the box counter
    selected_points.clear()  # Clear previously selected points
    print("\nPlease select two new boxes.")

    response = input("\nDo you want to select another 2 boxes? (1/0): ").strip().lower()
    if response == '1':
        active_boxes = 0  # Reset the box counter
        selected_points.clear()  # Clear previously selected points
        print("\nPlease select two new boxes.")
    elif response == '0':
        print("Selection process completed!")
    else:
        print("\nInvalid input. Please type '1' or '0'.")
        ask_continue()

# # # Create Rectangle Selector
# rect_selector = RectangleSelector(ax, onselect, useblit=True, interactive=True, 
#                                   props=dict(facecolor='red', alpha=0.3))

# plt.show()

def start_selector():
    """Start interactive selection and allow repeated selection of boxes."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot t-SNE scatter plot
    for i in range(n_gauss):
        ax.scatter(low_dm_emb[y_loaded == i, 0], low_dm_emb[y_loaded == i, 1], color=colors[i],
                   label=f'Gaussian{i + 1}', zorder=3, edgecolor='k', s=50)

    ax.set_title("t-SNE Visualization with Interactive Selection")
    
    # Create Rectangle Selector
    rect_selector = RectangleSelector(ax, onselect, useblit=True, interactive=True,
                                      props=dict(facecolor='red', alpha=0.3))
    plt.show()


# Initialize global variables
selected_points = {1: [], 2: []}  # Store selected points from each box
active_boxes = 0  # Keep track of active boxes

start_selector()




# def apply_pca_on_selected_classes():
#     """Apply PCA to the two selected classes and highlight the selected points."""
#     class_1, class_2 = selected_classes[1], selected_classes[2]

#     if class_1 is None or class_2 is None:
#         print(" Error: One of the selected boxes does not contain a single class.")
#         return

#     print(f"\n Applying PCA on Class {class_1} and Class {class_2}...")

#     # Extract original data for the two classes
#     mask_classes = (y == class_1) | (y == class_2)
#     X_filtered = X[mask_classes]
#     y_filtered = y[mask_classes]

#     # Apply PCA
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_filtered)

#     # Identify selected points in PCA space
#     # selected_indices = np.concatenate([selected_points[1], selected_points[2]])
#     # mask_selected = np.isin(np.where(mask_classes)[0], selected_indices)

#     mask_selected_1 = np.isin(np.where(mask_classes)[0], selected_points[1])
#     mask_selected_2 = np.isin(np.where(mask_classes)[0], selected_points[2])
    
#     # Plot PCA
#     fig, ax_pca = plt.subplots(figsize=(10, 7))
#     for class_label in [class_1, class_2]:
#         mask = y_filtered == class_label
#         ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1], 
#                        color = colors[class_label],
#                        label=f'Class {class_label}', edgecolor='k', alpha=0.8)

#     # Highlight selected points class 1
#     ax_pca.scatter(X_pca[mask_selected_1, 0], X_pca[mask_selected_1, 1], 
#                    color=colors[class_1], edgecolor='yellow', s=150, label="Selected Points {class_1}")
    
#     # Highlight selected points_class 2
#     ax_pca.scatter(X_pca[mask_selected_2, 0], X_pca[mask_selected_2, 1], 
#                    color=colors[class_2], edgecolor='yellow', s=150, label=f"Selected Points {class_2}")

#     ax_pca.set_title("PCA Visualization with Highlighted Selection")
#     ax_pca.legend()
#     plt.show()

# def ask_continue():
#     """Ask the user if they want to select more boxes."""
#     global active_boxes

#     active_boxes = 0  # Reset the box counter
#     selected_points.clear()  # Clear previously selected points
#     print("\nPlease select two new boxes.")

#     response = input("\nDo you want to select another 2 boxes? (1/0): ").strip().lower()
#     if response == '1':
#         active_boxes = 0  # Reset the box counter
#         selected_points.clear()  # Clear previously selected points
#         print("\nPlease select two new boxes.")
#     elif response == '0':
#         print("Selection process completed!")
#     else:
#         print("\nInvalid input. Please type '1' or '0'.")
#         ask_continue()

# # # # Create Rectangle Selector
# # rect_selector = RectangleSelector(ax, onselect, useblit=True, interactive=True, 
# #                                   props=dict(facecolor='red', alpha=0.3))

# # plt.show()

# def start_selector():
#     """Start interactive selection and allow repeated selection of boxes."""
#     fig, ax = plt.subplots(figsize=(10, 10))

#     # Plot t-SNE scatter plot
#     for i in range(n_gauss):
#         ax.scatter(low_dm_emb[y_loaded == i, 0], low_dm_emb[y_loaded == i, 1], color=colors[i],
#                    label=f'Gaussian{i + 1}', zorder=3, edgecolor='k', s=50)

#     ax.set_title("t-SNE Visualization with Interactive Selection")
    
#     # Create Rectangle Selector
#     rect_selector = RectangleSelector(ax, onselect, useblit=True, interactive=True,
#                                       props=dict(facecolor='red', alpha=0.3))
#     plt.show()


# # Initialize global variables
# selected_points = {1: [], 2: []}  # Store selected points from each box
# active_boxes = 0  # Keep track of active boxes

# start_selector()