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
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.patches import Rectangle
# from sklearn.manifold import TSNE
# from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from pathlib import Path
from itertools import combinations
from sklearn.metrics import pairwise_distances
import matplotlib.patches as patches
import matplotlib.tri as tri



# ##################################################################################################################################

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



# breakpoint()
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
# low_dm_emb = reducer.fit_transform(D)

output_path = Path(os.path.join(output_folder, f"low_dim_emb_{method_name}_{perplexity}_{dataset}_{method}"))

low_dm_emb = load_metrics(output_path)
# breakpoint()

output_path = Path(os.path.join(output_folder, f"barycentric_interpolation_hd_edge_lengths_data_{method_name}_{perplexity}_{dataset}_{method}_new"))
intensity_interp_hd_lengths = load_metrics(output_path)

x_min, x_max = np.min(low_dm_emb[:, 0]), np.max(low_dm_emb[:, 0])
y_min, y_max = np.min(low_dm_emb[:, 1]), np.max(low_dm_emb[:, 1])


# Apply Delaunay triangulation
tri_delaunay = Delaunay(low_dm_emb)
# Extract triangle vertices (indices)
tri_nodes = tri_delaunay.simplices

# --- Storage for selection ---

selected_points_each_box = {}
selected_classes = {}  # Track class of each selection
selected_boxes = {}  # Store box coordinates (x_min, x_max, y_min, y_max)
edges_per_box = {}   # Create edges between all pairs of points in each box

active_boxes = 0  # Track drawn boxes

# fig, ax_ld = plt.subplots(figsize=(10, 10))
fig, ax_ld = plt.subplots(figsize=(10, 10), constrained_layout=True)


def plot_data():
    """Re-draws the scatter plot and highlights selected edges & rectangles."""
    ax_ld.clear()

    # Plot Delaunay Triangulation
    ax_ld.triplot(low_dm_emb[:, 0], low_dm_emb[:, 1], tri_nodes, color='blue', linewidth=0.5, linestyle='-')

    # Plot scatter points (Clustered points)
    for i in range(n_gauss):
        ax_ld.scatter(low_dm_emb[c == i, 0], low_dm_emb[c == i, 1], color=colors[i],
                      edgecolor='k', s=50, zorder=3)

    x_min, x_max = np.min(low_dm_emb[:, 0]), np.max(low_dm_emb[:, 0])
    y_min, y_max = np.min(low_dm_emb[:, 1]), np.max(low_dm_emb[:, 1])
    ax_ld.imshow(intensity_interp_hd_lengths,
            extent=(x_min, x_max, y_min, y_max),
            origin='lower',
            cmap='hot',
            alpha=1.0,
            # interpolation='nearest'
        )

    # # Draw selected rectangles
    for key, box in selected_boxes.items():
        x_min, x_max, y_min, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                linewidth=4, edgecolor='#FFD700', facecolor='none', linestyle='-', zorder=3)
        ax_ld.add_patch(rect)

        # Add label (e.g., "Box 1", "Box 2", etc.)
        label_x = x_min + (x_max - x_min) / 2
        label_y = y_max + 0.02 * (y_max - y_min)  # Adjust vertical offset if needed
        ax_ld.text(label_x, label_y, f"Box {key}", color='black', fontsize=12,
                ha='center', va='bottom', fontweight='bold', zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # Remove default axes labels
    ax_ld.axis('equal')
    plt.axis("off")

    ax_ld.set_xticks([])  
    ax_ld.set_yticks([])
    ax_ld.spines['top'].set_visible(True)
    ax_ld.spines['right'].set_visible(True)
    ax_ld.spines['bottom'].set_visible(True)
    ax_ld.spines['left'].set_visible(True)

    output_path = os.path.join(output_folder, f"Sub_Delanay_with_selected_sub_boxes_{dataset}_{method}")

    plt.draw()
    fig.canvas.draw_idle()  # Update the figure
    
    plt.ioff()  # Turn off interactive mode
    plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
    plt.ion()


# ---------- Continue button (initially hidden) ----------
continue_ax = fig.add_axes([0.4, 0.01, 0.2, 0.05])
continue_button = Button(continue_ax, 'Continue', color='lightgray', hovercolor='lightblue')
continue_button.ax.set_visible(False)


def show_continue_button():
    continue_button.ax.set_visible(True)
    fig.canvas.draw_idle()


def onselect(eclick, erelease):
    """Callback function to select points within the drawn box."""
    global active_boxes


    box_id = active_boxes + 1
    x_min, x_max = sorted([eclick.xdata, erelease.xdata])
    y_min, y_max = sorted([eclick.ydata, erelease.ydata])

    selected_boxes[box_id] = (x_min, x_max, y_min, y_max)  # Store box coordinates

    # Find points inside the box
    mask = (low_dm_emb[:, 0] >= x_min) & (low_dm_emb[:, 0] <= x_max) & \
           (low_dm_emb[:, 1] >= y_min) & (low_dm_emb[:, 1] <= y_max)

    selected = np.where(mask)[0]  # Indices of selected points
    selected_points_each_box[box_id] = selected.tolist()

    # Identify the class of the selected points
    unique_classes = np.unique(orig_label[selected])


    # breakpoint()
    if len(unique_classes) == 1:
        selected_classes[box_id] = int(unique_classes[0])
        print(f"\n Selected {len(selected)} points from Class {unique_classes[0]} for Box {box_id}")
    else:
        print(f"\n Box {box_id} contains mixed classes: {unique_classes}. Select carefully.")

    active_boxes += 1
    
    show_continue_button()
    plt.draw()  # Refresh the plot
    plot_data()
    # print('Draw another box')
    create_display_edges_between_selected_boxes()
    


def update_distance_matrix(data, selected_points_each_box, bNorm_mean = False):
    """
    Builds a full symmetric pairwise distance matrix from selected points across all boxes.
    Computes intra- and inter-box mean distances for each block.
    
    Args:
        data: full dataset (N x D)
        selected_points_each_box: dict of {box_id: [indices]}
    
    Returns:
        dist_matrix: (M x M) full pairwise distance matrix
        labels: list of box ids for each point
        selected_points_flat: ordered list of all selected indices
        block_mean_distances: dict {(box_i, box_j): mean_distance}
    """
    selected_points_flat = []
    labels = []

    for box_id, indices in selected_points_each_box.items():
        selected_points_flat.extend(indices)
        labels.extend([box_id] * len(indices))

    # Get the data of the selected points
    selected_data = data[selected_points_flat]
    dist_matrix = pairwise_distances(selected_data, selected_data, metric='euclidean')

    # Compute block-wise means
    labels = np.array(labels)
    unique_boxes = sorted(set(labels))
    block_mean_distances = {}

    for i in unique_boxes:
        idx_i = np.where(labels == i)[0]
        for j in unique_boxes:
            idx_j = np.where(labels == j)[0]

            block = dist_matrix[np.ix_(idx_i, idx_j)]

            if i == j:
                # Intra-cluster: remove diagonal to compute true mean
                if block.shape[0] > 1:
                    # mask = ~np.eye(len(idx_i), dtype=bool)
                    # mean_val = np.mean(block[mask])
                    mean_val = np.mean(block)
                else:
                    mean_val = 0.0
            else:
                mean_val = np.mean(block)

            block_mean_distances[(i, j)] = mean_val
    
    if bNorm_mean:
        # Normalize the block mean distances if more than 1 box is selected
        if len(unique_boxes) > 1:
            values = np.array(list(block_mean_distances.values()))
            min_val, max_val = np.min(values), np.max(values)

            # Avoid division by zero
            if max_val != min_val:
                for key in block_mean_distances:
                    block_mean_distances[key] = (block_mean_distances[key] - min_val) / (max_val - min_val)


    return dist_matrix, labels.tolist(), selected_points_flat, block_mean_distances


def visualize_distance_matrix(dist_matrix, labels, mean_block_distance, box_colors=None, figsize=(10, 10), output_path = None):
    """
    Visualize the pairwise distance matrix with intra and inter cluster blocks highlighted.
    
    Args:
        dist_matrix: (N x N) symmetric matrix of distances
        labels: list of box IDs for each row/col in dist_matrix
        box_colors: optional dict {box_id: color}
        figsize: size of the figure
        title: plot title
    """
    unique_boxes = sorted(set(labels))
    N = len(labels)
    border_thickness = 10

    # Count how many points in each box

    # Compute cluster boundaries dynamically
    num_points_per_cluster = [np.sum(np.array(labels) == i) for i in unique_boxes]
    cumulative_positions = np.cumsum([0] + num_points_per_cluster)  # Start and end positions for each cluster

    # Create figure
    # fig, ax = plt.subplots(figsize=figsize)
    fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    # ax.set_aspect(aspect='auto')
    # plt.axis("off")
    im = ax.imshow(dist_matrix, cmap='hot')

    # Remove x and y ticks but keep the box
    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines[:].set_visible(False)  # Hide all borders first

    # breakpoint()
    # Draw only the outer left and outer top borders
    for i in range(len(unique_boxes)):
        pad_border_corner = 0
        start = cumulative_positions[i] - pad_border_corner
        end = cumulative_positions[i + 1] - pad_border_corner if i + 1 < len(cumulative_positions) else cumulative_positions[-1] - pad_border_corner

        color = colors[i]  # Assign cluster color
        # color = box_colors[i]  # Assign cluster color
        # breakpoint()
        # Left border (Outer vertical) - precisely aligned
        ax.plot([-0.5, -0.5], [start-0.5, end-0.5], color=color, linewidth=border_thickness, solid_capstyle='butt', clip_on=False)  

        # Top border (Outer horizontal) - precisely aligned
        ax.plot([start-0.5, end-0.5], [-0.5, -0.5], color=color, linewidth=border_thickness, solid_capstyle='butt', clip_on=False)  


    # Annotate mean distances inside cluster blocks
    for i, start_i in enumerate(cumulative_positions[:-1]):
        box_i = unique_boxes[i]
        for j, start_j in enumerate(cumulative_positions[:-1]):
            box_j = unique_boxes[j]
            mean_distance = mean_block_distance[box_i, box_j]
            center_x = (start_j + cumulative_positions[j + 1]) / 2
            center_y = (start_i + cumulative_positions[i + 1]) / 2
            ax.text(center_x, center_y, f"{mean_distance:.2f}", color="black", ha="center", va="center", fontsize=7,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3", alpha=0.6))

    # Final plot tweaks
    # ax.set_title(title)
    plt.tight_layout()
    plt.show()

    # Save or display the plot
    if output_path:
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_display_edges_between_selected_boxes():

    selected_points_flat = []
    labels = []

    for box_id, indices in selected_points_each_box.items():
        selected_points_flat.extend(indices)
        labels.extend([box_id] * len(indices))

    n_class = len(np.unique(labels))
    # Get the data of the selected points
    sub_orig_data = D[selected_points_flat]
    sub_low_dm_emb = low_dm_emb[selected_points_flat]

    # Apply Delaunay triangulation
    sub_tri_delaunay = Delaunay(sub_low_dm_emb)
    # plot_delaunay_triangulation(sub_tri_delaunay, sub_low_dm_emb, labels,n_class ,  title="Delaunay Triangulation of t-SNE Output", 
    #                             point_color=colors, tri_color='blue', alpha=1.0, figsize=(10,10))

    # Extract triangle vertices (indices)
    sub_tri_nodes = sub_tri_delaunay.simplices
    # Create a triangulation object using the t-SNE coordinates
    sub_triang_t_sne = tri.Triangulation(low_dm_emb[:, 0], low_dm_emb[:, 1], tri_nodes)

    all_sub_tri_edges_len_hd, all_sub_tri_edges_len_ld = calculate_delanay_edge_length(sub_orig_data, sub_low_dm_emb, sub_tri_nodes)
    
    n_grid_points_inter = 500
    print('run barycentric interpolation')
    sub_intensity_interp_hd_lengths, x_min_bary, x_max_bary, y_min_bary, y_max_bary = barycentric_interpolation(n_grid_points_inter, sub_low_dm_emb, sub_tri_delaunay, all_sub_tri_edges_len_hd, blog = False, bclamping = False)

    output_path = os.path.join(output_folder, f"barycentric_interpolation_sub_hd_edge_lengths_{method_name}_{perplexity}_{dataset}_{method}")
    plot_interpolation(n_class, sub_low_dm_emb, labels, colors, sub_intensity_interp_hd_lengths, x_min_bary, x_max_bary, y_min_bary, y_max_bary, bscatter_plot = True, background_color =  'white', output_path = output_path, figsize=figsize)

    # breakpoint()
    
    # # Rebuild distance matrix
    # dist_matrix, labels, point_indices,  dist_matrix_mean = update_distance_matrix(D, selected_points_each_box, bNorm_mean = True)

    # # output_path = os.path.join(output_folder, f"Delanay_edges_over_sub_distance_matrix_box_dynamic_{dataset}_{method}")
    # output_path = os.path.join(output_folder, f"Sub_Delanay_with_selected_sub_boxes_distance_matrix_dynamic_{dataset}_{method}")
    # visualize_distance_matrix(dist_matrix, labels, dist_matrix_mean, output_path = output_path)

    ##################################################################################
    print('Select next box ... ')


def start_selector():
    """Start interactive selection and allow repeated selection of boxes."""
    # Plot Delaunay Triangulation
    ax_ld.triplot(low_dm_emb[:, 0], low_dm_emb[:, 1], tri_nodes, color='blue', linewidth=0.5, linestyle='-')

    # Plot scatter points (Clustered points)
    for i in range(n_gauss):
        ax_ld.scatter(low_dm_emb[c == i, 0], low_dm_emb[c == i, 1], color=colors[i],
                      edgecolor='k', s=50, zorder=3)

    ax_ld.imshow(intensity_interp_hd_lengths,
            extent=(x_min, x_max, y_min, y_max),
            origin='lower',
            cmap='hot',
            alpha=1.0,
            # interpolation='nearest'
        )
    
    ax_ld.axis('equal')
    plt.axis("off")


    ax_ld.set_title("t-SNE Visualization with Interactive Selection")
    # ax_ld.legend()

    global rect_selector
    # # Create Rectangle Selector
    rect_selector = RectangleSelector(ax_ld, onselect, useblit=True, interactive=True,
                                      props=dict(facecolor='red', alpha=0.3))

    plt.show()



start_selector()





########################################################################################################################

