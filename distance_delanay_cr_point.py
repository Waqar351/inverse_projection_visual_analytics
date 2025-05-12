import numpy as np
import os
import argparse
import itertools
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from datasets import *
from projections_methods import get_reducer
from plots import *
from utility import *
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.spatial import Delaunay
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



###_____Margin_______________________________________
margin = 0.05  # 5% margin (adjustable)
global x_min , x_max, y_min, y_max

x_min, x_max = np.min(low_dm_emb[:, 0]), np.max(low_dm_emb[:, 0])
y_min, y_max = np.min(low_dm_emb[:, 1]), np.max(low_dm_emb[:, 1])

# # Expand the range by a percentage of the range
# x_range = x_max - x_min
# y_range = y_max - y_min

# x_min -= margin * x_range
# x_max += margin * x_range
# y_min -= margin * y_range
# y_max += margin * y_range



# Apply Delaunay triangulation
tri_delaunay = Delaunay(low_dm_emb)
# Extract triangle vertices (indices)
tri_nodes = tri_delaunay.simplices

# --- Selection Storage ---
selected_points = {1: [], 2: []}
selected_classes = {1: None, 2: None}
active_selection = 1  # Track selection group

# Initialize figure
fig, ax_ld = plt.subplots(figsize=(15, 15), constrained_layout=True)

def plot_data():
    """Re-draws the scatter plot and highlights selected points."""
    ax_ld.clear()  # Clear previous plot

    # # Plot all points
    # for i in range(n_gauss):
    #     ax_ld.scatter(low_dm_emb[orig_label == i, 0], low_dm_emb[orig_label == i, 1], 
    #                   color=colors[i], label=f'Class {i}', edgecolor='k', s=50)

    # # Plot Delaunay Triangulation
    ax_ld.triplot(low_dm_emb[:, 0], low_dm_emb[:, 1], tri_nodes, color='blue', linewidth=0.3, linestyle='-')

    # Plot scatter points (Clustered points)
    for i in range(n_gauss):
        ax_ld.scatter(low_dm_emb[c == i, 0], low_dm_emb[c == i, 1], color=colors[i],
                      edgecolor='k', s=50, zorder=3)

    # x_min, x_max = np.min(low_dm_emb[:, 0]), np.max(low_dm_emb[:, 0])
    # y_min, y_max = np.min(low_dm_emb[:, 1]), np.max(low_dm_emb[:, 1])
    ax_ld.imshow(intensity_interp_hd_lengths,
            extent=(x_min, x_max, y_min, y_max),
            origin='lower',
            cmap='hot',
            alpha=1.0,
            # interpolation='nearest'
        )
    
    # ax_ld.set_xlim(x_min, x_max)
    # ax_ld.set_ylim(y_min, y_max)

    # Highlight selected points
    for group, points in selected_points.items():
        if points:
            ax_ld.scatter(low_dm_emb[points, 0], low_dm_emb[points, 1], 
                          color='#FFD700', marker='x', s=100, linewidths=3, label=f'Selected Group {group}', zorder=5)

    # ax_ld.set_title("t-SNE Visualization with Interactive Selection (Click to Select)")
    # ax_ld.legend()
    
    
    # Remove default axes labels
    ax_ld.axis('equal')
    plt.axis("off")

    ax_ld.set_xticks([])  
    ax_ld.set_yticks([])
    ax_ld.spines['top'].set_visible(True)
    ax_ld.spines['right'].set_visible(True)
    ax_ld.spines['bottom'].set_visible(True)
    ax_ld.spines['left'].set_visible(True)


    output_path = os.path.join(output_folder, f"Delanay_with_selected_points_{dataset}_{method}")
    plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
    fig.canvas.draw_idle()  # Update plot



def on_click(event):
    """Handles point selection on mouse click."""
    global active_selection

    if event.xdata is None or event.ydata is None:
        return  # Ignore clicks outside the plot area

    # Find the closest point to the clicked position
    distances = np.sqrt((low_dm_emb[:, 0] - event.xdata) ** 2 + (low_dm_emb[:, 1] - event.ydata) ** 2)
    selected_idx = np.argmin(distances)

    # Avoid selecting the same point twice
    if selected_idx in selected_points[1] or selected_idx in selected_points[2]:
        print("Point already selected, choose another.")
        return

    selected_points[active_selection].append(selected_idx)
    selected_classes[active_selection] = orig_label[selected_idx]  # Assign class

    print(f"Selected Point {selected_idx} from Class {orig_label[selected_idx]} in Group {active_selection}")

    # Update plot with the new selection
    plot_data()

    # Check if both selections are complete
    if active_selection == 2 and len(selected_points[1]) > 0 and len(selected_points[2]) > 0:
        create_display_edges_between_selected_boxes()
        ask_continue()
    else:
        active_selection = 2  # Switch to second selection group

def create_display_edges_between_selected_boxes():
    """Compute edges between selected points from two groups and highlight them."""
    if selected_classes[1] is None or selected_classes[2] is None:
        print("Error: Both selections must contain points.")
        return

    print(f"\n Selected Class {selected_classes[1]} and Class {selected_classes[2]}...")

    selected_points_c_1 = selected_points[1]
    selected_points_c_2 = selected_points[2]

    # Generate edges between selected points from two groups
    edges = list(itertools.product(selected_points_c_1, selected_points_c_2))

    # Compute pairwise distance matrix
    distance_matrix_hd, _ = inter_intra_cluster_pairwise_distance(D, orig_label, np.unique(orig_label), metric='euclidean', norm_distance=bNormFlag, norm_type=norm_type)

    output_path = os.path.join(output_folder, f"Delanay_with_selected_points_distance_matrix_{dataset}_{method}")
    # Plot selected edges on distance matrix
    plot_highlight_select_edges_on_distance_matrix(
        distance_matrix_hd, edges, orig_label, np.unique(orig_label), colors=colors, border_thickness=10, output_path=output_path ,figsize=(20, 20)
    )

def ask_continue():
    """Ask the user if they want to select more points."""
    global active_selection

    response = input("\nDo you want to select another set of points? (1: Yes, 0: No): ").strip().lower()
    if response == '1':
        selected_points.clear()  # Reset selected points
        selected_classes.clear()  # Reset classes
        active_selection = 1  # Reset selection group
        print("\nPlease select new points.")
    else:
        print("Selection process completed!")

def start_selector():
    """Start interactive point selection using mouse clicks."""
    # fig, ax_ld = plt.subplots(figsize=(15, 15))

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
    # ax_ld.set_xlim(x_min, x_max)
    # ax_ld.set_ylim(y_min, y_max)
    ax_ld.axis('equal')
    plt.axis("off")

    # Connect mouse click event to handler
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

# Run the selector
start_selector()
