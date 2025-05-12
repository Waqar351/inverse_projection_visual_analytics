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
from scipy.spatial import Delaunay
from pathlib import Path
from matplotlib.widgets import PolygonSelector
from matplotlib.widgets import RectangleSelector, Line2D

from shapely.geometry import LineString
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

# Apply Delaunay triangulation
tri_delaunay = Delaunay(low_dm_emb)
# Extract triangle vertices (indices)
tri_nodes = tri_delaunay.simplices

###################################################################################################################################################

# # Initialize selection variables
# selected_edges = []
# line_start = None
# line_end = None
# drawn_line = None  # To store the drawn line for visualization
fig, ax_ld = plt.subplots(figsize=(10, 10))

# # Function to find closest edges to a given line
# def find_edges_under_line(start, end):
#     """Finds edges that intersect with the drawn line."""
#     selected = []
    
#     for tri in tri_nodes:
#         edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
#         for edge in edges:
#             p1, p2 = low_dm_emb[edge[0]], low_dm_emb[edge[1]]
            
#             # Check if the edge intersects with the drawn line
#             if lines_intersect(start, end, p1, p2):
#                 selected.append(edge)
    
#     return selected

# # Function to check if two line segments intersect
# def lines_intersect(A, B, C, D):
#     """Returns True if line segment AB intersects with CD"""
#     def ccw(P, Q, R):
#         return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])

#     return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# # Function to start drawing
# def on_press(event):
#     global line_start, drawn_line
#     if event.xdata is None or event.ydata is None:
#         return
#     line_start = (event.xdata, event.ydata)
#     if drawn_line:
#         drawn_line.remove()
#     drawn_line, = ax_ld.plot([], [], color='cyan', linewidth=2)

# # Function to update drawing
# def on_motion(event):
#     global line_start, drawn_line
#     if line_start is None or event.xdata is None or event.ydata is None:
#         return
#     x_vals = [line_start[0], event.xdata]
#     y_vals = [line_start[1], event.ydata]
#     drawn_line.set_xdata(x_vals)
#     drawn_line.set_ydata(y_vals)
#     plt.draw()

# # Function to finalize drawing and select edges
# def on_release(event):
#     global line_end, selected_edges
#     if event.xdata is None or event.ydata is None:
#         return
#     line_end = (event.xdata, event.ydata)

#     # Find edges that intersect the drawn line
#     selected_edges = find_edges_under_line(line_start, line_end)
#     print(f"Selected {len(selected_edges)} edges.")
    
#     # Update plot with selected edges
#     plot_data()
#     create_display_edges_on_distance_matrix()


# def plot_data():
#     """Re-draws the scatter plot and highlights selected edges."""
#     ax_ld.clear()

#     # Plot triangulation
#     ax_ld.triplot(low_dm_emb[:, 0], low_dm_emb[:, 1], tri_nodes, color='blue', linewidth=0.8, linestyle='-')

#     # Plot scatter points
#     # Add scatter points for Gaussian clusters
#     # for i in range(n_gauss):
#     #     ax_ld.scatter(low_dm_emb[c == i, 0], low_dm_emb[c == i, 1], color=colors[i],
#     #                         label=f'Class {i}', zorder=3, edgecolor='k', s=50)
    
#     for i in range(n_gauss):
#         ax_ld.scatter(low_dm_emb[orig_label == i, 0], low_dm_emb[orig_label == i, 1], 
#                       color=colors[i], zorder=3, edgecolor='k', s=50)
    
#     ax_ld.imshow(intensity_interp_hd_lengths,
#             extent=(x_min, x_max, y_min, y_max),
#             origin='lower',
#             cmap='hot',
#             alpha=1.0,
#             # interpolation='nearest'
#         )
    
#     for edge in selected_edges:
#         p1, p2 = low_dm_emb[edge[0]], low_dm_emb[edge[1]]
#         ax_ld.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#BFFF00', linewidth=3)

#     # Draw the selected line
#     if line_start and line_end:
#         ax_ld.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], color='cyan', linewidth=2)
    

#     # Remove default y-ticks (cluster numbers)
#     ax_ld.set_yticks([])
#     ax_ld.set_xticks([])
#     ax_ld.axis('equal')
#     plt.axis("off")
#     # ax_ld.legend()

#     output_path = os.path.join(output_folder, f"Delanay_with_selected_line_curve_{dataset}_{method}")
#     plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
#     fig.canvas.draw_idle()

# def create_display_edges_on_distance_matrix():
#     """Highlight selected edges on the pairwise distance matrix."""
#     if len(selected_edges) < 1:
#         print("No edges selected.")
#         return
    
#     print(f"\nDisplaying {len(selected_edges)} selected edges on distance matrix...")
    
#     distance_matrix_hd, _ = inter_intra_cluster_pairwise_distance(
#         D, orig_label, np.unique(orig_label), metric='euclidean', norm_distance=False, norm_type=None
#     )
#     output_path = os.path.join(output_folder, f"Delaunay_selected_line_curve_distance_matrix_{dataset}_{method}")
    
#     plot_highlight_select_edges_on_distance_matrix(
#         distance_matrix_hd, selected_edges, orig_label, np.unique(orig_label),
#         colors=colors, border_thickness=10, output_path=output_path, figsize=(10, 10)
#     )
#     print("Edges plotted on the pairwise distance matrix!")

# # Connect mouse events
# fig.canvas.mpl_connect('button_press_event', on_press)
# fig.canvas.mpl_connect('motion_notify_event', on_motion)
# fig.canvas.mpl_connect('button_release_event', on_release)

# plot_data()
# plt.show()

fig, ax_ld = plt.subplots(figsize=(10, 10))

# Initialize variables
drawing = False
curve_points = []
drawn_line = None
drawn_curves = []
selected_edges = []

# --------------------
# Mouse interaction handlers
# --------------------
def on_press(event):
    global drawing, curve_points, drawn_line
    if event.xdata is None or event.ydata is None:
        return
    drawing = True
    curve_points = [(event.xdata, event.ydata)]
    if drawn_line:
        drawn_line.remove()
    drawn_line, = ax_ld.plot([], [], color='cyan', linewidth=2)

def on_motion(event):
    global curve_points, drawn_line
    if not drawing or event.xdata is None or event.ydata is None:
        return
    curve_points.append((event.xdata, event.ydata))
    xs, ys = zip(*curve_points)
    drawn_line.set_data(xs, ys)
    fig.canvas.draw_idle()

def on_release(event):
    global drawing, selected_edges, drawn_curves
    if not drawing or event.xdata is None or event.ydata is None:
        return
    drawing = False
    curve_points.append((event.xdata, event.ydata))
    drawn_curves.append(curve_points.copy())

    # Find edges intersecting with the curve
    selected_edges = find_edges_under_curve(curve_points)
    print(f"Selected {len(selected_edges)} edges.")

    plot_data()
    create_display_edges_on_distance_matrix()

# --------------------
# Intersection Logic
# --------------------
def find_edges_under_curve(curve):
    """Returns all Delaunay edges intersecting the drawn curve."""
    curve_line = LineString(curve)
    selected = []
    for tri in tri_nodes:
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for edge in edges:
            p1, p2 = low_dm_emb[edge[0]], low_dm_emb[edge[1]]
            edge_line = LineString([p1, p2])
            if curve_line.intersects(edge_line):
                selected.append(edge)
    return selected

# --------------------
# Plotting functions
# --------------------
def plot_data():
    ax_ld.clear()

    # Plot triangulation
    ax_ld.triplot(low_dm_emb[:, 0], low_dm_emb[:, 1], tri_nodes, color='blue', linewidth=0.8)

    # Plot scatter points
    for i in range(n_gauss):
        ax_ld.scatter(low_dm_emb[orig_label == i, 0], low_dm_emb[orig_label == i, 1],
                      color=colors[i], edgecolor='k', s=50, zorder=3)

    # Show heatmap
    ax_ld.imshow(intensity_interp_hd_lengths,
                 extent=(x_min, x_max, y_min, y_max),
                 origin='lower',
                 cmap='hot',
                 alpha=1.0)

    # Plot selected edges
    for edge in selected_edges:
        p1, p2 = low_dm_emb[edge[0]], low_dm_emb[edge[1]]
        ax_ld.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#BFFF00', linewidth=3)

    # Plot drawn curves
    for curve in drawn_curves:
        xs, ys = zip(*curve)
        ax_ld.plot(xs, ys, color='black', linewidth=2)

    ax_ld.set_yticks([])
    ax_ld.set_xticks([])
    ax_ld.axis('equal')
    plt.axis("off")

    output_path = os.path.join(output_folder, f"Delanay_with_selected_curve_{dataset}_{method}")
    plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
    fig.canvas.draw_idle()

def create_display_edges_on_distance_matrix():
    if len(selected_edges) < 1:
        print("No edges selected.")
        return

    print(f"\nDisplaying {len(selected_edges)} selected edges on distance matrix...")

    distance_matrix_hd, _ = inter_intra_cluster_pairwise_distance(
        D, orig_label, np.unique(orig_label), metric='euclidean', norm_distance=False, norm_type=None
    )

    output_path = os.path.join(output_folder, f"Delanay_with_selected_curve_distance_matrix_{dataset}_{method}")
    
    plot_highlight_select_edges_on_distance_matrix(
        distance_matrix_hd, selected_edges, orig_label, np.unique(orig_label),
        colors=colors, border_thickness=10, output_path=output_path, figsize=(10, 10)
    )

    print("Edges plotted on the pairwise distance matrix!")

# --------------------
# Register event handlers and show
# --------------------
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plot_data()
plt.show()
