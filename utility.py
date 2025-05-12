import pickle  # For saving and loading objects
import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
import gc
from inver_project_model import model_train, model_test


# Utility Functions
def save_metrics(metrics, filepath):
    """Save metrics to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(metrics, f)

def load_metrics(filepath):
    """Load metrics from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_binary_dt(data_path):
    with open(data_path, 'rb') as f:  # Open the file in binary mode
        return pickle.load(f, encoding='latin1')  # Use 'latin1' for compatibility


def process_quality_metrics(prj_quality_score, prj_metrics):
    """
    Process and flatten a nested dictionary of projection quality scores 
    into a structured ProjectionMetrics object.

    Parameters:
    - prj_quality_score (dict): A nested dictionary containing quality metrics. 
      Example structure:
        {
            k: {
                metric_category: values,  # values can be scalar or a dictionary (sub-metrics)
                ...
            },
            ...
        }
      - `k` represents different values for metrics (e.g., k-dependent metrics).
      - `metric_category` is the name of the metric or category of sub-metrics.
      - `values` can be a scalar (single value) or a dictionary (sub-metrics).

    - prj_metrics (ProjectionMetrics): An instance of the ProjectionMetrics class 
      to store the processed and flattened metrics.

    Processing Steps:
    1. Iterates through each k-value in `prj_quality_score`.
    2. Checks if a metric category contains sub-metrics (nested dictionary).
        - Flattens sub-metrics and creates combined names (e.g., "category_sub_metric").
        - Adds these sub-metrics to `prj_metrics`.
    3. Adds scalar metrics directly under their respective names.

    Returns:
    - ProjectionMetrics: The updated ProjectionMetrics object with the processed metrics.

    Example:
    Input:
        prj_quality_score = {
            5: {"tnc": 0.85, "nh": {"sub1": 0.9, "sub2": 0.95}},
            10: {"tnc": 0.87, "nh": {"sub1": 0.88, "sub2": 0.91}},
        }
        prj_metrics = ProjectionMetrics()

    Output:
        prj_metrics.metrics = {
            5: {"tnc": [0.85], "nh_sub1": [0.9], "nh_sub2": [0.95]},
            10: {"tnc": [0.87], "nh_sub1": [0.88], "nh_sub2": [0.91]},
        }
    """

    for k, metrics_for_k in prj_quality_score.items():
        for metric_category, values in metrics_for_k.items():
            if isinstance(values, dict):  # Nested dictionary (e.g., sub-metrics)
                for sub_metric, score in values.items():
                    metric_name = f"{metric_category}_{sub_metric}"
                    prj_metrics.add_metric(k, metric_name, score)
            else:
                metric_name = metric_category
                prj_metrics.add_metric(k, metric_name, values)
    return prj_metrics



    
# def jacobian_norm_calculation(S, num_grid_points, inverse_model, input_size, output_size):

#     ## Grid creation for Jacobian estimation
#     x_min, x_max = np.min(S[:, 0]), np.max(S[:, 0])
#     y_min, y_max = np.min(S[:, 1]), np.max(S[:, 1])
#     x_vals = np.linspace(x_min, x_max, num_grid_points)
#     y_vals = np.linspace(y_min, y_max, num_grid_points)
#     xx, yy = np.meshgrid(x_vals, y_vals)
#     grid_points = np.c_[xx.ravel(), yy.ravel()]

#     ## Jacobian estimation
#     jacobian_norms = np.zeros(len(grid_points))
#     for idx, point in enumerate(grid_points):
#         point_tensor = torch.tensor(point, dtype=torch.float32, requires_grad=True).view(1, 2)
#         jacobian = torch.autograd.functional.jacobian(lambda x: inverse_model(x), point_tensor)
#         jacobian_2d = jacobian.view(output_size, input_size)
#         jacobian_norms[idx] = torch.linalg.norm(jacobian_2d, ord=2).item()

#     ########################################
#     jacobian_norms = np.array(jacobian_norms)

#     # Normalize the Jacobian norms for better visualization
#     jacobian_norms = (jacobian_norms - jacobian_norms.min()) / (jacobian_norms.max() - jacobian_norms.min())

#     #######################################
#     jacobian_norms = jacobian_norms.reshape(xx.shape)

#     return jacobian_norms, x_min, x_max, y_min, y_max


### Below is the optimized version of above

import torch
import numpy as np

# def jacobian_norm_calculation(S, num_grid_points, inverse_model, input_size, output_size, batch_size=512, normalize_input=False):
#     # Grid creation for Jacobian estimation
#     x_min, x_max = np.min(S[:, 0]), np.max(S[:, 0])
#     y_min, y_max = np.min(S[:, 1]), np.max(S[:, 1])
#     x_vals = np.linspace(x_min, x_max, num_grid_points)
#     y_vals = np.linspace(y_min, y_max, num_grid_points)
#     xx, yy = np.meshgrid(x_vals, y_vals)
#     grid_points = np.c_[xx.ravel(), yy.ravel()]

#     # Normalize grid points if required
#     if normalize_input:
#         grid_points = (grid_points - np.mean(grid_points, axis=0)) / np.std(grid_points, axis=0)

#     # Placeholder for Jacobian norms
#     jacobian_norms = np.zeros(len(grid_points))

#     # Convert grid points to tensor
#     grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
#     grid_tensor.requires_grad = True

#     # Process in batches
#     for i in range(0, len(grid_tensor), batch_size):
#         batch = grid_tensor[i:i + batch_size]

#         # Forward pass through the model
#         outputs = inverse_model(batch)

#         # Validate output size
#         if outputs.shape[1] != output_size:
#             raise ValueError(f"Model output size mismatch. Expected {output_size}, got {outputs.shape[1]}.")

#         # Compute Jacobian for the batch
#         for j in range(batch.size(0)):
#             # Compute gradient of each output w.r.t input
#             grad_outputs = torch.ones_like(outputs[j])
#             output_grad = torch.autograd.grad(outputs[j], batch, grad_outputs=grad_outputs,
#                                               create_graph=True, retain_graph=True, allow_unused=True)[0]

#             if output_grad is None:
#                 raise RuntimeError("Gradient computation failed.")

#             # Check dimensions
#             if output_grad.shape[-1] != input_size:
#                 raise ValueError(f"Gradient size mismatch. Expected {input_size}, got {output_grad.shape[-1]}.")

#             # Calculate Jacobian norm for the current point
#             jacobian_norms[i + j] = torch.linalg.norm(output_grad, ord=2).item()

#     # Reshape norms to grid shape
#     jacobian_norms = jacobian_norms.reshape(xx.shape)

#     # Normalize Jacobian norms for better visualization
#     jacobian_norms = (jacobian_norms - jacobian_norms.min()) / (jacobian_norms.max() - jacobian_norms.min())

#     return jacobian_norms, x_min, x_max, y_min, y_max


import torch
import numpy as np

def jacobian_norm_calculation(S, num_grid_points, inverse_model, input_size, output_size, batch_size=512, normalize_input=False):
    # Grid creation for Jacobian estimation
    x_min, x_max = np.min(S[:, 0]), np.max(S[:, 0])
    y_min, y_max = np.min(S[:, 1]), np.max(S[:, 1])

    ###_____Margin_______________________________________
    margin = 0.05  # 5% margin (adjustable)
    x_min, x_max = np.min(S[:, 0]), np.max(S[:, 0])
    y_min, y_max = np.min(S[:, 1]), np.max(S[:, 1])

    # Expand the range by a percentage of the range
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range

    #_________________________________________________________

    x_vals = np.linspace(x_min, x_max, num_grid_points)
    y_vals = np.linspace(y_min, y_max, num_grid_points)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # # Normalize grid points if required
    if normalize_input:
        grid_points = (grid_points - np.mean(grid_points, axis=0)) / np.std(grid_points, axis=0)

    # Convert grid points to tensor
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
    grid_tensor.requires_grad = True

    # print("working 1")
    # Placeholder for Jacobian norms
    jacobian_norms = np.zeros(len(grid_points))
    U_matrix = []
    Vt_matrix = []
    sing_values = []
    # breakpoint()
    # Process in batches
    for i in range(0, len(grid_tensor), batch_size):
        batch = grid_tensor[i:i + batch_size]
        # print(f"Processing batch {i // batch_size + 1}/{len(grid_tensor) // batch_size}")

        # Forward pass through the model
        outputs = inverse_model(batch)
        # print("Model forward pass complete")

        # Validate output size
        if outputs.shape[1] != output_size:
            raise ValueError(f"Model output size mismatch. Expected {output_size}, got {outputs.shape[1]}.")
        
        # Initialize a tensor to hold gradients
        grad_outputs = torch.ones_like(outputs)  # Assuming all outputs have the same gradient

        ##__________________New Change______________________________________________________________
        # # Compute Jacobian for the batch
        # output_grads = torch.autograd.grad(outputs, batch, grad_outputs=grad_outputs,
        #                                    create_graph=True, retain_graph=True, allow_unused=False)[0]
        
        jacobian_list = []
        for d in range(outputs.shape[1]):  # Loop over output dimension
            grad_output = torch.zeros_like(outputs)  # Create zero gradient tensor
            grad_output[:, d] = 1  # Set gradient only for the d-th output dimension
            
            # print(f"Computing gradients for output dimension {d}...")
            grad = torch.autograd.grad(outputs, batch, grad_outputs=grad_output,
                                    create_graph=True, retain_graph=True, allow_unused=True)[0]  # (512, 2)
            
            grad = grad.detach()  # Detach tensor from autograd graph
            jacobian_list.append(grad.unsqueeze(1))  # Make it (512, 1, 2) before stacking

        # breakpoint()
        output_grads = torch.cat(jacobian_list, dim=1)  # Now shape is (512, 3, 2)
        # print(f"Batch {i // batch_size + 1} gradient calculation complete")

        ##_________________________________________________________________________________________________
        
        # Check if gradients are successfully computed
        if output_grads is None:
            raise RuntimeError("Gradient computation failed.")

        # Compute the Jacobian norms for each grid point in the batch
        for j in range(batch.size(0)):
            jacobian_norms[i + j] = torch.linalg.norm(output_grads[j], ord=2).item()
            # U_mat, St, Vt = np.linalg.svd(output_grads[j], full_matrices=False)  # Perform SVD

            # breakpoint()
            # Append U_mat to the list
            # U_matrix.append(U_mat)
            # Vt_matrix.append(Vt)
            # sing_values.append(St)
            # print(U.shape, S.shape, Vt.shape)
                
        # print(output_grads.shape)
        # del batch, outputs, output_grads, grad_output, grad, jacobian_list #, U_mat, Vt, St
        
    # gc.collect()

    # Reshape norms to grid shape
    jacobian_norms = jacobian_norms.reshape(xx.shape)
    # breakpoint()

    # Convert list to NumPy array
    # U_matrix = np.array(U_matrix)  # Shape: (total_grid_points, 3, 2)
    # Vt_matrix = np.array(Vt_matrix)  # Shape: (total_grid_points, 2, 2)
    # sing_values = np.array(sing_values)  # Shape: (total_grid_points, 2)


    # Reshape to match grid
    # U_matrices = U_matrix.reshape(num_grid_points,num_grid_points,output_size, input_size)

    # Vt_matrix = Vt_matrix.reshape(num_grid_points,num_grid_points,input_size, input_size)
    # sing_values = sing_values.reshape(num_grid_points,num_grid_points,input_size)

    # Normalize Jacobian norms for better visualization
    jacobian_norms_minmax = (jacobian_norms - jacobian_norms.min()) / (jacobian_norms.max() - jacobian_norms.min())

    X_log = np.log(jacobian_norms)  # log(X + 1) to avooid log(0)
    X_min, X_max = X_log.min(), X_log.max()
    jacobian_norms_log = (X_log - X_min) / (X_max - X_min)

    # breakpoint()
    return jacobian_norms_minmax, jacobian_norms_log, x_min, x_max, y_min, y_max, grid_points,xx, yy
    # return jacobian_norms_minmax, jacobian_norms_log, U_matrices, Vt_matrix, sing_values, x_min, x_max, y_min, y_max, grid_points,xx, yy




def generate_gaussian_clusters(n_gauss=4, n_pts_per_gauss=200, dim=3, centers=None, cluster_spacing=10.0, overlap_factors=None):
    """
    Generate Gaussian clusters with well-separated positioning.
    
    Parameters:
        n_gauss (int): Number of Gaussian clusters.
        n_pts_per_gauss (int): Number of points per cluster.
        dim (int): Dimensionality of the dataset.
        centers (list or None): Custom center positions for each Gaussian.
        cluster_spacing (float): Default spacing between clusters.
        overlap_factors (list or None): List controlling spread of each Gaussian (higher = more spread/overlap).

    Returns:
        D (ndarray): Normalized dataset.
        c (ndarray): Corresponding class labels.
    """
    if centers is None:
        centers = [
            [-cluster_spacing, -cluster_spacing, -cluster_spacing],  # Cluster 1
            [cluster_spacing, cluster_spacing, -cluster_spacing],    # Cluster 2
            [-cluster_spacing, cluster_spacing, cluster_spacing],    # Cluster 3
            [cluster_spacing, -cluster_spacing, cluster_spacing]     # Cluster 4
        ]

    if overlap_factors is None:
        overlap_factors = [0.02, 0.02, 0.02, 0.02]  # Small variance to avoid overlap

    D = np.zeros((n_pts_per_gauss * n_gauss, dim))
    c = np.zeros(n_pts_per_gauss * n_gauss)

    for i in range(n_gauss):
        cov_matrix = np.diag([overlap_factors[i] for _ in range(dim)])
        D[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = np.random.multivariate_normal(
            centers[i], cov_matrix, n_pts_per_gauss
        )
        c[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = i  

    # Normalize dataset
    D = (D - np.min(D, axis=0)) / (np.max(D, axis=0) - np.min(D, axis=0))

    return D, c, centers

def cluster_position(cluster_spacing, mode):

    # if mode == 'cluster_1_far_other_close':
    #     centers = [
    #             [-10 * cluster_spacing, -10 * cluster_spacing, -10 * cluster_spacing],  # Cluster 1 (Far Away)
    #             [cluster_spacing, cluster_spacing, 0],                                 # Cluster 2 (Close to Cluster 3 and 4)
    #             [cluster_spacing + 1, cluster_spacing - 1, 0],                         # Cluster 3 (Close to Cluster 2 and 4)
    #             [cluster_spacing, cluster_spacing - 2, 0]                              # Cluster 4 (Close to Cluster 2 and 3)
    #         ]
    #     overlap_factors = [0.02, 0.02, 0.02, 0.02]

    if mode == 'cluster_1_far_other_close':   ## Cluster 1 is very far away from the other clusters. 
        ###The other three clusters (Clusters 2, 3, and 4) are equidistant from each other but are still equally far from Cluster 1
        # centers = [
        #         [-10 * cluster_spacing, -10 * cluster_spacing, -10 * cluster_spacing],  # Cluster 1 (Far Away)
        #         [cluster_spacing, 0, 0],                                               # Cluster 2
        #         [0, cluster_spacing, 0],                                               # Cluster 3
        #         [0, 0, cluster_spacing]                                                # Cluster 4
        #     ]
        # overlap_factors = [2.5, 2.5, 2.5, 2.5]
        distance_far = 20
        centers = [
                [-cluster_spacing + distance_far, -cluster_spacing + distance_far, -cluster_spacing + distance_far],  # Cluster 1 (Far Away)
                [cluster_spacing, 0, 0],                                               # Cluster 2
                [0, cluster_spacing, 0],                                               # Cluster 3
                [0, 0, cluster_spacing]                                                # Cluster 4
            ]
        over_value = 2.5
        overlap_factors = [over_value, over_value, over_value, over_value]

    # elif mode in ['tetrahedron_eq', 'tetrahedron_close', 'tetrahedron_far', 'tetrahedron_more_far']:
    elif any(mode == m for m in ['tetrahedron_eq_1_far','tetrahedron_eq', 'tetrahedron_eq_1_close', 'tetrahedron_eq_2_close', 'tetrahedron_more_far']):
        centers = np.array([
                [cluster_spacing, cluster_spacing, cluster_spacing],   
                [-cluster_spacing, -cluster_spacing, cluster_spacing],  
                [-cluster_spacing, cluster_spacing, -cluster_spacing],  
                [cluster_spacing, -cluster_spacing, -cluster_spacing]   
            ]) / np.sqrt(3)
        
        overlap_factors = None
     
    elif mode == 'equidistant':   #equidistant_old
        # cluster_spacing = 5* cluster_spacing
        # centers = [
        #         [cluster_spacing, cluster_spacing, cluster_spacing],        # Cluster 1
        #         [-cluster_spacing, cluster_spacing, cluster_spacing],    # Cluster 2
        #         [cluster_spacing, -cluster_spacing, cluster_spacing],    # Cluster 3
        #         [cluster_spacing, cluster_spacing, -cluster_spacing]     # Cluster 4
        #     ]
        # overlap_factors = [0.1, 0.1, 0.1, 0.1]

        centers = [
            [cluster_spacing, cluster_spacing, cluster_spacing],      # Cluster 1
            [-cluster_spacing, -cluster_spacing, cluster_spacing],    # Cluster 2
            [-cluster_spacing, cluster_spacing, -cluster_spacing],    # Cluster 3
            [cluster_spacing, -cluster_spacing, -cluster_spacing]     # Cluster 4
        ]
        over_value = 1.5
        overlap_factors = [over_value, over_value, over_value, over_value]
        
        # cluster_spacing = cluster_spacing           # equidistant_tetrahedron_centers
        # cluster_spacing = 5* cluster_spacing      # equidistant_tetrahedron_centers_more_far
        # centers = [
        #         [0, 0, 0],                    # Cluster 1
        #         [cluster_spacing, 0, 0],                    # Cluster 2
        #         [cluster_spacing / 2, np.sqrt(3) * cluster_spacing / 2, 0],  # Cluster 3
        #         [cluster_spacing / 2, np.sqrt(3) * cluster_spacing / 6, np.sqrt(6) * cluster_spacing / 3]  # Cluster 4
        #     ]
        # overlap_factors = [0.01, 0.01, 0.01, 0.01]       

        
    elif mode == '2_close_pairs':
        # Custom Centers: Clusters 1 & 2 are close, Clusters 3 & 4 are close, but pairs 1-2 and 3-4 are far apart.
        
        # The farthest pairs involve Cluster 1 and Cluster 3 or 4.
        # Cluster 2 and Cluster 3/4 are closer than Cluster 1 and Cluster 3/4
        # centers = [
        #     [-cluster_spacing, -cluster_spacing, 0],  
        #     [-cluster_spacing + 1, -cluster_spacing + 1, 0],  
        #     [cluster_spacing, cluster_spacing, 0],  
        #     [cluster_spacing + 1, cluster_spacing - 1, 0] 
        # ] 
        centers = [
            [-cluster_spacing, -cluster_spacing, 0],  
            [-cluster_spacing + 5, -cluster_spacing - 5, 0],  
            [cluster_spacing, cluster_spacing, 0],  
            [cluster_spacing + 5, cluster_spacing - 5, 0] 
        ] 
        # overlap_factors = [0.02, 0.02, 0.02, 0.02]
        over_value = 2.5
        overlap_factors = [over_value, over_value, over_value, over_value]

    elif mode == '1_close_pairs_1_pair_far':
        # Custom Centers: Clusters 1 & 2 are close, Clusters 3 & 4 are close, but Groups 1-2 and 3-4 are far apart
        # centers = [
        #     [-cluster_spacing, -cluster_spacing, 0],  
        #     [-cluster_spacing + 10, -cluster_spacing + 10, 0],  
        #     [cluster_spacing, cluster_spacing, 0],  
        #     [cluster_spacing + 1, cluster_spacing - 1, 0]  
        # ]
        # overlap_factors = [0.02, 0.02, 0.02, 0.02]
        centers = [
            [-cluster_spacing, -cluster_spacing, 0],  
            [-cluster_spacing + 10, -cluster_spacing + 10, 0],  
            [cluster_spacing - 3, cluster_spacing + 3, 0],  
            [cluster_spacing + 3, cluster_spacing - 3, 0]  
        ]
        over_value = 2
        over_value_2 =0.5
        overlap_factors = [over_value, over_value, over_value, over_value]

    elif mode == '2_10_points_far':
        # Custom Centers: Clusters 1 & 2 are close, Clusters 3 & 4 are close, but Groups 1-2 and 3-4 are far apart
        centers = [
            [-cluster_spacing, -cluster_spacing, 0],  
            [-cluster_spacing + 100, -cluster_spacing + 100, 0],  
            [cluster_spacing, cluster_spacing, 0],  
            [cluster_spacing + 100, cluster_spacing - 100, 0]  
        ]

        # centers = [       ### cluster 1/2 are close and cluster 3/4 are close. Cl 1/2 are far from cl 3/4. cluster 2 is little close to 3 and 4 than 1.
        #     [-cluster_spacing, -cluster_spacing, cluster_spacing],  
        #     [-cluster_spacing + 5, -cluster_spacing + 5, cluster_spacing],  
        #     [cluster_spacing, cluster_spacing, -cluster_spacing],  
        #     [cluster_spacing + 5, cluster_spacing - 5, -cluster_spacing]  
        # ]
        overlap_factors = [0.02, 0.02, 0.02, 0.02]
    
    elif mode == 'non_symmetric':                                                                              #Non-symmetric and unevenly spaced cluster centers simulate real-world scenarios where clusters are irregular. This setup helps test clustering algorithms' adaptability to imbalanced or irregular data.
        centers = [
            [3, 1.5, 0.9],
            [-1.8, -2.5, 2.3],
            [1.2, -2.2, 0.7],
            [-3.0, 2.8, -1.4]
        ]

        # Overlap factors for each cluster
        overlap_factors = [0.03, 0.02, 0.05, 0.04]
    
    elif mode == 'irregular':                                                                              # Mimics real-world Gaussian mixture models, where clusters follow distinct but natural distributions. Ideal for evaluating models like k-means and Gaussian Mixture Models (GMM) 
        centers = [
                [0, 0, 0],    # Dense cluster near origin
                [10, 10, 10], # Farther cluster
                [5, 0, 0],    # Cluster along one axis
                [0, 5, 10]    # Mixed position cluster
        ]
        overlap_factors = [0.5, 5.0, 2.0, 3.0]

    elif mode == 'sparse':
        centers = [
                [10, 0, 0],    
                [0, 10, 0], 
                [0, 0, 10],    
                [0, 0, 10]    
        ]
        overlap_factors = [0.1, 0.1, 0.1, 0.1]
    else:
        raise ValueError(f"Invalid mode '{mode}' provided.")
     
    return centers, overlap_factors

def generate_high_dimension_gaussians(num_dim, n_pts_per_gauss=200, spread_factor=0.1, distance_factor=1.0, distance_factor_2 = 1.0, move_cluster_index=0, random_seed=5):

    if random_seed is None:
        raise ValueError("random_seed cannot be None")

    # breakpoint()
    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)
    

    # breakpoint()
    if num_dim < 1:
        raise ValueError("Number of dimensions must be at least 1.")
    
    centers = np.zeros((4, num_dim))  # Initialize with zeros

    # Second center: All ones except the last dimension is 0
    centers[1, :-1] = 1  # Set all elements except the last one to 1
    
    # Third center: All ones
    centers[2, :] = 1  # Set all elements to 1
    
    # Fourth center: First element is 1, rest are zeros
    centers[3, 0] = 1  # Set first element to 1

    # Move one cluster away dynamically
    moved_tetrahedron = centers.copy()
    moved_tetrahedron[move_cluster_index] *= distance_factor  # Move selected cluster
    moved_tetrahedron[1] *= distance_factor_2  # Move selected cluster

    # Generate Gaussian clusters
    centers = moved_tetrahedron
    cov_matrices = [np.eye(num_dim) * spread_factor for _ in range(4)]

    # Create dataset
    D = np.zeros((n_pts_per_gauss * 4, num_dim))
    c = np.zeros(n_pts_per_gauss * 4)

    for i in range(4):
        # D[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = np.random.multivariate_normal(
        D[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = rng.multivariate_normal(
            centers[i], cov_matrices[i], n_pts_per_gauss
        )
        c[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = i  

    # Normalize dataset
    D = (D - np.min(D, axis=0)) / (np.max(D, axis=0) - np.min(D, axis=0))

    return D, c, centers




def generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=200, base_tetrahedron = None, spread_factor=0.1, distance_factor=1.0, distance_factor_2 = 1.0, move_cluster_index=0):
    """
    Generate well-separated 3D Gaussian clusters inside a tetrahedron with an option 
    to dynamically move one cluster away from others.

    Parameters:
        n_pts_per_gauss (int): Number of points per Gaussian.
        spread_factor (float): Variance of the Gaussian clusters.
        distance_factor (float): Scaling factor to move one cluster dynamically.
        move_cluster_index (int): Index (0-3) of the cluster to be moved.

    Returns:
        D (ndarray): Dataset of points.
        c (ndarray): Corresponding class labels.
        centers (ndarray): Updated centers of Gaussian clusters.
    """
    # Regular Tetrahedron Vertices (inside unit sphere)
    # base_tetrahedron = np.array([
    #     [1, 1, 1],   
    #     [-1, -1, 1],  
    #     [-1, 1, -1],  
    #     [1, -1, -1]   
    # ]) / np.sqrt(3)

    # Move one cluster away dynamically
    moved_tetrahedron = base_tetrahedron.copy()
    moved_tetrahedron[move_cluster_index] *= distance_factor  # Move selected cluster
    moved_tetrahedron[1] *= distance_factor_2  # Move selected cluster

    # Generate Gaussian clusters
    centers = moved_tetrahedron
    cov_matrices = [np.eye(3) * spread_factor for _ in range(4)]

    # Create dataset
    D = np.zeros((n_pts_per_gauss * 4, 3))
    c = np.zeros(n_pts_per_gauss * 4)

    for i in range(4):
        D[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = np.random.multivariate_normal(
            centers[i], cov_matrices[i], n_pts_per_gauss
        )
        c[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = i  

    # Normalize dataset
    D = (D - np.min(D, axis=0)) / (np.max(D, axis=0) - np.min(D, axis=0))

    return D, c, centers



from sklearn.neighbors import KernelDensity
import numpy as np

def kde_cluster_labeling(tsne_embedding, labels, grid_points, bandwidth=0.1):
    unique_labels = np.unique(labels)
    kde_models = {}

    # Train a KDE model for each cluster
    for lbl in unique_labels:
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(tsne_embedding[labels == lbl])
        kde_models[lbl] = kde

    # Compute probability densities for each grid point
    probabilities = np.zeros((grid_points.shape[0], len(unique_labels)))
    for i, lbl in enumerate(unique_labels):
        probabilities[:, i] = np.exp(kde_models[lbl].score_samples(grid_points))

    # Assign each grid point to the cluster with highest probability
    return unique_labels[np.argmax(probabilities, axis=1)]


def inter_intra_cluster_mean_distance(data, label, unique_clusters):
    
    distance_matrix = []
    for i in unique_clusters:
        row_distances = []
        for j in unique_clusters:
            cluster_i_points = data[label == i]
            cluster_j_points = data[label == j]
            if i == j:
                # Intra-cluster distance
                intra_distances = pairwise_distances(cluster_i_points)
                row_distances.append(np.mean(intra_distances))  # Average intra-cluster distance
            else:
                # Inter-cluster distance
                inter_distances = pairwise_distances(cluster_i_points, cluster_j_points)
                row_distances.append(np.mean(inter_distances))  # Average inter-cluster distance
        distance_matrix.append(row_distances)

    distance_matrix = np.array(distance_matrix)
    scaler = MinMaxScaler()
    final_distance_matrix = scaler.fit_transform(distance_matrix)
    return final_distance_matrix

# def inter_intra_cluster_pairwise_distance(data, label, unique_clusters, metric = 'euclidean',  norm_distance = False, norm_type = 'row'):
#     pairwise_distances_matrix = []
#     mean_pairwise_distances_matrix = []

#     for i in unique_clusters:
#         row_distances = []
#         mean_row_distance = []
#         for j in unique_clusters:
#             # Get points in cluster i and cluster j
#             cluster_i_points = data[label == i]
#             cluster_j_points = data[label == j]
            
#             # Compute pairwise distances between all points in cluster i and cluster j
#             inter_cluster_distances = pairwise_distances(cluster_i_points, cluster_j_points, metric=metric)
            
#             # if norm_distance:
#             #     scaler = MinMaxScaler()
#             #     inter_cluster_distances = scaler.fit_transform(inter_cluster_distances)
            
#             row_distances.append(inter_cluster_distances)
#             mean_row_distance.append(np.mean(inter_cluster_distances))

        
#         # Append row distances (list of 2D arrays) to the full matrix
#         pairwise_distances_matrix.append(row_distances)
#         mean_pairwise_distances_matrix.append(mean_row_distance)

#     # Assemble the full matrix into one large 2D matrix
#     final_distance_matrix = np.block(pairwise_distances_matrix)
    
#     mean_pairwise_distances_matrix = np.array(mean_pairwise_distances_matrix)
#     mean_pairwise_distance_block = np.block(mean_pairwise_distances_matrix)
     
#     # Apply normalization if specified
#     if norm_distance:
#         if norm_type == 'global':
#             # Global normalization
#             scaler = MinMaxScaler()
#             final_distance_matrix = scaler.fit_transform(final_distance_matrix)
#             mean_pairwise_distance_block = scaler.fit_transform(mean_pairwise_distance_block)
#         elif norm_type == 'row':
#             # Per-row normalization
#             # Normalize each row of the final pairwise distance matrix
#             final_distance_matrix = np.apply_along_axis(
#                 lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x,
#                 axis=1,
#                 arr=final_distance_matrix
#             )
#             # Normalize each row of the mean pairwise distance matrix
#             mean_pairwise_distance_block = np.apply_along_axis(
#                 lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x,
#                 axis=1,
#                 arr=mean_pairwise_distance_block
#             )

#     return final_distance_matrix, mean_pairwise_distance_block


def inter_intra_cluster_pairwise_distance_old(data, label, unique_clusters, metric='euclidean', norm_distance=False, norm_type='global'):
    pairwise_distances_matrix = []
    mean_pairwise_distances_matrix = []

    for i in unique_clusters:
        row_distances = []
        mean_row_distance = []
        for j in unique_clusters:
            cluster_i_points = data[label == i]
            cluster_j_points = data[label == j]
            inter_cluster_distances = pairwise_distances(cluster_i_points, cluster_j_points, metric=metric)
            row_distances.append(inter_cluster_distances)
            mean_row_distance.append(np.mean(inter_cluster_distances))

        pairwise_distances_matrix.append(row_distances)
        mean_pairwise_distances_matrix.append(mean_row_distance)

    final_distance_matrix = np.block(pairwise_distances_matrix)
    mean_pairwise_distances_matrix = np.array(mean_pairwise_distances_matrix)
    mean_pairwise_distance_block= np.block(mean_pairwise_distances_matrix)

    # Normalization if required
    if norm_distance:
        min_value = np.min(full_pairwise_distance_matrix)
        max_value = np.max(full_pairwise_distance_matrix)
        full_pairwise_distance_matrix = (full_pairwise_distance_matrix - min_value) / (max_value - min_value)

        min_value_mean = np.min(mean_pairwise_distances_matrix)
        max_value_mean = np.max(mean_pairwise_distances_matrix)
        mean_pairwise_distances_matrix = (mean_pairwise_distances_matrix - min_value_mean) / (max_value_mean - min_value_mean)
        
    
    mean_pairwise_distance_block

    return final_distance_matrix, mean_pairwise_distance_block

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

def inter_intra_cluster_pairwise_distance(data, labels, unique_clusters, metric='euclidean', norm_distance=False, norm_type='global'):
    
    num_clusters = len(unique_clusters)
    unique_clusters = np.sort(unique_clusters)

    # Initialize storage
    pairwise_distances_dict = {}  
    mean_pairwise_distances_matrix = np.zeros((num_clusters, num_clusters))

    # breakpoint()
    for i, cluster_i in enumerate(unique_clusters):
        cluster_i_points = data[labels == cluster_i]

        for j, cluster_j in enumerate(unique_clusters):
            if j < i:  
                continue  # Skip lower triangle, will be filled from (i, j)

            cluster_j_points = data[labels == cluster_j]

            # Compute pairwise distances
            inter_cluster_distances = pairwise_distances(cluster_i_points, cluster_j_points, metric=metric)
            # inter_cluster_distances_2 = pairwise_distances(cluster_j_points, cluster_i_points, metric=metric)
            # diff = inter_cluster_distances - inter_cluster_distances_2
            # breakpoint()
            # Store distance matrix in both directions
            pairwise_distances_dict[(i, j)] = inter_cluster_distances
            if i != j:  
                pairwise_distances_dict[(j, i)] = inter_cluster_distances.T  # True symmetry

            # Store mean distances
            mean_distance = np.mean(inter_cluster_distances)
            mean_pairwise_distances_matrix[i, j] = mean_distance
            if i != j:  
                mean_pairwise_distances_matrix[j, i] = mean_distance  # True symmetry

    # breakpoint()
    # Convert to full pairwise matrix
    row_blocks = []
    for i in range(num_clusters):
        row = [pairwise_distances_dict[(i, j)] for j in range(num_clusters)]
        row_blocks.append(np.hstack(row))  

    full_pairwise_distance_matrix = np.vstack(row_blocks)
    # breakpoint()
    # Normalization if required
    if norm_distance:
        min_value = np.min(full_pairwise_distance_matrix)
        max_value = np.max(full_pairwise_distance_matrix)
        full_pairwise_distance_matrix = (full_pairwise_distance_matrix - min_value) / (max_value - min_value)

        min_value_mean = np.min(mean_pairwise_distances_matrix)
        max_value_mean = np.max(mean_pairwise_distances_matrix)
        mean_pairwise_distances_matrix = (mean_pairwise_distances_matrix - min_value_mean) / (max_value_mean - min_value_mean)
    # if norm_distance:
    #     min_value = np.min(full_pairwise_distance_matrix)
    #     max_value = np.max(full_pairwise_distance_matrix)
    #     min_value_mean = np.min(mean_pairwise_distances_matrix)
    #     max_value_mean = np.max(mean_pairwise_distances_matrix)

    #     full_pairwise_distance_matrix = (full_pairwise_distance_matrix - min_value_mean) / (max_value_mean - min_value_mean)
    #     mean_pairwise_distances_matrix = (mean_pairwise_distances_matrix - min_value_mean) / (max_value_mean - min_value_mean)
    # breakpoint()
    return full_pairwise_distance_matrix, mean_pairwise_distances_matrix


##_____________Compactness ratio_________________________
def compute_relative_compactness(distance_matrix_3D, distance_matrix_2D, unique_clusters):
    """Computes how each cluster's relative distances change in 2D vs 3D."""
    
    # Compute relative distance ratios (D_2D / D_3D) for each pair of clusters
    relative_distance_ratios = distance_matrix_2D / (distance_matrix_3D + 1e-8)  # Avoid divide by zero

    # breakpoint()
    
    # Compute the compactness ratio for each cluster
    compactness_ratios = np.mean(relative_distance_ratios, axis=1)

    cluster_ratios = {}
    for i, cluster in enumerate(unique_clusters):
        cluster_ratios[cluster] = compactness_ratios[i]
        print(f"Cluster {cluster}: Relative Compactness Ratio = {compactness_ratios[i]:.4f} "
              f"({'More Compact' if compactness_ratios[i] < 1 else 'More Stretched'})")

    return cluster_ratios

def compute_difference_3D_to_2D(mean_distance_3D, mean_distance_2D):
    """Computes how much t-SNE stretches or compresses each cluster."""
    
    # # Compute the ratio (2D distance / 3D distance)
    # relative_distance_ratios = mean_distance_2D / (mean_distance_3D + 1e-8)  # Avoid division by zero
    
    # # Compute compactness ratio for each cluster
    # compactness_ratios = np.mean(relative_distance_ratios, axis=1)  # Mean across rows

    # cluster_ratios = {}
    # for i, cluster in enumerate(unique_clusters):
    #     cluster_ratios[cluster] = compactness_ratios[i]
    #     print(f"Cluster {cluster}: Compactness Ratio = {compactness_ratios[i]:.4f} "
    #           f"({'More Compact' if compactness_ratios[i] < 1 else 'More Stretched'})")

    # Compute relative change in percentage
    epsilon = 1e-8  # Small value to avoid division by zero

    # relative_change_inter = (mean_distance_2D - mean_distance_3D) / (mean_distance_3D + epsilon) * 100
    abs_diff = np.abs(mean_distance_3D - mean_distance_2D)

    # breakpoint()

    return abs_diff

def extract_off_diagonal(matrix):
    # Create the dictionary to hold the distances
    pairwise_dict = {}
    num_clusters = matrix.shape[0]

    for i in range(num_clusters):
        # Initialize the sub-dictionary for cluster i
        cluster_i_dict = {}
        for j in range(num_clusters):
            if i != j:  # Exclude diagonal (i == j)
                cluster_i_dict[j+1] = matrix[i, j]  # Store pairwise distance (1-based index)

        pairwise_dict[i+1] = cluster_i_dict  # Store sub-dictionary under cluster key

    return pairwise_dict

def normalize_and_sort_pairwise_dict(pairwise_dict):
    # Iterate through the pairwise dictionary
    for cluster, distances in pairwise_dict.items():
        row_sum = sum(distances.values())  # Sum of the distances for the current cluster

        # Normalize each distance by dividing by the row sum
        if row_sum > 0:  # Avoid division by zero
            for key in distances:
                distances[key] /= row_sum  # Normalize each distance
        
        # Sort the distances in ascending order (optional, can change to reverse=True for descending)
        pairwise_dict[cluster] = dict(sorted(distances.items(), key=lambda item: item[1]))

    return pairwise_dict

def add_nested_dicts(dict1, dict2):
    result = {}
    for key in set(dict1.keys()).union(dict2.keys()):  # Ensure all primary keys are covered
        result[key] = {}
        subkeys = set(dict1.get(key, {}).keys()).union(dict2.get(key, {}).keys())  # Ensure all subkeys are covered
        for subkey in subkeys:
            val1 = dict1.get(key, {}).get(subkey, 0)  # Default to 0 if key is missing
            val2 = dict2.get(key, {}).get(subkey, 0)  # Default to 0 if key is missing
            result[key][subkey] = np.float64(val1) + np.float64(val2)  # Ensure np.float64 addition
    return result


##____________________________________________________________
# def hybrid_normalization(X, epsilon_factor=0.001, min_threshold=0.01):
#     """
#     Applies Hybrid Normalization with a more robust handling for small value ranges.

#     Parameters:
#     - X: The data to be normalized
#     - epsilon_factor: Factor used to calculate epsilon based on the standard deviation
#     - min_threshold: Minimum threshold for epsilon. If std_dev is smaller than this, no epsilon is applied.
#     """
#     # Calculate standard deviation
#     std_dev = np.std(X)
    
#     # If the standard deviation is too small, avoid applying epsilon
#     epsilon = max(epsilon_factor * std_dev, min_threshold)

#     # Min-Max normalization
#     X_min = np.min(X)
#     X_max = np.max(X)

#     # Ensure there's no division by zero in case X_max == X_min
#     if X_max == X_min:
#         return np.zeros_like(X)  # Return all zeros if all values are identical

#     # Apply Hybrid Normalization
#     X_normalized = (X - X_min) / (X_max - X_min + epsilon)
    
#     # Clip the values to make sure they stay within [0, 1]
#     X_normalized = np.clip(X_normalized, 0, 1)
    
#     return X_normalized

# def hybrid_normalization(data):  #z_score_normalization
#     mean_val = np.mean(data)
#     std_val = np.std(data)
#     return (data - mean_val) / std_val
def hybrid_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)
def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')



# def hybrid_normalization(data, window_size=3):  #combined_normalization
   
#     normalized_data = min_max_normalization(data)
    
#     # Apply smoothing if necessary
#     smoothed_data = moving_average(normalized_data, window_size)
    
#     return smoothed_data



import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import random_walker

def segment_grid_by_spectral_norm(spectral_norm_values, beta=10, threshold=None):
    """
    Segment the 2D grid using a random walker approach based on spectral norm values.
    
    Parameters:
    - spectral_norm_values (2D numpy array): Spectral norm values at each grid point.
    - beta (float): The smoothing parameter for random walker (higher values = smoother segmentation).
    - threshold (float, optional): Threshold to determine boundary vs. non-boundary regions.
    
    Returns:
    - labels (2D numpy array): Segmented regions (1 = cluster, 2 = boundary)
    """
    grid_shape = spectral_norm_values.shape

    # Define threshold automatically if not provided
    if threshold is None:
        threshold = np.percentile(spectral_norm_values, 80)  # Top 20% as boundary

    # Create label matrix: 
    # 1 → Low spectral norm (inside clusters)
    # 2 → High spectral norm (decision boundary)
    markers = np.zeros(grid_shape, dtype=int)
    markers[spectral_norm_values <= threshold] = 1  # Cluster regions
    markers[spectral_norm_values > threshold] = 2   # Boundary regions

    # Apply the Random Walker algorithm
    labels = random_walker(spectral_norm_values, markers, beta=beta, mode='bf')

    return labels

# Function to collect seed points interactively
def select_seeds(image, num_seeds, x_min, x_max, y_min, y_max, output_path = None):
    plt.figure(figsize=(8, 6))
    # plt.imshow(image, cmap='tab10', interpolation='nearest')

    ax = plt.gca()
    # Plot the Jacobian norm heatmap
    ax.imshow(
        image,
        extent=(x_min, x_max, y_min, y_max),
        origin='upper',
        cmap='seismic',
        alpha=1.0
    )
    # plt.colorbar(label="Spectral Norm")
    plt.title("Click to Select Seeds (Close window when done)")
    
    # Get user input (clicks on the image)
    clicks = plt.ginput(n=num_seeds, timeout=0)  # Wait until all seeds are selected

    # Convert clicked points to NumPy array
    seeds = np.array(clicks)

    # Plot selected points on the heatmap
    if seeds.size > 0:
        plt.scatter(seeds[:, 0], seeds[:, 1], c='blue', marker='o', label="Selected Seeds")
        plt.legend()

    # Save the plot only if a path is specified
    if output_path:
        plt.savefig(f"{output_path}.png", dpi=300, format='png', bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()

    # Convert clicked points to indices
    seeds = np.array(clicks).astype(int)
    return seeds

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import random_walker
from skimage.morphology import remove_small_objects

def segment_with_random_walker(norm_values, num_seeds=10, threshold=0.8, beta=10, min_size=5):
    """
    Segments a 2D space using the Random Walker algorithm based on Jacobian norm values.

    Parameters:
    - norm_values: 2D NumPy array of Jacobian norm values.
    - num_seeds: Number of random seed points inside the non-boundary region.
    - threshold: Threshold to define decision boundaries (high norm values).
    - beta: Random Walker smoothing parameter (higher values -> smoother segmentation).
    - min_size: Minimum segment size to retain (removes small noisy regions).

    Returns:
    - segmented_grid: 2D NumPy array with segment labels.
    """
    grid_shape = norm_values.shape

    # Identify boundary regions (high-norm areas)
    boundary_mask = norm_values > threshold  

    # Initialize label grid: 0 = unlabeled, 1-N = segment labels
    labels = np.zeros(grid_shape, dtype=int)

    # Find valid (non-boundary) regions
    valid_points = np.argwhere(~boundary_mask)

    # Ensure reproducibility and select seed points
    np.random.seed(5)
    num_seeds = min(num_seeds, len(valid_points))  # Adjust if not enough points
    selected_seeds = valid_points[np.random.choice(len(valid_points), num_seeds, replace=False)]

    # Assign unique labels to seed points
    for i, (x, y) in enumerate(selected_seeds):
        labels[x, y] = i + 1  

    # Handle the case where no valid seed points exist
    if num_seeds == 0:
        print("Warning: No valid seed points found. Returning empty segmentation.")
        return np.zeros(grid_shape, dtype=int)

    # Perform random walker segmentation
    segmented_grid = random_walker(norm_values, labels, beta=beta, mode='bf')

    # Remove small noisy segments
    segmented_grid = remove_small_objects(segmented_grid, min_size=min_size)

    # Restore boundaries using the original mask
    segmented_grid[boundary_mask] = -1

    return segmented_grid



####_____________________Delanay______________________________________________

def calculate_delanay_edge_length(hd_data, ld_data, tri_nodes):
    all_edges_length_ld = []
    all_edges_length_hd = []

    for simplex in tri_nodes:
        # Get the coordinates of the three triangle points
        pt_v1_LD, pt_v2_LD, pt_v3_LD = ld_data[simplex]  # Shape (3, 2)
        pt_v1_HD, pt_v2_HD, pt_v3_HD = hd_data[simplex]  # Shape (3, feature_dim)

        # breakpoint()
        
        # Compute Euclidean distances between the three edges in Low Dimension
        edge_lengths_LD = [
            np.linalg.norm(pt_v1_LD - pt_v2_LD),
            np.linalg.norm(pt_v2_LD - pt_v3_LD),
            np.linalg.norm(pt_v3_LD - pt_v1_LD)
        ]

        # breakpoint()
        # Compute Euclidean distances between the three edges in High Dimension
        edge_lengths_HD = [
            np.linalg.norm(pt_v1_HD - pt_v2_HD),
            np.linalg.norm(pt_v2_HD - pt_v3_HD),
            np.linalg.norm(pt_v3_HD - pt_v1_HD)
        ]

        all_edges_length_ld.append(edge_lengths_LD)
        all_edges_length_hd.append(edge_lengths_HD)

    return np.array(all_edges_length_hd), np.array(all_edges_length_ld)

def triangle_area_nd(p1, p2, p3):
    """
    Compute the area of a triangle given three points in n-dimensional space.
    
    Parameters:
    p1, p2, p3 : array-like, shape (n,)
        Coordinates of the three vertices of the triangle in n-dimensional space.
    
    Returns:
    float : The area of the triangle.
    """
    # Convert to numpy arrays
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    
    # Compute edge vectors
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Compute cross product (only valid for 2D and 3D)
    if len(p1) == 2:
        # In 2D, cross product is scalar (signed area), so take absolute value
        cross_product = np.cross(v1, v2)
        area = 0.5 * np.abs(cross_product)
        # print('low_dim')
        # breakpoint()
    elif len(p1) == 3:
        # In 3D, cross product is a vector, so take its magnitude
        cross_product = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(cross_product)
        # print('3D dim')
        # breakpoint()
    else:
        # For n-dimensional space, use determinant of the Gram matrix
        M = np.vstack([v1, v2])  # Create a matrix with two row vectors
        area = 0.5 * np.sqrt(np.linalg.det(M @ M.T))  # Generalized cross product norm
        # print('greater than 3 dim')
    
    return area

def calculate_area_traingle_hd_ld(hd_data, ld_data, tri_nodes):
    all_tri_area_ld = []
    all_tri_area_hd = []

    for simplex in tri_nodes:
        # Get the coordinates of the three triangle points
        pt_a_LD, pt_b_LD, pt_c_LD = ld_data[simplex]  # Shape (3, 2)
        pt_a_HD, pt_b_HD, pt_c_HD = hd_data[simplex]  # Shape (3, feature_dim)

        tri_area_ld = triangle_area_nd(pt_a_LD, pt_b_LD, pt_c_LD)
        # breakpoint()
        tri_area_hd = triangle_area_nd(pt_a_HD, pt_b_HD, pt_c_HD)
        all_tri_area_ld.append(tri_area_ld)
        all_tri_area_hd.append(tri_area_hd)
        # breakpoint()
    
    return np.array(all_tri_area_hd), np.array(all_tri_area_ld)


def check_triangle_data_integrity(triang_t_sne = None, all_triangle_area_hd= None, all_triangle_area_ld= None):
    """
    Ensures that the given triangle-related arrays match the number of triangles 
    in the Delaunay triangulation and do not contain NaN values.

    Parameters:
        triang_t_sne: A Delaunay triangulation object (must have a `.triangles` attribute).
        all_tri_max_edges_ld (array): Array of maximum edge lengths in the low-dimensional space.
        all_triangle_area_hd (array): Array of triangle areas in the high-dimensional space.
        all_triangle_area_ld (array): Array of triangle areas in the low-dimensional space.

    Raises:
        ValueError: If any array length does not match the number of triangles.
    """
    num_triangles = len(triang_t_sne.triangles)

    # # Check all_tri_max_edges_ld
    # if len(all_tri_max_edges_ld) != num_triangles:
    #     raise ValueError(f"The length of 'all_tri_max_edges_ld' ({len(all_tri_max_edges_ld)}) "
    #                      f"does not match the number of triangles ({num_triangles})")
    # if np.any(np.isnan(all_tri_max_edges_ld)):
    #     print("NaN values detected in 'all_tri_max_edges_ld'")

    # Check all_triangle_area_hd
    if len(all_triangle_area_hd) != num_triangles:
        raise ValueError(f"The length of 'all_triangle_area_hd' ({len(all_triangle_area_hd)}) "
                         f"does not match the number of triangles ({num_triangles})")
    if np.any(np.isnan(all_triangle_area_hd)):
        print("NaN values detected in 'all_triangle_area_hd'")

    # Check all_triangle_area_ld
    if len(all_triangle_area_ld) != num_triangles:
        raise ValueError(f"The length of 'all_triangle_area_ld' ({len(all_triangle_area_ld)}) "
                         f"does not match the number of triangles ({num_triangles})")
    if np.any(np.isnan(all_triangle_area_ld)):
        print("NaN values detected in 'all_triangle_area_ld'")

    print("All triangle data arrays are correctly formatted ✅")




##_________________ Interpolation_______________________________________

def get_triangle_vertices(point, tri):
    """
    Check if a point is inside a triangle and return its vertices.
    
    Parameters:
    - point: (x, y) coordinates of the point.
    - tri: Delaunay triangulation object.

    Returns:
    - vertices: List of three vertices if inside a triangle, else None.
    """
    simplex_index = tri.find_simplex(point)
    
    if simplex_index >= 0:  # The point is inside a triangle
        triangle_indices = tri.simplices[simplex_index]  # Indices of triangle vertices
        vertices = tri.points[triangle_indices]  # Get the actual (x, y) coordinates
        return vertices
    else:
        return None  # Point is outside the triangulation


def create_matrix_determinant(V1, V2, P):
    """
    Constructs a 3x3 matrix using the given two vertices and a point.

    Parameters:
    - V1: Tuple (x1, y1) representing the first vertex.
    - V2: Tuple (x2, y2) representing the second vertex.
    - P: Tuple (xp, yp) representing the point.

    Returns:
    - A: 3x3 NumPy matrix with the specified elements.
    """
    A = np.array([
        [V1[0], V2[0], P[0]],  # First row (x-coordinates)
        [V1[1], V2[1], P[1]],  # Second row (y-coordinates)
        [1,     1,     1]      # Third row (all ones)
    ])

    det_A = np.linalg.det(A)/2
    return A, det_A

def barycentric_interpolation(num_grid_points_inter, embedding, tri_delaunay, relative_edge_ratio, blog, bclamping = 'False'):
    x_min, x_max = np.min(embedding[:, 0]), np.max(embedding[:, 0])
    y_min, y_max = np.min(embedding[:, 1]), np.max(embedding[:, 1])
    x_vals = np.linspace(x_min, x_max, num_grid_points_inter)
    y_vals = np.linspace(y_min, y_max, num_grid_points_inter)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Assign colors
    intensity_interp = []
    for point in grid_points:
        simplex_index = tri_delaunay.find_simplex(point)
        # breakpoint()
        if  simplex_index >= 0:  # If inside a triangle
            
            triangle_indices = tri_delaunay.simplices[simplex_index]  # Indices of triangle vertices

            # breakpoint()
            vertices = tri_delaunay.points[triangle_indices]  # Get the actual (x, y) coordinates
            v1 = vertices[0, :]
            v2 = vertices[1, :]
            v3 = vertices[2, :]

            a_ratio = relative_edge_ratio[simplex_index, 0]
            b_ratio = relative_edge_ratio[simplex_index, 1]
            c_ratio = relative_edge_ratio[simplex_index, 2]


            # breakpoint()
            A1, det_A1 = create_matrix_determinant(v2, v3, point)
            A2, det_A2 = create_matrix_determinant(v3, v1, point)
            A3, det_A3 = create_matrix_determinant(v1, v2, point)

            A = det_A1 + det_A2 + det_A3
            # f_x = (((det_A1 + det_A2)*a_ratio + (det_A2 + det_A3)*b_ratio + (det_A1 + det_A3)*c_ratio))* (1/(2*A))

            # f_x = ((a_ratio + c_ratio) * det_A1 + (a_ratio + b_ratio) * det_A2 + (b_ratio + c_ratio) * det_A3)* (1/(2*A))

            # f_x = ((A - det_A3) * a_ratio + (A - det_A1)* b_ratio + (A- det_A2) * c_ratio) * (1/(2* A))

            ###################

            h0 = 1 - (det_A1/A) - (det_A2/A)
            h1 = 1 - (det_A2/A) - (det_A3/A)
            h2 = 1 - (det_A3/A) - (det_A1/A)

            e_0 = (det_A1/A) * h2/(h2 + h0) +  (det_A2/A) * h1/(h1 + h0)
            e_1 = (det_A2/A) * h0/(h0 + h1) +  (det_A3/A) * h2/(h2 + h1)
            e_2 = (det_A3/A) * h1/(h1 + h2) +  (det_A1/A) * h0/(h0 + h2)

            f_x = a_ratio * e_0 + b_ratio * e_1 + c_ratio * e_2
            # breakpoint()
            ####################


            intensity_interp.append(f_x)
        else:  # If grid outside the denaulay triangle
            intensity_interp.append(-1)  # Use 0 to indicate black color

    # Convert colors to a NumPy array
    intensity_interp = np.array(intensity_interp)
    # breakpoint()

    # # Create a mask
    mask_not_valid = intensity_interp == -1   

    # ___Checks_________________________
    np.any((intensity_interp < 0) & (intensity_interp != -1))   ##check negative value exists other than -1


    # # Set outside-triangle points to NaN
    intensity_interp[mask_not_valid] = np.nan  
    # breakpoint()
    # # Normalize only valid (non-NaN) values
    if np.any(~mask_not_valid):  
        min_val = np.nanmin(intensity_interp)  # if we not use simple np.min(), max() it will give all nan.
        max_val = np.nanmax(intensity_interp)

        if max_val > min_val:  # Avoid division by zero
            if blog:
                intensity_interp[~mask_not_valid] = (np.log(intensity_interp[~mask_not_valid]) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))
            else:
                intensity_interp[~mask_not_valid] = (intensity_interp[~mask_not_valid] - min_val) / (max_val - min_val)
        else:
            print("Warning: No variation in values, skipping normalization.")
        
        if bclamping:
            # Compute the 5th and 95th percentiles
            lower_bound = np.percentile(intensity_interp[~mask_not_valid], 1)   # Bottom 5% threshold

            upper_bound = np.percentile(intensity_interp[~mask_not_valid], 99)  # Top 5% threshold

            upper_bound = 1

            intensity_interp[~mask_not_valid] = np.clip(intensity_interp[~mask_not_valid], lower_bound, upper_bound)  # Clamping
            intensity_interp[~mask_not_valid] = (intensity_interp[~mask_not_valid] - lower_bound) / (upper_bound - lower_bound)

    # breakpoint()
    # # Reshape for plotting
    intensity_interp_reshape = intensity_interp.reshape(xx.shape)
    # breakpoint()
    return intensity_interp_reshape, x_min, x_max, y_min, y_max

def barycentric_coords(P, A, B, C):
    v0 = B - A
    v1 = C - A
    v2 = P - A

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    return u, v, w
# def barycentric_coords_batch(P, A, B, C):
#     v0 = B - A
#     v1 = C - A
#     v2 = P - A

#     d00 = np.dot(v0, v0)
#     d01 = np.dot(v0, v1)
#     d11 = np.dot(v1, v1)
#     d20 = np.dot(v2, v0)
#     d21 = np.dot(v2, v1)

#     denom = d00 * d11 - d01 * d01
#     v = (d11 * d20 - d01 * d21) / denom
#     w = (d00 * d21 - d01 * d20) / denom
#     u = 1.0 - v - w

#     return u, v, w

def barycentric_coords_batch(points, v1, v2, v3):
    v0 = v2 - v1  # Edge v1→v2
    v1v = v3 - v1 # Edge v1→v3
    v2p = points - v1   # Vector from v1 to point

    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1v)
    d11 = np.einsum('ij,ij->i', v1v, v1v)
    d20 = np.einsum('ij,ij->i', v2p, v0)
    d21 = np.einsum('ij,ij->i', v2p, v1v)

    denom = d00 * d11 - d01 * d01
    
    epsilon = 1e-8
    denom_safe = np.where(np.abs(denom) < epsilon, epsilon, denom)
    denom = denom_safe

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom

    u = 1 - v - w

    # # mask invalid barycentric coords
    # u = np.where(np.abs(denom) < epsilon, np.nan, u)
    # v = np.where(np.abs(denom) < epsilon, np.nan, v)
    # w = np.where(np.abs(denom) < epsilon, np.nan, w)

    return u, v, w
def barycentric_interpolation_coordiantes(num_grid_points_inter, orig_data, embedding, tri_delaunay, model, blog, bclamping = 'False'):
    x_min, x_max = np.min(embedding[:, 0]), np.max(embedding[:, 0])
    y_min, y_max = np.min(embedding[:, 1]), np.max(embedding[:, 1])
    x_vals = np.linspace(x_min, x_max, num_grid_points_inter)
    y_vals = np.linspace(y_min, y_max, num_grid_points_inter)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Assign colors
    intensity_interp = []
    for point in grid_points:
        simplex_index = tri_delaunay.find_simplex(point)
        # breakpoint()
        if  simplex_index >= 0:  # If inside a triangle
            
            triangle_indices = tri_delaunay.simplices[simplex_index]  # Indices of triangle vertices

            # breakpoint()
            vertices = tri_delaunay.points[triangle_indices]  # Get the actual (x, y) coordinates
            v1 = vertices[0, :]
            v2 = vertices[1, :]
            v3 = vertices[2, :]

            vertices_orig = orig_data[triangle_indices]

            v1_high_dim = vertices_orig[0, :]
            v2_high_dim = vertices_orig[1, :]
            v3_high_dim = vertices_orig[2, :]

            lambda_1, lambda_2, lambda_3 = barycentric_coords(point, v1, v2, v3)
            sum_cor = lambda_1 + lambda_2 + lambda_3

            ## Reconstruct point in HD
            point_hd = lambda_1 * v1_high_dim + lambda_2 * v2_high_dim + lambda_3 * v3_high_dim

            
            point = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
            # breakpoint()
            # y_test = np.array([])
            
            # point_inv, _ = model_test(X_test = point, model = model, bLossFlag = False)
            point_inv = model(point)
            f_x = np.linalg.norm(point_hd - point_inv.detach().numpy())
            # breakpoint()

            # breakpoint()
            ####################

            # print('inverse complete')
            intensity_interp.append(f_x)
        else:  # If grid outside the denaulay triangle
            intensity_interp.append(-1)  # Use 0 to indicate black color

    print('Convert Each point in traingle into  color array')
    # Convert colors to a NumPy array
    intensity_interp = np.array(intensity_interp)
    # breakpoint()

    # # Create a mask
    mask_not_valid = intensity_interp == -1   

    # ___Checks_________________________
    np.any((intensity_interp < 0) & (intensity_interp != -1))   ##check negative value exists other than -1


    # # Set outside-triangle points to NaN
    intensity_interp[mask_not_valid] = np.nan  
    # breakpoint()
    # # Normalize only valid (non-NaN) values
    if np.any(~mask_not_valid):  
        min_val = np.nanmin(intensity_interp)  # if we not use simple np.min(), max() it will give all nan.
        max_val = np.nanmax(intensity_interp)

        if max_val > min_val:  # Avoid division by zero
            if blog:
                intensity_interp[~mask_not_valid] = (np.log(intensity_interp[~mask_not_valid]) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))
            else:
                intensity_interp[~mask_not_valid] = (intensity_interp[~mask_not_valid] - min_val) / (max_val - min_val)
        else:
            print("Warning: No variation in values, skipping normalization.")
        
        if bclamping:
            # Compute the 5th and 95th percentiles
            lower_bound = np.percentile(intensity_interp[~mask_not_valid], 1)   # Bottom 5% threshold

            upper_bound = np.percentile(intensity_interp[~mask_not_valid], 99)  # Top 5% threshold

            upper_bound = 1

            intensity_interp[~mask_not_valid] = np.clip(intensity_interp[~mask_not_valid], lower_bound, upper_bound)  # Clamping
            intensity_interp[~mask_not_valid] = (intensity_interp[~mask_not_valid] - lower_bound) / (upper_bound - lower_bound)

    # breakpoint()
    # # Reshape for plotting
    intensity_interp_reshape = intensity_interp.reshape(xx.shape)
    # breakpoint()
    return intensity_interp_reshape, x_min, x_max, y_min, y_max

def barycentric_interpolation_coordiantes_batch(num_grid_points_inter, orig_data, embedding, tri_delaunay, model, blog, bclamping = 'False'):
    x_min, x_max = np.min(embedding[:, 0]), np.max(embedding[:, 0])
    y_min, y_max = np.min(embedding[:, 1]), np.max(embedding[:, 1])
    x_vals = np.linspace(x_min, x_max, num_grid_points_inter)
    y_vals = np.linspace(y_min, y_max, num_grid_points_inter)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Step 2: Initialize all to -1 (default for outside triangle)
    intensity_interp = np.full(len(grid_points), fill_value=-1.0, dtype=np.float32)

    # Vectorized simplex detection
    simplex_indices = tri_delaunay.find_simplex(grid_points)
    valid_mask = simplex_indices >= 0
    valid_points = grid_points[valid_mask]
    valid_simplices = simplex_indices[valid_mask]

    # Get triangle vertices for valid simplices
    tri_vertices = tri_delaunay.simplices[valid_simplices]  # (N, 3)
    v1_low_dim = embedding[tri_vertices[:, 0]]
    v2_low_dim = embedding[tri_vertices[:, 1]]
    v3_low_dim = embedding[tri_vertices[:, 2]]

    # breakpoint()
    print('barycentric coordinates calculation ...')
    lambda_1, lambda_2, lambda_3 = barycentric_coords_batch(valid_points, v1_low_dim, v2_low_dim, v3_low_dim)
    sum_cor = lambda_1 + lambda_2 + lambda_3

    # breakpoint()
    # Projected high-dim from barycentric weights
    v1_hd = orig_data[tri_vertices[:, 0]]
    v2_hd = orig_data[tri_vertices[:, 1]]
    v3_hd = orig_data[tri_vertices[:, 2]]

    # reconstructed_hd = lambda_1[:, None] * v1_hd + lambda_2[:, None] * v2_hd + lambda_3[:, None] * v3_hd

    lambdas = np.stack([lambda_1, lambda_2, lambda_3], axis=1)  # (N, 3)
    vertices_hd = np.stack([v1_hd, v2_hd, v3_hd], axis=1)          # (N, 3, D)
    # breakpoint()
    reconstructed_hd = np.einsum('ni,nid->nd', lambdas, vertices_hd)

    # reconstructed_hd = np.einsum('i,ij->ij', lambda_1, v1_hd) + \
    #                np.einsum('i,ij->ij', lambda_2, v2_hd) + \
    #                np.einsum('i,ij->ij', lambda_3, v3_hd)
    # breakpoint()
    # Predict using inverse model (batch)
    model.eval()

    print('model inference')
    with torch.no_grad():
        predicted_hd = model(torch.tensor(valid_points, dtype=torch.float32)).numpy()
    
    # breakpoint()
    print('intensity_per_point calculation ...')
    # Compute norm
    intensity_per_point = np.linalg.norm(reconstructed_hd - predicted_hd, axis=1)

    intensity_interp[valid_mask] = intensity_per_point
    # breakpoint()
    # # Assign colors
    # intensity_interp = []
    # for point in grid_points:
    #     simplex_index = tri_delaunay.find_simplex(point)
    #     # breakpoint()
    #     if  simplex_index >= 0:  # If inside a triangle
            
    #         triangle_indices = tri_delaunay.simplices[simplex_index]  # Indices of triangle vertices

    #         # breakpoint()
    #         vertices = tri_delaunay.points[triangle_indices]  # Get the actual (x, y) coordinates
    #         v1 = vertices[0, :]
    #         v2 = vertices[1, :]
    #         v3 = vertices[2, :]

    #         vertices_orig = orig_data[triangle_indices]

    #         v1_high_dim = vertices_orig[0, :]
    #         v2_high_dim = vertices_orig[1, :]
    #         v3_high_dim = vertices_orig[2, :]

    #         lambda_1, lambda_2, lambda_3 = barycentric_coords(point, v1, v2, v3)
    #         sum_cor = lambda_1 + lambda_2 + lambda_3

    #         ## Reconstruct point in HD
    #         point_hd = lambda_1 * v1_high_dim + lambda_2 * v2_high_dim + lambda_3 * v3_high_dim

            
    #         point = torch.tensor(point, dtype=torch.float32).unsqueeze(0)
    #         # breakpoint()
    #         # y_test = np.array([])
            
    #         # point_inv, _ = model_test(X_test = point, model = model, bLossFlag = False)
    #         point_inv = model(point)
    #         f_x = np.linalg.norm(point_hd - point_inv.detach().numpy())
    #         # breakpoint()

    #         # breakpoint()
    #         ####################

    #         # print('inverse complete')
    #         intensity_interp.append(f_x)
    #     else:  # If grid outside the denaulay triangle
    #         intensity_interp.append(-1)  # Use 0 to indicate black color

    print('Convert Each point in traingle into  color array')
    # Convert colors to a NumPy array
    intensity_interp = np.array(intensity_interp)
    # breakpoint()

    # # Create a mask
    mask_not_valid = intensity_interp == -1   

    # ___Checks_________________________
    np.any((intensity_interp < 0) & (intensity_interp != -1))   ##check negative value exists other than -1


    # # Set outside-triangle points to NaN
    intensity_interp[mask_not_valid] = np.nan  
    # breakpoint()
    # # Normalize only valid (non-NaN) values
    if np.any(~mask_not_valid):  
        min_val = np.nanmin(intensity_interp)  # if we not use simple np.min(), max() it will give all nan.
        max_val = np.nanmax(intensity_interp)

        if max_val > min_val:  # Avoid division by zero
            if blog:
                intensity_interp[~mask_not_valid] = (np.log(intensity_interp[~mask_not_valid]) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))
            else:
                intensity_interp[~mask_not_valid] = (intensity_interp[~mask_not_valid] - min_val) / (max_val - min_val)
        else:
            print("Warning: No variation in values, skipping normalization.")
        
        if bclamping:
            # Compute the 5th and 95th percentiles
            lower_bound = np.percentile(intensity_interp[~mask_not_valid], 1)   # Bottom 5% threshold

            upper_bound = np.percentile(intensity_interp[~mask_not_valid], 99)  # Top 5% threshold

            upper_bound = 1

            intensity_interp[~mask_not_valid] = np.clip(intensity_interp[~mask_not_valid], lower_bound, upper_bound)  # Clamping
            intensity_interp[~mask_not_valid] = (intensity_interp[~mask_not_valid] - lower_bound) / (upper_bound - lower_bound)

    # breakpoint()
    # # Reshape for plotting
    intensity_interp_reshape = intensity_interp.reshape(xx.shape)
    # breakpoint()
    return intensity_interp_reshape, x_min, x_max, y_min, y_max

def max_min_interpolation(num_grid_points_inter, embedding, tri_delaunay, max_min_relative_edge_ratio, blog, bclamping = False):
    x_min, x_max = np.min(embedding[:, 0]), np.max(embedding[:, 0])
    y_min, y_max = np.min(embedding[:, 1]), np.max(embedding[:, 1])
    x_vals = np.linspace(x_min, x_max, num_grid_points_inter)
    y_vals = np.linspace(y_min, y_max, num_grid_points_inter)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Assign colors
    intensity_interp = []
    for point in grid_points:
        simplex_index = tri_delaunay.find_simplex(point)
        # breakpoint()
        if  simplex_index >= 0:  # If inside a triangle
            

            max_min_ratio = max_min_relative_edge_ratio[simplex_index]

            f_x = max_min_ratio
            # breakpoint()
            ####################


            intensity_interp.append(f_x)
        else:  # If grid outside the denaulay triangle
            intensity_interp.append(-1)  # Use 0 to indicate black color

    # Convert colors to a NumPy array
    intensity_interp = np.array(intensity_interp)
    # breakpoint()

    # # Create a mask
    mask_not_valid = intensity_interp == -1   

    # ___Checks_________________________
    np.any((intensity_interp < 0) & (intensity_interp != -1))   ##check negative value exists other than -1


    # # Set outside-triangle points to NaN
    intensity_interp[mask_not_valid] = np.nan  
    # breakpoint()
    # # Normalize only valid (non-NaN) values
    if np.any(~mask_not_valid):  
        min_val = np.nanmin(intensity_interp)  # if we not use simple np.min(), max() it will give all nan.
        max_val = np.nanmax(intensity_interp)

        if max_val > min_val:  # Avoid division by zero
            if blog:
                intensity_interp[~mask_not_valid] = (np.log(intensity_interp[~mask_not_valid]) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))
            else:
                intensity_interp[~mask_not_valid] = (intensity_interp[~mask_not_valid] - min_val) / (max_val - min_val)
        else:
            print("Warning: No variation in values, skipping normalization.")
        
        if bclamping:
            # Compute the 5th and 95th percentiles
            lower_bound = np.percentile(intensity_interp[~mask_not_valid], 1)  
            lower_bound = 0

            upper_bound = np.percentile(intensity_interp[~mask_not_valid], 95)  

            # upper_bound = 1

            intensity_interp[~mask_not_valid] = np.clip(intensity_interp[~mask_not_valid], lower_bound, upper_bound)  # Clamping
            intensity_interp[~mask_not_valid] = (intensity_interp[~mask_not_valid] - lower_bound) / (upper_bound - lower_bound)

    # breakpoint()
    # # Reshape for plotting
    intensity_interp_reshape = intensity_interp.reshape(xx.shape)
    # breakpoint()
    return intensity_interp_reshape, x_min, x_max, y_min, y_max



###______________ Discrete FTLE ______________________________

def discreteFTLE(Y,X):
    # """
    # Y is a 3x2 matrix where each row correspond to a vertex of the triangle in R2
    # X is a 3xn matrix where each row correspond to a vertex of the triangle in Rn
    #    - the vertices in the rows of X must correspond to the vertices in the same row of U
             
    #                    y0         |- y0 -|                      x0      |- x0 -|
    # (triangle in R2)  / \   -> Y= |- y1 -|   (triangle in Rn)  / \    U=|- x1 -|
    #                 y1 - y2       |- y2 -|                   x1 - x2    |- x2 -|
                    
    #    - y0 corresponds to x0, y1 to x1, and y2 to x2
    # """
    V = np.zeros((2,2))
    V[0] = Y[1]-Y[0]
    V[1] = Y[2]-Y[0]

    # breakpoint()
    
    U = np.zeros((2,X.shape[1]))
    U[0] = X[1]-X[0]
    U[1] = X[2]-X[0]
    # breakpoint()
    A = np.linalg.solve(V,U)
    
    rho = np.linalg.norm(A,ord=2)
    return(rho)


def calculate_FTLE_all_traingle(hd_data, ld_data, tri_nodes):
    # all_edges_length_ld = []
    # all_edges_length_hd = []
    all_ftle_rho = []

    for simplex in tri_nodes:
        # Get the coordinates of the three triangle points
        Y = ld_data[simplex]  # Shape (3, 2)
        X = hd_data[simplex]  # Shape (3, n)

        Y_norm = (Y - Y.min()) / (Y.max() - Y.min())
        X_norm = (X - X.min()) / (X.max() - X.min())

        rho = discreteFTLE(Y,X)
        # rho = discreteFTLE(Y_norm,X_norm)

        # breakpoint()
        all_ftle_rho.append(rho)

    return np.array(all_ftle_rho)


def normalised_kendall_tau_distance(values1, values2):
        """Compute the Kendall tau distance."""
        n = len(values1)
        assert len(values2) == n, "Both lists have to be of equal length"
        i, j = np.meshgrid(np.arange(n), np.arange(n))
        a = np.argsort(values1)
        b = np.argsort(values2)
        ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
        return ndisordered / (n * (n - 1))


    
        

     


