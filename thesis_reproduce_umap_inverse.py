import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_iris
import os
import torch

# Load the Iris dataset
iris = load_iris()
data = iris['data']
c = iris['target']
target_names = iris['target_names']

# Center the data (subtract the mean of each feature)
D = data - np.mean(data, axis=0)

# UMAP dimensionality reduction
n_neighbors = 15  # Set a default value for UMAP
umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
S = umap_model.fit_transform(D)

# Generate a grid of points for Jacobian computation
num_grid_points = 50
x_min, x_max = np.min(S[:, 0]), np.max(S[:, 0])
y_min, y_max = np.min(S[:, 1]), np.max(S[:, 1])
x_vals = np.linspace(x_min, x_max, num_grid_points)
y_vals = np.linspace(y_min, y_max, num_grid_points)
xx, yy = np.meshgrid(x_vals, y_vals)
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Function for inverse mapping (using UMAP's inverse_transform method)
def inverse_mapping_umap(point_2d):
    # Ensure the input is a NumPy array and not a tensor that requires gradients
    point_2d = point_2d.detach().numpy()  # Detach tensor to avoid gradient tracking
    high_dim_point = umap_model.inverse_transform(point_2d)
    return torch.tensor(high_dim_point, dtype=torch.float32)  # Return as Tensor

# Compute the Jacobian norm for each grid point
jacobian_norms = np.zeros(len(grid_points))

print('jacobian_calculation start')
for idx, point in enumerate(grid_points):
    print('point', idx, point)
    # Convert point to tensor and ensure it doesn't require gradients
    point_tensor = torch.tensor(point, dtype=torch.float32).view(1, 2)
    
    # Get the original high-dimensional point using inverse transform from UMAP
    high_dim_point = inverse_mapping_umap(point_tensor)
    
    # Define a simple model to compute the Jacobian
    high_dim_point_tensor = high_dim_point.clone().detach().requires_grad_(True)

    # Calculate the Jacobian by differentiating the inverse transform
    jacobian = torch.autograd.functional.jacobian(lambda x: inverse_mapping_umap(x), point_tensor)
    jacobian_2d = jacobian.view(D.shape[1], 2)  # Adjust based on the dimensionality of your original data
    jacobian_norms[idx] = torch.linalg.norm(jacobian_2d, ord=2).item()

print('jacobian_calculation ends')

# Reshape the jacobian norms to match the grid for visualization
jacobian_norms = jacobian_norms.reshape(xx.shape)

# Plot and save the heatmap
output_folder = f"thesis_reproduced/UMAP_plots"
print(f"Creating output folder at: {output_folder}")  # Debug print
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(10, 8))
for i in range(len(target_names)):
    plt.scatter(S[c == i, 0], S[c == i, 1], label=f'Class {target_names[i]}')

plt.imshow(
    jacobian_norms,
    extent=(x_min, x_max, y_min, y_max),
    origin='lower',
    cmap='hot',
    alpha=0.6
)
plt.colorbar(label='Spectral Norm of Jacobian')
plt.title(f"UMAP Jacobian Spectral Norm Heatmap")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
output_path = os.path.join(output_folder, "umap_jacobian_heatmap.png")
plt.savefig(output_path)
plt.close()

print(f"Plots saved in folder: {output_folder}")
