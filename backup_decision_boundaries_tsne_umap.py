import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import umap
from sklearn.preprocessing import StandardScaler
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import argparse
from inver_project_model import NNinv, model_train, model_test

from sklearn.datasets import load_iris


# Argument Parser
parser = argparse.ArgumentParser(description="Select dimensionality reduction technique: t-SNE or UMAP.")
parser.add_argument(
    "--method",
    type=str,
    choices=["tsne", "umap", "pca"],
    required=True,
    help="Choose 'tsne' or 'umap' or 'pca' for dimensionality reduction."
)
args = parser.parse_args()

# # Define MLP inverse_model
# class NNinv(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(NNinv, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_size),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.layers(x)

# Create 3D Gaussians
dim = 3
n_gauss = 6
n_pts_per_gauss = 300
num_grid_points = 100
# n_gauss = 3
np.random.seed(5)

# # Load the Iris dataset
# dim = 4
# iris = load_iris()
# data = iris['data']
# c = iris['target']
# target_names = iris['target_names']
# feature_names = iris['feature_names']

# # Center the data (subtract the mean of each feature)
# D = data - np.mean(data, axis=0)

# # Compute the covariance matrix of the centered data
# covariance_matrix = np.cov(D.T)

centers = np.random.uniform(-1, 1, size=(n_gauss, 3))
cov_m = [np.diag([0.01 for _ in range(dim)]), np.diag([0.01 if i % 2 != 0 else 0.01 for i in range(dim)])]

D = np.zeros((n_pts_per_gauss * n_gauss, dim))
c = np.zeros(n_pts_per_gauss * n_gauss)
for i in range(n_gauss):
    k = np.random.randint(0, 2, 1)[0]
    D[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = np.random.multivariate_normal(
        centers[i], cov_m[k], n_pts_per_gauss
    )
    c[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = i
D = (D - np.min(D, axis=0)) / (np.max(D, axis=0) - np.min(D, axis=0))

colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#000000']
# perplexities = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
perplexities = [1, 2, 3, 4, 5]

for perplexity in perplexities:
    if args.method == "tsne":
        reducer = manifold.TSNE(n_components=2, perplexity=perplexity, init="random", random_state=0)
        method_name = "t-SNE"
        title_var = 'perplexity: '
    elif args.method == "umap":
        reducer = umap.UMAP(n_components=2, n_neighbors=perplexity, min_dist=0.1, init="random", random_state=0)
        method_name = "UMAP"
        title_var = 'n_neighbors: '
    elif args.method == "pca":
        # Apply PCA to the 3D Gaussian data
        reducer = PCA(n_components=2, random_state=0)
        method_name = "PCA"
        title_var = 'PCA: '

    S = reducer.fit_transform(D)

    output_folder = f"thesis_reproduced/testing/{method_name}_plots"
    os.makedirs(output_folder, exist_ok=True)

    x_min, x_max = np.min(S[:, 0]), np.max(S[:, 0])
    y_min, y_max = np.min(S[:, 1]), np.max(S[:, 1])
    x_vals = np.linspace(x_min, x_max, num_grid_points)
    y_vals = np.linspace(y_min, y_max, num_grid_points)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        S, D, c, test_size=0.33, random_state=42, stratify=c
    )

    input_size = 2
    output_size = dim
    # inverse_model = NNinv(input_size, output_size)
    batch_size = 64
    num_epochs = 20
    # t_X_train = torch.tensor(X_train)
    # t_y_train = torch.tensor(y_train)
    # dataset = TensorDataset(t_X_train, t_y_train)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # inverse_model = NNinv(input_size, output_size)
    loss_fn = nn.L1Loss()
    # optimizer = optim.Adam(inverse_model.parameters(), lr=0.001)
    # num_epochs = 20

    # for epoch in range(num_epochs):
    #     running_loss = 0.0
    #     for inputs, targets in dataloader:
    #         outputs = inverse_model(inputs)
    #         loss = loss_fn(outputs, targets)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()

    inverse_model = model_train(epochs = 25, input_size= input_size, output_size= output_size, batch_size= batch_size, 
                                X_train= X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                out_folder=output_folder)

    loss = model_test(input_size =input_size, output_size = output_size, X_test = X_test, y_test = y_test, model = inverse_model)
    
    # t_X_test = torch.tensor(X_test)
    # t_y_test = torch.tensor(y_test)
    # outputs_test = inverse_model(t_X_test)
    # loss_test = loss_fn(outputs_test, t_y_test)
    # print(f'Test loss for {method_name} perplexity {perplexity}: {loss_test.item() / y_test.shape[0]:.4f}')
    print(f'Test loss for {method_name} perplexity {perplexity}: {loss:.4f}')

    jacobian_norms = np.zeros(len(grid_points))
    for idx, point in enumerate(grid_points):
        point_tensor = torch.tensor(point, dtype=torch.float32, requires_grad=True).view(1, 2)
        jacobian = torch.autograd.functional.jacobian(lambda x: inverse_model(x), point_tensor)
        jacobian_2d = jacobian.view(output_size, input_size)
        jacobian_norms[idx] = torch.linalg.norm(jacobian_2d, ord=2).item()

    jacobian_norms = jacobian_norms.reshape(xx.shape)
    

    plt.figure(figsize=(10, 8))
    for i in range(n_gauss):
        plt.scatter(S[c == i, 0], S[c == i, 1], color=colors[i], label=f'Gaussian{i + 1}', edgecolor=None)
    plt.imshow(
        jacobian_norms,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        cmap='hot',
        alpha=1
    )
    plt.colorbar(label='Spectral Norm of Jacobian')
    plt.title(f"{method_name} {title_var}  {perplexity}")
    plt.xlabel(f"{method_name} Dimension 1")
    plt.ylabel(f"{method_name} Dimension 2")
    # plt.legend()
    output_path = os.path.join(output_folder, f"jacobian_heatmap_{method_name}_{perplexity}.png")
    plt.savefig(output_path)
    plt.close()

print(f"Plots saved in folder: {output_folder}")
