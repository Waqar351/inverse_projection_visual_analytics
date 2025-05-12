import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.model_selection import ParameterGrid, ParameterSampler, train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Define MLP inverse_model
class NNinv(nn.Module):
    def __init__(self, input_size, output_size):
        super(NNinv, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Create 3D Gaussians
dim = 3
n_gauss = 6
n_pts_per_gauss = 300
num_grid_points = 100
np.random.seed(5)

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

# Define the parameter grid
param_grid = {
    "perplexity": [5, 10, 20, 30, 50],
    "learning_rate": [10, 50, 100, 200, 500],
    "n_iter": [250, 500, 1000],
    "metric": ["euclidean", "cosine"]
}
param_combinations = list(ParameterGrid(param_grid))

results = []
output_folder = "tSNE_parameter_tuning"
os.makedirs(output_folder, exist_ok=True)

for params in param_combinations:
    perplexity = params["perplexity"]
    learning_rate = params["learning_rate"]
    n_iter = params["n_iter"]
    metric = params["metric"]

    # t-SNE Reducer
    reducer = manifold.TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        metric=metric,
        init="random",
        random_state=0
    )

    S = reducer.fit_transform(D)

    x_min, x_max = np.min(S[:, 0]), np.max(S[:, 0])
    y_min, y_max = np.min(S[:, 1]), np.max(S[:, 1])
    x_vals = np.linspace(x_min, x_max, num_grid_points)
    y_vals = np.linspace(y_min, y_max, num_grid_points)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Split data
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        S, D, c, test_size=0.33, random_state=42, stratify=c
    )

    # Define model and training
    input_size = 2
    output_size = dim
    batch_size = 64
    t_X_train = torch.tensor(X_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(t_X_train, t_y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    inverse_model = NNinv(input_size, output_size)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(inverse_model.parameters(), lr=0.001)
    num_epochs = 20

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            outputs = inverse_model(inputs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate test loss
    t_X_test = torch.tensor(X_test, dtype=torch.float32)
    t_y_test = torch.tensor(y_test, dtype=torch.float32)
    outputs_test = inverse_model(t_X_test)
    loss_test = loss_fn(outputs_test, t_y_test)
    test_loss = loss_test.item() / y_test.shape[0]
    results.append((params, test_loss))

    # Jacobian norm
    jacobian_norms = np.zeros(len(grid_points))
    for idx, point in enumerate(grid_points):
        point_tensor = torch.tensor(point, dtype=torch.float32, requires_grad=True).view(1, 2)
        jacobian = torch.autograd.functional.jacobian(lambda x: inverse_model(x), point_tensor)
        jacobian_2d = jacobian.view(output_size, input_size)
        jacobian_norms[idx] = torch.linalg.norm(jacobian_2d, ord=2).item()

    jacobian_norms = jacobian_norms.reshape(xx.shape)

    # Save plots
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
    plt.title(f"t-SNE (perp={perplexity}, lr={learning_rate}, iter={n_iter}, metric={metric})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    output_path = os.path.join(output_folder, f"jacobian_tSNE_perp{perplexity}_lr{learning_rate}_iter{n_iter}_metric{metric}.png")
    plt.savefig(output_path)
    plt.close()

# Analyze results
results = sorted(results, key=lambda x: x[1])
print("Top 5 parameter combinations with lowest loss:")
for params, loss in results[:5]:
    print(f"Params: {params}, Test Loss: {loss:.4f}")

# Plot performance vs perplexity
import pandas as pd
df = pd.DataFrame(results, columns=["params", "loss"])
df["perplexity"] = df["params"].apply(lambda x: x["perplexity"])
df.groupby("perplexity")["loss"].mean().plot(kind="line", title="Loss vs Perplexity", marker='o')
plt.xlabel("Perplexity")
plt.ylabel("Average Loss")
plt.grid()
plt.show()
