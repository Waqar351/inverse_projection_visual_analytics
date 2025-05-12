# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import manifold
# import umap
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from shiny import App, ui, render, reactive
# import tempfile

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

# # Define UI
# app_ui = ui.page_fluid(
#     ui.panel_title("Dimensionality Reduction and Jacobian Visualization"),
#     ui.layout_sidebar(
#         ui.sidebar(  # Sidebar content
#             ui.input_select("method", "Dimensionality Reduction Method:", 
#                             {"tsne": "t-SNE", "umap": "UMAP", "pca": "PCA"}),
#             ui.input_slider("perplexity", "Perplexity / n_neighbors:", min=1, max=50, value=10),
#             ui.input_slider("epochs", "Number of Training Epochs:", min=10, max=50, value=20),
#             ui.input_slider("gaussians", "Number of Gaussians:", min=2, max=10, value=6),
#             ui.input_action_button("run", "Run Experiment"),
#         ),
#         ui.div(  # Main panel using div to display outputs
#             ui.output_image("heatmap"),
#             ui.output_image("gaussian_3d"),
#             # ui.output_text("loss_output")
#         ),
#     ),
# )

# def server(input, output, session):
#     # Reactive container to store Gaussian data
#     gaussian_data = reactive.Value(None)  # Initialize as None

#     @reactive.event(input.run)
#     def generate_gaussian_data():
#         """Generate and store Gaussian data when 'Run Experiment' is clicked."""
#         n_gauss = input.gaussians()
#         dim = 3
#         n_pts_per_gauss = 300
#         np.random.seed(5)

#         # Generate Gaussian data
#         centers = np.random.uniform(-1, 1, size=(n_gauss, dim))
#         cov_m = [np.diag([0.01 for _ in range(dim)]), np.diag([0.01 if i % 2 != 0 else 0.01 for i in range(dim)])]
#         D = np.zeros((n_pts_per_gauss * n_gauss, dim))
#         c = np.zeros(n_pts_per_gauss * n_gauss)

#         for i in range(n_gauss):
#             k = np.random.randint(0, 2, 1)[0]
#             D[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = np.random.multivariate_normal(
#                 centers[i], cov_m[k], n_pts_per_gauss
#             )
#             c[i * n_pts_per_gauss:(i + 1) * n_pts_per_gauss] = i

#         # Save normalized data and class labels in reactive container
#         D = (D - np.min(D, axis=0)) / (np.max(D, axis=0) - np.min(D, axis=0))
#         gaussian_data.set({"D": D, "c": c})
#         # Ensure Gaussian data is correctly updated
#         print("Generated Gaussian data")

#     @render.image
#     @reactive.event(input.run)
#     def gaussian_3d():
#         """Generate 3D Gaussian scatter plot."""
#         data = gaussian_data.get()
#         if not data:
#             return  # Return nothing if data hasn't been generated

#         D, c = data["D"], data["c"]

#         # Save the 3D scatter plot
#         temp_dir = tempfile.mkdtemp()
#         output_path = os.path.join(temp_dir, "gaussian_3d.png")

#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(111, projection="3d")
#         ax.scatter(D[:, 0], D[:, 1], D[:, 2], c=c, cmap="tab10", s=5)
#         ax.set_title("3D Gaussian Scatter Plot")
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_zlabel("Z")
#         plt.savefig(output_path)
#         plt.close()

#         return {"src": output_path, "width": "600px", "height": "500px"}

#     @render.image
#     @reactive.event(input.run)
#     def heatmap():
#         """Generate Jacobian heatmap after dimensionality reduction."""
#         data = gaussian_data.get()
#         if not data:
#             return  # Return nothing if data hasn't been generated

#         D, c = data["D"], data["c"]

#         # Parameters
#         method = input.method()
#         perplexity = input.perplexity()
#         num_epochs = input.epochs()
#         num_grid_points = 100
#         dim = 3

#         # Dimensionality reduction
#         if method == "tsne":
#             reducer = manifold.TSNE(n_components=2, perplexity=perplexity, init="random", random_state=0)
#         elif method == "umap":
#             reducer = umap.UMAP(n_components=2, n_neighbors=perplexity, min_dist=0.1, init="random", random_state=0)
#         elif method == "pca":
#             reducer = PCA(n_components=2, random_state=0)

#         S = reducer.fit_transform(D)

#         # Train inverse model
#         X_train, X_test, y_train, y_test = train_test_split(S, D, test_size=0.33, random_state=42, stratify=c)
#         input_size = 2
#         batch_size = 64
#         t_X_train = torch.tensor(X_train, dtype=torch.float32)
#         t_y_train = torch.tensor(y_train, dtype=torch.float32)
#         dataset = TensorDataset(t_X_train, t_y_train)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#         inverse_model = NNinv(input_size, dim)
#         loss_fn = nn.L1Loss()
#         optimizer = optim.Adam(inverse_model.parameters(), lr=0.001)

#         for epoch in range(num_epochs):
#             for inputs, targets in dataloader:
#                 outputs = inverse_model(inputs)
#                 loss = loss_fn(outputs, targets)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         # Jacobian heatmap
#         x_min, x_max = np.min(S[:, 0]), np.max(S[:, 0])
#         y_min, y_max = np.min(S[:, 1]), np.max(S[:, 1])
#         x_vals = np.linspace(x_min, x_max, num_grid_points)
#         y_vals = np.linspace(y_min, y_max, num_grid_points)
#         xx, yy = np.meshgrid(x_vals, y_vals)
#         grid_points = np.c_[xx.ravel(), yy.ravel()]

#         jacobian_norms = np.zeros(len(grid_points))
#         for idx, point in enumerate(grid_points):
#             point_tensor = torch.tensor(point, dtype=torch.float32, requires_grad=True).view(1, 2)
#             jacobian = torch.autograd.functional.jacobian(lambda x: inverse_model(x), point_tensor)
#             jacobian_2d = jacobian.view(dim, input_size)
#             jacobian_norms[idx] = torch.linalg.norm(jacobian_2d, ord=2).item()
#         jacobian_norms = jacobian_norms.reshape(xx.shape)

#         # Save heatmap
#         temp_dir = tempfile.mkdtemp()
#         output_path = os.path.join(temp_dir, "heatmap.png")
#         plt.figure(figsize=(10, 8))
#         plt.imshow(jacobian_norms, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='hot', alpha=1)
#         plt.colorbar(label='Spectral Norm of Jacobian')
#         plt.title(f"{method.upper()} Heatmap")
#         plt.xlabel("Dimension 1")
#         plt.ylabel("Dimension 2")
#         plt.savefig(output_path)
#         plt.close()

#         return {"src": output_path, "width": "600px", "height": "500px"}

# # Launch app
# app = App(app_ui, server)


#################################################################################################


import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import umap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from shiny import App, ui, render, reactive
import tempfile

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

# Define UI
app_ui = ui.page_fluid(
    ui.panel_title("Dimensionality Reduction and Jacobian Visualization"),
    ui.layout_sidebar(
        ui.sidebar(  # Sidebar content
            ui.input_select("method", "Dimensionality Reduction Method:", 
                            {"tsne": "t-SNE", "umap": "UMAP", "pca": "PCA"}),
            ui.input_slider("perplexity", "Perplexity / n_neighbors:", min=1, max=50, value=10),
            ui.input_slider("epochs", "Number of Training Epochs:", min=10, max=50, value=20),
            ui.input_slider("gaussians", "Number of Gaussians:", min=2, max=10, value=6),
            ui.input_action_button("run", "Run Experiment"),
        ),
        ui.div(  # Main panel using div to display outputs
            ui.output_image("heatmap"),
            ui.output_image("gaussian_3d"),
            # ui.output_text("loss_output")
        ),
    ),
)



# Define Server
def server(input, output, session):
    @render.image
    @reactive.event(input.run)
    def heatmap():
        # Parameters from user input
        method = input.method()
        perplexity = input.perplexity()
        num_epochs = input.epochs()
        n_gauss = input.gaussians()

        # Create synthetic 3D Gaussians
        dim = 3
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

        # Dimensionality reduction
        if method == "tsne":
            reducer = manifold.TSNE(n_components=2, perplexity=perplexity, init="random", random_state=0)
        elif method == "umap":
            reducer = umap.UMAP(n_components=2, n_neighbors=perplexity, min_dist=0.1, init="random", random_state=0)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=0)

        S = reducer.fit_transform(D)

        # Train inverse model
        X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
            S, D, c, test_size=0.33, random_state=42, stratify=c
        )

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

        for epoch in range(num_epochs):
            for inputs, targets in dataloader:
                outputs = inverse_model(inputs)
                loss = loss_fn(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Jacobian heatmap
        x_min, x_max = np.min(S[:, 0]), np.max(S[:, 0])
        y_min, y_max = np.min(S[:, 1]), np.max(S[:, 1])
        x_vals = np.linspace(x_min, x_max, num_grid_points)
        y_vals = np.linspace(y_min, y_max, num_grid_points)
        xx, yy = np.meshgrid(x_vals, y_vals)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        jacobian_norms = np.zeros(len(grid_points))
        for idx, point in enumerate(grid_points):
            point_tensor = torch.tensor(point, dtype=torch.float32, requires_grad=True).view(1, 2)
            jacobian = torch.autograd.functional.jacobian(lambda x: inverse_model(x), point_tensor)
            jacobian_2d = jacobian.view(output_size, input_size)
            jacobian_norms[idx] = torch.linalg.norm(jacobian_2d, ord=2).item()
        jacobian_norms = jacobian_norms.reshape(xx.shape)

        # Plot and save heatmap
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "heatmap.png")
        plt.figure(figsize=(10, 8))
        plt.imshow(
            jacobian_norms,
            extent=(x_min, x_max, y_min, y_max),
            origin='lower',
            cmap='hot',
            alpha=1
        )
        plt.colorbar(label='Spectral Norm of Jacobian')
        plt.title(f"{method.upper()} Heatmap")
        plt.xlabel(f"Dimension 1")
        plt.ylabel(f"Dimension 2")
        plt.savefig(output_path)
        plt.close()
        return {"src": output_path, "width": "600px", "height": "500px"}

    @render.text
    def loss_output():
        return f"Model trained successfully with {input.method().upper()} and {input.epochs()} epochs!"

# Launch app
app = App(app_ui, server)
