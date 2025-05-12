import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import os


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

#########################################################################################
# Define Jacobian
def compute_jacobian(x, y):
    """
    Computes the Jacobian matrix for each point in the grid.
    
    Args:
        grid_points (ndarray): A 2D array of shape (n_points, 2) representing the grid points.

    Returns:
        jacobian_matrices (list): A list of jacobian matrices for each grid point.
    """
    jacobian_matrices = []
    
    # Define the model's forward pass to use autograd
    def model_forward(input):
        return inverse_model(input)  # Model's forward pass
    
    # Iterate through the grid points
    # for point in grid_points:
    point_tensor = torch.tensor([x, y], dtype=torch.float32, requires_grad=True)  # (1, 2) tensor
    
    # Compute the Jacobian using autograd's jacobian function
    jacobian = torch.autograd.functional.jacobian(model_forward, point_tensor)
    
        # The output of jacobian will have shape (1, 3, 2), so we need to squeeze to get (3, 2)
        # jacobian_matrices.append(jacobian.squeeze(0))  # Remove the batch dimension
    
    return jacobian
## Define the Inverse Projection


# Define the MLP inverse_model
class NNinv(nn.Module):
    def __init__(self, input_size, output_size):
        super(NNinv, self).__init__()
        
        # Define the layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),  # Input to first hidden layer
            nn.ReLU(),
            nn.Linear(64, 128),  # First hidden layer to second hidden layer
            nn.ReLU(),
            nn.Linear(128, 256),  # Second hidden layer to third hidden layer
            nn.ReLU(),
            nn.Linear(256, 512),  # Third hidden layer to fourth hidden layer
            nn.ReLU(),
            nn.Linear(512, output_size),  # Fifth hidden layer to output
            nn.Sigmoid()  # Output layer with sigmoid activation
        )
    
    def forward(self, x):
        return self.layers(x)

################################################################################################

### Create 3D Gaussians

dim = 3
n_gauss = 6
n_pts_per_gauss = 300
num_grid_points = 100
np.random.seed(5)


centers = np.random.uniform(-1,1,size=(n_gauss,3))
    

cov_m = [np.diag([0.01 for i in range(dim)]),np.diag([0.01 if i%2 !=0 else 0.01 for i in range(dim)])]

D = np.zeros((n_pts_per_gauss*n_gauss,dim))
c = np.zeros(n_pts_per_gauss*n_gauss)      # storage for labels
for i in range(n_gauss):
    k = np.random.randint(0,2,1)[0]
    D[i*n_pts_per_gauss:(i+1)*n_pts_per_gauss] = np.random.multivariate_normal(centers[i],cov_m[k],n_pts_per_gauss)
    c[i*n_pts_per_gauss:(i+1)*n_pts_per_gauss] = i 
D = (D-np.min(D,axis=0))/(np.max(D,axis=0)-np.min(D,axis=0))


colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#000000']

### # Define Projections



perplexities = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
for perplexity in perplexities:

    # t_sne = manifold.TSNE(
    #     n_components=2,
    #     perplexity=perplexity,
    #     init="random",
    #     random_state=0,
    # )

    # S = t_sne.fit_transform(D)

    t_sne = umap.UMAP(
    n_components=2,     # Targeting 2D projection
    n_neighbors=perplexity,     # Similar to t-SNE perplexity (10)
    min_dist=0.1,       # Controls the compactness of the clusters
    init="random",
    random_state=0, 
    )
    #Apply UMAP on the 3D Gaussian data `D`
    S = t_sne.fit_transform(D)

    ## Create a 2D Grid for Jacobian Calculation

    # # Define min and max values
    x_min, x_max = np.min(S[:, 0]), np.max(S[:, 0])
    y_min, y_max = np.min(S[:, 1]), np.max(S[:, 1])

    # Generate grid
    x_vals = np.linspace(x_min, x_max, num_grid_points)
    y_vals = np.linspace(y_min, y_max, num_grid_points)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]


    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(S, D,c, test_size=0.33, random_state=42, stratify=c)


    ### Training the model

    input_size = 2  
    output_size = dim   # Binary classification (sigmoid output for single output)

    # Create DataLoader for batch processing
    batch_size = 64
    t_X_train = torch.tensor(X_train)
    t_y_train = torch.tensor(y_train)
    dataset = TensorDataset(t_X_train, t_y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the inverse_model, loss function, and optimizer
    inverse_model = NNinv(input_size, output_size)
    loss_fn = nn.L1Loss()  # Mean Absolute Error (MAE)
    optimizer = optim.Adam(inverse_model.parameters(), lr=0.001)

    # Number of epochs to train
    num_epochs = 20

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = inverse_model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Print the average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    print(f"Training complete for perplexity {perplexity}")

    t_X_test = torch.tensor(X_test)
    t_y_test = torch.tensor(y_test)
    outputs_test = inverse_model(t_X_test)
    loss_test = loss_fn(outputs_test, t_y_test)
    print(f'Test loss for {perplexity} ', loss_test/y_test.shape[0])

    ### Estimating Jacobian and spectral norm

    jacobian_norms = np.zeros(len(grid_points))
    for idx, point in enumerate(grid_points):
        point_tensor = torch.tensor(point, dtype=torch.float32, requires_grad=True).view(1, 2)
        
        # Compute the Jacobian for the current point
        jacobian = torch.autograd.functional.jacobian(lambda x: inverse_model(x), point_tensor)
        
        # Reshape Jacobian to 2D: (output_dim, input_dim)
        jacobian_2d = jacobian.view(output_size, input_size)  # Assuming output is (1, 3), input is (1, 2)
        
        # Compute spectral norm (largest singular value)
        jacobian_norms[idx] = torch.linalg.norm(jacobian_2d, ord=2).item()

    jacobian_norms = jacobian_norms.reshape(xx.shape)

    # Create folder for saving plots
    output_folder = "n_neighbors_UMAP_plots"
    os.makedirs(output_folder, exist_ok=True)
    ### Plot heatmap with t-SNE points overlayed
    plt.figure(figsize=(10, 8))

    # Overlay t-SNE points
    # plt.scatter(S[:, 0], S[:, 1], c='blue', edgecolor='k', label='t-SNE points')
    for i in range(n_gauss):
        plt.scatter(S[c == i, 0], S[c == i, 1], color=colors[i], label=f'Gaussian{i+1}', edgecolor=None)

    # Plot heatmap
    plt.imshow(
        jacobian_norms,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        cmap='hot',
        alpha=1
    )
    plt.colorbar(label='Spectral Norm of Jacobian')




    # Labels and title
    plt.title(f"n_neighbors {perplexity}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2") 
    output_path = os.path.join(output_folder, f"jacobian_heatmap_n_neighbors_{perplexity}.png")
    plt.savefig(output_path)  # Save plot
    plt.close()  # Close the plot to avoid overlap

print(f"Plots saved in folder: {output_folder}")