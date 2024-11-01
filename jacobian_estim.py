import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.model_selection import train_test_split
import torch

from inver_project_model import NNinv, model_train, model_test

###__________ Create 3D Gaussian________________________
dim = 3
n_gauss = 3
n_pts_per_gauss = 300
np.random.seed(5)

centers = np.zeros((n_gauss,dim))
for i in range(1,n_gauss):
    centers[i] = np.random.randint(0,2,3)
    
print(centers)

cov_m = [np.diag([0.01 for i in range(dim)]),np.diag([0.1 if i%2 !=0 else 0.1 for i in range(dim)])]

D = np.zeros((n_pts_per_gauss*n_gauss,dim))
c = np.zeros(n_pts_per_gauss*n_gauss)
for i in range(dim):
    k = np.random.randint(0,2,1)[0]
    D[i*n_pts_per_gauss:(i+1)*n_pts_per_gauss] = np.random.multivariate_normal(centers[i],cov_m[k],n_pts_per_gauss)
    c[i*n_pts_per_gauss:(i+1)*n_pts_per_gauss] = i
D = (D-np.min(D,axis=0))/(np.max(D,axis=0)-np.min(D,axis=0))

#############_____________Project 3D to 2D using T-SNE____________________#############################################################################################

t_sne = manifold.TSNE(
    n_components=2,
    perplexity=30,
    init="random",
    random_state=0,
)

S = t_sne.fit_transform(D)

# Plotting the t-SNE results with the same color scheme
# %matplotlib qt

colors = ['r', 'g', 'b']  # Red, Green, Blue
plt.figure(figsize=(10, 8))
for i in range(n_gauss):
    plt.scatter(S[c == i, 0], S[c == i, 1], color=colors[i], label=f'Gaussian {i+1}')

plt.title('t-SNE Visualization of 3D Gaussian Distributions into 2D')
plt.legend()
# plt.show()

#####################____________Create a 2D Grid for Jacobian Calculation___________############################################################

# Define min and max values
x_min, x_max = np.min(S[:, 0]), np.max(S[:, 0])
y_min, y_max = np.min(S[:, 1]), np.max(S[:, 1])
# Define grid resolution
num_grid_points = 10

# Generate grid
x_vals = np.linspace(x_min, x_max, num_grid_points)
y_vals = np.linspace(y_min, y_max, num_grid_points)
xx, yy = np.meshgrid(x_vals, y_vals)

plt.figure(figsize=(10, 8))
# Visualize the grid on top of the t-SNE data
plt.scatter(S[:, 0], S[:, 1], c='blue', s=10, label="t-SNE Output")
plt.scatter(xx, yy, c='red', s=5, label="Grid Points")
plt.title("2D t-SNE Output with Grid Points")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
# plt.grid(True)
# plt.show()

##################___ Inverse Model Training_________________###################

X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(S, D,c, test_size=0.33, random_state=42)
input_size = 2  # Example input size (can be changed)
output_size = dim

model_train(input_size= 2, output_size=3, batch_size= 64, X_train= X_train, y_train= y_train, X_test= X_test, y_test= y_test)

# Create a new instance of the model
inverse_model = NNinv(input_size, output_size)
# Load the trained model parameters
inverse_model.load_state_dict(torch.load('inverse_model.pth', weights_only=True))

model_test(X_test= X_test, y_test= y_test, model=inverse_model)

