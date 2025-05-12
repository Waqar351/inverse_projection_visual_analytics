from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer, load_linnerud, fetch_openml
from utility import *
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

def selected_dataset_dt(dataset, num_dim, n_pts_per_gauss, cluster_spacing = 1.0, spread_factor = 0.01):

    if dataset == "gaussian":
        centers, overlap_factor = cluster_position(cluster_spacing, mode = dataset)
        dim = 3
        output_size = dim
        n_gauss = 6
        D, c, centers = gaussian_dt(n_gauss, n_pts_per_gauss, dim)
    elif dataset == "tetrahedron_eq":
        distance_factor = 1.0
        distance_factor_2 = 1.0
        move_cluster_index=0
        centers, overlap_factor = cluster_position(cluster_spacing, mode = dataset)
        D, c, centers = generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=n_pts_per_gauss, base_tetrahedron = centers, spread_factor= spread_factor, distance_factor=distance_factor, distance_factor_2= distance_factor_2, move_cluster_index=move_cluster_index)
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
        
    elif dataset == "tetrahedron_eq_1_far":
        distance_factor = 3.0
        distance_factor_2 = 1.0
        move_cluster_index=0
        centers, overlap_factor = cluster_position(cluster_spacing, mode = dataset)
        D, c, centers = generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=n_pts_per_gauss, base_tetrahedron = centers, spread_factor= spread_factor, distance_factor=distance_factor, distance_factor_2= distance_factor_2, move_cluster_index=move_cluster_index)
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == "tetrahedron_eq_1_close":
        distance_factor = 0.0
        distance_factor_2 = 1.0
        move_cluster_index=0
        centers, overlap_factor = cluster_position(cluster_spacing, mode = dataset)
        D, c, centers = generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=n_pts_per_gauss, base_tetrahedron = centers, spread_factor= spread_factor, distance_factor=distance_factor, distance_factor_2= distance_factor_2, move_cluster_index=move_cluster_index)
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == "tetrahedron_eq_2_close":
        distance_factor = 0.5
        distance_factor_2 = 0.5
        move_cluster_index=0
        centers, overlap_factor = cluster_position(cluster_spacing, mode = dataset)
        D, c, centers = generate_dynamic_tetrahedral_gaussians(n_pts_per_gauss=n_pts_per_gauss, base_tetrahedron = centers, spread_factor= spread_factor, distance_factor=distance_factor, distance_factor_2= distance_factor_2, move_cluster_index=move_cluster_index)
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))

    elif dataset == "iris":
        dim = 4
        output_size = dim
        n_gauss = 3  # number of classes
        D, c = iris_dt()

    elif dataset == 'digits':

        D, c = digits_dt()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
        
    elif dataset == 'har':

        D, c = har_dt()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == 'covariance':

        D, c = covariance_type()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == 'wine':

        D, c = wine_dt()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == 'breast':

        D, c = breast_cancer_dt()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == 'cifar':

        D, c = cifar_10()
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == 'high_dim':
        D, c, centers = generate_high_dimension_gaussians(num_dim = num_dim, n_pts_per_gauss=200, spread_factor= spread_factor)
        dim = D.shape[1]
        output_size = dim
        n_gauss = len(np.unique(c))
    elif dataset == 'mnist':
        D, c = MNIST()
        dim = D.shape[1]
        output_size = dim
        breakpoint()
        n_gauss = len(np.unique(c))
    elif dataset == 'linnerud':
        D, c = linnerud()
        dim = D.shape[1]
        output_size = dim
        # breakpoint()
        n_gauss = len(np.unique(c))

    else:
        raise ValueError("Invalid dataset name.")

    return D,c, dim, output_size, n_gauss
##################################################################################################################################
# Get the current working directory (main project folder)
project_dir = os.getcwd()

def gaussian_dt(n_gauss, n_pts_per_gauss, dim):
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

    return D, c, centers

def iris_dt():
    iris = load_iris()
    data = iris['data']
    c = iris['target']
    target_names = iris['target_names']
    feature_names = iris['feature_names']

    # # Center the data (subtract the mean of each feature)
    # X_normalized = data - np.mean(data, axis=0)

    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(data)

    # Compute the covariance matrix of the centered data
    covariance_matrix = np.cov(X_normalized.T)

    return X_normalized, c

def digits_dt():
    # Load the digits dataset
    digits = load_digits()

    # Features and target
    X = digits.data
    y = digits.target

    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X)
    # # Standardize the features
    # scaler = StandardScaler()
    # X_stand = scaler.fit_transform(X)
    return X_normalized, y


def har_dt(sample_size=None, random_state=5):
    """
    Load and preprocess the HAR dataset, retaining only specific activities.

    Parameters:
        sample_size (int): Number of samples to retain (optional).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Normalized features and filtered labels.
    """
    # Construct the path to the datasets folder dynamically
    datasets_folder = os.path.join(project_dir, "datasets", "UCI_ HAR_ Dataset", "train")

    # File paths
    x_train_path = os.path.join(datasets_folder, "X_train.txt")
    y_train_path = os.path.join(datasets_folder, "y_train.txt")

    # Load the data as NumPy arrays
    X_train = np.loadtxt(x_train_path)
    y_train = np.loadtxt(y_train_path)

    # Filter for specific labels (WALKING: 1, SITTING: 4, STANDING: 5,  Laying: 6)
    desired_labels = [1, 2, 3, 4, 5, 6]
    # desired_labels = [1, 3]# 3, 4, 5, 6]   # C_0 = 1226, C_1= 1073,  C_2= 986
    mask = np.isin(y_train, desired_labels)
    X_filtered = X_train[mask]
    y_filtered = y_train[mask]

    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X_filtered)

    # Remap the labels to consecutive integers
    label_mapping = {original: new for new, original in enumerate(desired_labels, start=0)}
    y_filtered = np.array([label_mapping[label] for label in y_filtered])
    # If sample_size is specified, perform stratified sampling
    if sample_size is not None and sample_size < len(y_filtered):
        X_normalized, _, y_filtered, _ = train_test_split(
            X_normalized,
            y_filtered,
            train_size=sample_size,
            stratify=y_filtered,
            random_state=random_state
        )
    # breakpoint()
    # Count samples for each class
    unique_labels, counts = np.unique(y_filtered, return_counts=True)
    print("Number of samples for each class:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {int(label)}: {count} samples")
    
        # Sort indices based on class labels
    sorted_indices = np.argsort(y_filtered)

    # Reorder dataset and labels
    X_sorted = X_normalized[sorted_indices]
    y_sorted = y_filtered[sorted_indices]
    # return X_normalized, y_filtered
    return X_sorted, y_sorted

def har_dt_v2(sample_size=None, random_state=5):
    """
    Load and preprocess the HAR dataset, retaining only specific activities.

    Parameters:
        sample_size (int): Number of samples to retain (optional).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Normalized features and filtered labels.
    """
    # Construct the path to the datasets folder dynamically
    datasets_folder = os.path.join(project_dir, "datasets", "UCI_ HAR_ Dataset", "train")

    # File paths
    x_train_path = os.path.join(datasets_folder, "X_train.txt")
    y_train_path = os.path.join(datasets_folder, "y_train.txt")

    # Load the data as NumPy arrays
    X_train = np.loadtxt(x_train_path)
    y_train = np.loadtxt(y_train_path)

    # # Filter for specific labels (WALKING: 1, SITTING: 4, STANDING: 5,  Laying: 6)
    # desired_labels = [1, 2, 3, 4, 5, 6]
    # # desired_labels = [1, 3]# 3, 4, 5, 6]   # C_0 = 1226, C_1= 1073,  C_2= 986
    # mask = np.isin(y_train, desired_labels)
    # X_filtered = X_train[mask]
    # y_filtered = y_train[mask]

    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X_train)
    desired_labels = [1, 2, 3, 4, 5, 6]

    # # Remap the labels to consecutive integers
    label_mapping = {original: new for new, original in enumerate(desired_labels, start=0)}
    y_filtered = np.array([label_mapping[label] for label in y_train])
    # # If sample_size is specified, perform stratified sampling
    # if sample_size is not None and sample_size < len(y_filtered):
    #     X_normalized, _, y_filtered, _ = train_test_split(
    #         X_normalized,
    #         y_filtered,
    #         train_size=sample_size,
    #         stratify=y_filtered,
    #         random_state=random_state
    #     )
    # # breakpoint()
    # # Count samples for each class
    # breakpoint()
    unique_labels, counts = np.unique(y_train, return_counts=True)
    print("Number of samples for each class:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {int(label)}: {count} samples")
    return X_normalized, y_filtered



def covariance_type(sample_size=None, random_state=5):
    """
    Reads the covariance dataset, separates labels, normalizes features,
    and returns a stratified sample if sample_size is provided.

    Args:
    - sample_size (int): Number of samples to return (maintains class ratio).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - normalized_features (ndarray): Normalized feature matrix.
    - labels (ndarray): Corresponding labels.
    """
    # Construct the path to the datasets folder dynamically
    datasets_folder = os.path.join(project_dir, "datasets", "covariance_type")
    
    # File path
    data_path = os.path.join(datasets_folder, "covertype.csv")
    
    # Read the CSV file while skipping the first row (column names)
    data = pd.read_csv(data_path, header=0)
    
    # Separate the label (last column)
    labels = data.iloc[:, -1].values  # Extract the last column as the labels
    
    # Extract features (all columns except the last)
    features = data.iloc[:, :-1].values  # Extract all columns except the last
    # breakpoint()
    # Filter for specific labels
    # desired_labels = [3, 5, 7]
    desired_labels = [1,2,3, 4,5,6, 7]
    mask = np.isin(labels, desired_labels)
    features = features[mask]
    labels = labels[mask]
    
    # Normalize the features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Remap the labels to consecutive integers
    label_mapping = {original: new for new, original in enumerate(desired_labels, start=0)}
    labels = np.array([label_mapping[label] for label in labels])
    
    if sample_size is not None:
        # Stratified sampling to maintain class ratios
        normalized_features, _, labels, _ = train_test_split(
            normalized_features,
            labels,
            train_size=sample_size,
            stratify=labels,
            random_state=random_state
        )

    # Count samples for each class
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Number of samples for each class:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {int(label)}: {count} samples")
    
    return normalized_features, labels

def cifar_10(sample_size=None, random_state=5):
    """
    Reads the cifar_10 dataset, separates labels, normalizes features,
    and returns a stratified sample if sample_size is provided.

    Args:
    - sample_size (int): Number of samples to return (maintains class ratio).
    - random_state (int): Random seed for reproducibility.

    Returns:
    - normalized_features (ndarray): Normalized feature matrix.
    - labels (ndarray): Corresponding labels.
    """
    # Construct the path to the datasets folder dynamically
    datasets_folder = os.path.join(project_dir, "datasets", "cifar-10-python_cs_totonto", "cifar-10-batches-py")

    # # List of batch file names
    # batch_files = [f"data_batch_{i}" for i in range(1, 6)]  # Adjust if you have more than 4 batches

    # # Initialize empty lists to store data and labels
    # all_data = []
    # all_labels = []

    # # Load each batch and append data & labels
    # for batch_file in batch_files:
    #     data_path = os.path.join(datasets_folder, batch_file)
    #     dt = load_binary_dt(data_path)
        
    #     all_data.append(dt[b'data'])  # Image data
    #     all_labels.append(dt[b'labels'])  # Labels

    # # Convert lists to numpy arrays
    # X = np.concatenate(all_data, axis=0)  # Stack all image data
    # y = np.concatenate(all_labels, axis=0)  # Stack all labels

    
    # File path
    data_path = os.path.join(datasets_folder, "data_batch_1")

    dt = load_binary_dt(data_path)
    
    # Extract label
    labels = dt['labels']
    labels = np.array(labels)
    

    # Extract features 
    features = dt['data']
    breakpoint()
    # # Filter for specific labels
    # desired_labels = [0,2,3]   # ['airplane', automobile, bird, cat, deer, dog, frog, horse, ship, truck]
    desired_labels = [0,1,2,3,4, 5,6,7, 8,9]   # ['airplane', automobile, bird, cat, deer, dog, frog, horse, ship, truck]

    mask = np.isin(labels, desired_labels)
    features = features[mask]
    labels = labels[mask]
    
    # Normalize the features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # Remap the labels to consecutive integers
    label_mapping = {original: new for new, original in enumerate(desired_labels, start=0)}
    labels = np.array([label_mapping[label] for label in labels])
    if sample_size is not None:
        # Stratified sampling to maintain class ratios
        normalized_features, _, labels, _ = train_test_split(
            normalized_features,
            labels,
            train_size=sample_size,
            stratify=labels,
            random_state=random_state
        )

    # Count samples for each class
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Number of samples for each class:")
    for label, count in zip(unique_labels, counts):
        print(f"Class {int(label)}: {count} samples")
    
    return normalized_features, labels


def wine_dt():
    # Load the digits dataset
    digits = load_wine()

    # Features and target
    X = digits.data
    y = digits.target

    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X)

    # # Standardize the features
    # scaler = StandardScaler()
    # X_stand = scaler.fit_transform(X)
    return X_normalized, y

def breast_cancer_dt():
    # Load the digits dataset
    digits = load_breast_cancer()

    # Features and target
    X = digits.data
    y = digits.target

    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X)

    # # Standardize the features
    # scaler = StandardScaler()
    # X_stand = scaler.fit_transform(X)
    return X_normalized, y

def MNIST():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target  # X: images, y: labels
    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X)

    return X_normalized, y

def linnerud():
    linnerud = load_linnerud()
    X, y = linnerud.data, linnerud.target  # X: images, y: labels
    
    # Normalize the features
    normalizer = MinMaxScaler()
    X_normalized = normalizer.fit_transform(X)

    return X_normalized, y

