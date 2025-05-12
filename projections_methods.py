from sklearn import manifold
from sklearn.decomposition import PCA
import umap


def get_reducer(method, perplexity):
    if method == "tsne":
            reducer = manifold.TSNE(n_components=2, perplexity=perplexity, init="random", random_state=0)
            method_name = "t-SNE"
            title_var = 'perplexity: '
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, n_neighbors=perplexity, min_dist=0.1, init="random", random_state=0)
        method_name = "UMAP"
        title_var = 'n_neighbors: '
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=0)
        method_name = "PCA"
        title_var = 'PCA: '
    
    return reducer, method_name, title_var