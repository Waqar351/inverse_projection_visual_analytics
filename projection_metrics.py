from zadu_measure import *

# class ProjectionMetrics:
#     def __init__(self):
#         self.metrics = {}

#     def add_metric(self, metric_name, value):
#         if metric_name not in self.metrics:
#             self.metrics[metric_name] = []
#         self.metrics[metric_name].append(value)

#     # def get_metric(self, metric_name):
#     #     return self.metrics.get(metric_name, [])
    
#     def get_metric(self, metric_name):
#         if metric_name not in self.metrics:
#             raise KeyError(f"The specified key '{metric_name}' does not exist in the metrics dictionary.")
#         return self.metrics.get(metric_name)
    
#     def __repr__(self):
#         return f"ProjectionMetrics(metrics={self.metrics})"


# def calculate_projection_metrics(X, X_emb, c, n_neighbors):
#     """Calculate projection quality metrics."""
#     metrics = {}

#     # Aggregate metrics from each function
#     metrics.update(measure_tnc(X, X_emb, k=n_neighbors, knn_ranking_info=None, return_local=False))
#     metrics.update(measure_nh(X_emb, c, k=n_neighbors, knn_emb_info=None, return_local=False))
#     metrics.update(measure_mrre(X, X_emb, k=n_neighbors, knn_ranking_info=None, return_local=False))
#     metrics.update(measure_srcc(X, X_emb, distance_matrices=None))
#     metrics.update(measure_pcc(X, X_emb, distance_matrices=None))
#     metrics.update(measure_dtm(X, X_emb, sigma=0.1, distance_matrices=None))
#     metrics.update(measure_catnc(X, X_emb, c, k=n_neighbors, knn_ranking_info=None, return_local=False))
#     metrics.update(measure_cevmv(X_emb, c, measure="arand", clustering="kmeans", clustering_args=None))
#     metrics.update(measure_dsc(X_emb, c))
#     metrics.update(measure_ivmv(X_emb, c, measure="silhouette"))
#     metrics.update(measure_kl(X, X_emb, sigma=0.1, distance_matrices=None))
#     metrics.update(measure_ltnc(X, X_emb, c, cvm="dsc"))
#     metrics.update(measure_lcmc(X, X_emb, k=n_neighbors, knn_info=None, return_local=False))
#     metrics.update(measure_nd(X, X_emb, k=n_neighbors, snn_info=None, knn_info=None))
#     metrics.update(measure_nms(X, X_emb, distance_matrices=None))
#     metrics.update(measure_procrustes(X, X_emb, k=n_neighbors, knn_info=None))
#     metrics.update(measure_stress(X, X_emb, distance_matrices=None))
#     metrics.update(measure_norm_stress(X, X_emb, distance_matrices=None))
#     metrics.update(measure_topographic(X, X_emb, k=n_neighbors, distance_matrices=None, knn_info=None))
#     metrics.update(projection_precision_score_v1(X, X_emb, n_neighbors=n_neighbors))
#     metrics.update(projection_precision_score_common_neig(X, X_emb, n_neighbors=n_neighbors))
#     metrics.update(average_local_error(X, X_emb))

#     return metrics


######################## Above is correct#####################

# class jacobian_norm_processing

class ProjectionMetrics:
    def __init__(self):
        # Nested dictionary to hold metrics for each value of k
        self.metrics = {}

    def add_metric(self, k, metric_name, value):
        if k not in self.metrics:
            self.metrics[k] = {}  # Initialize a new dictionary for this k if not already present
        if metric_name not in self.metrics[k]:
            self.metrics[k][metric_name] = []  # Initialize an empty list for this metric
        self.metrics[k][metric_name].append(value)  # Append value to the corresponding metric

    def get_metric(self, k, metric_name):
        """Retrieve a specific metric for a given k."""
        if k not in self.metrics or metric_name not in self.metrics[k]:
            raise KeyError(f"Metric '{metric_name}' not found for k={k}")
        return self.metrics[k][metric_name]

    def get(self, k):
        """Retrieve a specific metric for a given k."""
        if k not in self.metrics:
            raise KeyError(f"Metric not found for k={k}")
        return self.metrics[k]

    def __repr__(self):
        return f"ProjectionMetrics(metrics={self.metrics})"


def calculate_projection_metrics(X, X_emb, c, n_neighbors_list):
    """Calculate projection quality metrics for each value of n_neighbors."""
    metrics_dict = {}
    k_independent_metrics = {}

    # # Compute k-independent metrics only once
    k_independent_metrics.update(measure_srcc(X, X_emb, distance_matrices=None))
    k_independent_metrics.update(measure_pcc(X, X_emb, distance_matrices=None))
    k_independent_metrics.update(measure_dtm(X, X_emb, sigma=0.1, distance_matrices=None))
    k_independent_metrics.update(measure_cevmv(X_emb, c, measure="arand", clustering="kmeans", clustering_args=None))
    k_independent_metrics.update(measure_dsc(X_emb, c))
    k_independent_metrics.update(measure_ivmv(X_emb, c, measure="silhouette"))
    k_independent_metrics.update(measure_kl(X, X_emb, sigma=0.1, distance_matrices=None))
    ### k_independent_metrics.update(measure_ltnc(X, X_emb, c, cvm="dsc"))               #not used in final plots
    ### k_independent_metrics.update(measure_nms(X, X_emb, distance_matrices=None))    #not used in final plots
    k_independent_metrics.update(measure_stress(X, X_emb, distance_matrices=None))
    ### k_independent_metrics.update(measure_norm_stress(X, X_emb, distance_matrices=None))  #not used inf final plots
    k_independent_metrics.update(average_local_error(X, X_emb))
            

    # Compute k-dependent metrics for each k (n_neighbors)
    for k in n_neighbors_list:
        # Initialize an empty dictionary for each value of k
        metrics = {}

        ## Aggregate metrics from each function
        metrics.update(measure_tnc(X, X_emb, k=k, knn_ranking_info=None, return_local=False))
        metrics.update(measure_nh(X_emb, c, k=k, knn_emb_info=None, return_local=False))
        metrics.update(measure_mrre(X, X_emb, k=k, knn_ranking_info=None, return_local=False))
        metrics.update(measure_catnc(X, X_emb, c, k=k, knn_ranking_info=None, return_local=False))
        metrics.update(measure_lcmc(X, X_emb, k=k, knn_info=None, return_local=False))
        metrics.update(measure_nd(X, X_emb, k=k, snn_info=None, knn_info=None))
        metrics.update(measure_procrustes(X, X_emb, k=k, knn_info=None))
        metrics.update(measure_std_coh(X, X_emb, iteration=150, walk_num_ratio=0.3, alpha=0.1, k=k, clustering_strategy="dbscan", knn_info=None, return_local=False))
        ## metrics.update(measure_topographic(X, X_emb, k=k, distance_matrices=None, knn_info=None)) #not used inf final plots
        metrics.update(projection_precision_score_v1(X, X_emb, n_neighbors=k))
        metrics.update(neighborhood_preservation_precision(X, X_emb, n_neighbors=k))  

        # Combine k-independent metrics with k-dependent metrics
        metrics_dict[k] = {**metrics, **k_independent_metrics}

    return metrics_dict



# def calculate_projection_metrics(X, X_emb, c, n_neighbors):
#     """Calculate projection quality metrics."""
#     metrics = {
#         measure_tnc(X, X_emb, k=n_neighbors, knn_ranking_info=None, return_local=False),
#         measure_nh(X_emb, c, k = n_neighbors, knn_emb_info=None, return_local=False),
#         measure_mrre(X, X_emb, k = n_neighbors, knn_ranking_info=None, return_local=False),
#         measure_srcc(X, X_emb, distance_matrices=None),
#         measure_pcc(X, X_emb, distance_matrices=None),
#         measure_dtm(X, X_emb, sigma=0.1, distance_matrices=None),
#         measure_catnc(X, X_emb, c, k = n_neighbors, knn_ranking_info=None, return_local=False),
#         measure_cevmv(X_emb, c, measure="arand",  clustering="kmeans", clustering_args=None),
#         measure_dsc(X_emb, c),
#         measure_ivmv(X_emb, c, measure="silhouette"),
#         measure_kl(X, X_emb, sigma=0.1, distance_matrices=None),
#         measure_ltnc(X, X_emb, c, cvm="dsc"),
#         measure_lcmc(X, X_emb, k = n_neighbors, knn_info=None, return_local=False),
#         measure_nd(X, X_emb, k = n_neighbors, snn_info=None, knn_info=None),
#         measure_nms(X, X_emb,distance_matrices=None),
#         measure_procrustes(X, X_emb, k = n_neighbors, knn_info=None),
#         measure_stress(X, X_emb, distance_matrices=None),
#         measure_norm_stress(X, X_emb, distance_matrices=None) ,
#         measure_topographic(X, X_emb, k = n_neighbors, distance_matrices=None, knn_info=None) ,
#         projection_precision_score_v1(X, X_emb, n_neighbors=n_neighbors),    #close to 0 is good
#         projection_precision_score_common_neig(X, X_emb, n_neighbors=n_neighbors),  # close to 1 is good
#         average_local_error(X, X_emb),  # close to 0 is good

#     }
    # metrics = {
    #     "trustworth_continuity": measure_tnc(X, X_emb, k=n_neighbors, knn_ranking_info=None, return_local=False),
    #     'neighbour_hit' : measure_nh(X_emb, c, k = n_neighbors, knn_emb_info=None, return_local=False),
    #     'mean_relative_rank_error' : measure_mrre(X, X_emb, k = n_neighbors, knn_ranking_info=None, return_local=False),
    #     'spearman_rho' : measure_srcc(X, X_emb, distance_matrices=None),
    #     'pearson_r' : measure_pcc(X, X_emb, distance_matrices=None),
    #     'distance_to_measure' : measure_dtm(X, X_emb, sigma=0.1, distance_matrices=None),
    #     'class_aware_trustworthiness_continuity' : measure_catnc(X, X_emb, c, k = n_neighbors, knn_ranking_info=None, return_local=False),
    #     'clustering_and_external_validation_measure' : measure_cevmv(X_emb, c, measure="arand",  clustering="kmeans", clustering_args=None),
    #     'distance_consistency' : measure_dsc(X_emb, c),
    #     'internal_validation_measure' : measure_ivmv(X_emb, c, measure="silhouette"),
    #     'kl_divergence' : measure_kl(X, X_emb, sigma=0.1, distance_matrices=None),
    #     'label_trustworthiness_and_continuity' : measure_ltnc(X, X_emb, c, cvm="dsc"),
    #     'local_continuity_meta_criteria' : measure_lcmc(X, X_emb, k = n_neighbors, knn_info=None, return_local=False),
    #     'neighbor_dissimilarity' : measure_nd(X, X_emb, k = n_neighbors, snn_info=None, knn_info=None),
    #     'non_metric_stress' : measure_nms(X, X_emb,distance_matrices=None),
    #     'procrustes' :  measure_procrustes(X, X_emb, k = n_neighbors, knn_info=None),
    #     'stress' :  measure_stress(X, X_emb, distance_matrices=None),
    #     'scale_normalized_stress' :  measure_norm_stress(X, X_emb, distance_matrices=None) ,
    #     'topographic_product' :  measure_topographic(X, X_emb, k = n_neighbors, distance_matrices=None, knn_info=None) ,
    #     'projection_precision_score' :  projection_precision_score_v1(X, X_emb, n_neighbors=n_neighbors),    #close to 0 is good
    #     'projection_precision_score_common_neig' :  projection_precision_score_common_neig(X, X_emb, n_neighbors=n_neighbors),  # close to 1 is good
    #     'average_local_error' :  average_local_error(X, X_emb),  # close to 0 is good

    # }
    return metrics