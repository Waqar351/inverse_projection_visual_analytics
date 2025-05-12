from utility import *
import os
import argparse




# Argument Parser
parser = argparse.ArgumentParser(description="Select dimensionality reduction technique: t-SNE or UMAP.")
parser.add_argument(
    "--method",
    type=str,
    choices=["tsne", "umap", "pca"],
    required=True,
    help="Choose 'tsne' or 'umap' or 'pca' for dimensionality reduction."
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["iris", "gaussian"],
    required=True,
    help="Choose 'iris' or 'gaussian' dataset for dimensionality reduction."
)
args = parser.parse_args()

prj_met_hd_ld_fold = f"thesis_reproduced/testing_new/temp/{args.dataset}/{args.method}_plots/hd_ld_metrics"
prj_met_hd_ld_filepath = f"{prj_met_hd_ld_fold}/{args.method}_prj_metrics_hd_ld.pkl"

if os.path.exists(prj_met_hd_ld_filepath):
    # Load precomputed metrics
    print(f"Loading precomputed metrics from {prj_met_hd_ld_filepath}...")
    prj_metrics_hd_to_ld = load_metrics(prj_met_hd_ld_filepath)
else:
    print('file not exits')