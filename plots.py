import numpy as np
import os
import torch
import plotly.graph_objects as go
from scipy.spatial.distance import squareform
import matplotlib
# print(matplotlib.get_backend()) 
# matplotlib.use('tkagg')
# print(matplotlib.get_backend()) 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utility import *
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
import itertools
import matplotlib.patches as patches

# colors = ['#FF0000', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF', '#0000FF', '#000000']
colors = ['#FF0000', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF', '#0000FF', '#000000', 
          '#FFA500', '#8000FF', '#FF1493']

save_format = "png"
dpi = 500


def spectral_norm_jacob_heatmap_vs_quality_metrics_plot_2Q_metrics(
    S, c, jacobian_norm, projection_metrics_hd_ld, projection_metrics_hd_hd,
    perplexity, n_gauss, selected_k, 
    x_min, x_max, y_min, y_max, clarity,
    method_name, title_var, output_folder=None):


    # colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#000000']  # Example colors
    # colors = ['#FFEB3B', '#1E88E5', '#D32F2F', '#8E24AA']
    if selected_k in projection_metrics_hd_ld:
        # Extract metrics for the selected k
        metrics_hd_ld = projection_metrics_hd_ld.get(selected_k)
        metrics_hd_hd = projection_metrics_hd_hd.get(selected_k)

        # Exclude specific metrics
        keys_to_exclude = ['projection_precision_score_common_neig', 'topographic_product', 'scale_normalized_stress',
                           'non_metric_stress', 'label_trustworthiness', 'label_continuity']
        metrics_hd_ld = {k: v for k, v in metrics_hd_ld.items() if k not in keys_to_exclude}
        metrics_hd_hd = {k: v for k, v in metrics_hd_hd.items() if k not in keys_to_exclude}

        # Create the figure
        fig = plt.figure(figsize=(20, 10))
        # Heatmap subplot
        ax_heatmap = plt.subplot2grid((1, 2), (0, 0))

        # Plot the Jacobian norm heatmap
        heatmap = ax_heatmap.imshow(
            jacobian_norm,
            extent=(x_min, x_max, y_min, y_max),
            origin='lower',
            cmap='hot',
            alpha=1.0
        )

        # Add scatter points for Gaussian clusters
        for i in range(n_gauss):
            ax_heatmap.scatter(S[c == i, 0], S[c == i, 1], color=colors[i],
                               label=f'Gaussian{i + 1}', zorder=3, edgecolor='k')

        # Add colorbar and labels
        fig.colorbar(heatmap, ax=ax_heatmap, label='Spectral Norm of Jacobian')
        ax_heatmap.set_title(f"{title_var} {perplexity} , clarity_score: {clarity}")
        ax_heatmap.set_xlabel(f"{method_name} Dimension 1")
        ax_heatmap.set_ylabel(f"{method_name} Dimension 2")

        # Combined metrics subplot
        ax_metrics = plt.subplot2grid((1, 2), (0, 1))

        # Prepare data for table
        metric_names = sorted(set(metrics_hd_ld.keys()).union(metrics_hd_hd.keys()))
        table_data = [["Quality Metrics", "HD to LD", "HD to HD"]]  # Table header

        
        for name in metric_names:
            value_hd_ld = metrics_hd_ld.get(name, "N/A")
            value_hd_hd = metrics_hd_hd.get(name, "N/A")
            
            # Format values for display
            if isinstance(value_hd_ld, list):
                value_hd_ld = ", ".join([f"{float(v):.3f}" if isinstance(v, (int, float)) else str(v) for v in value_hd_ld])
            elif isinstance(value_hd_ld, (int, float)):
                value_hd_ld = f"{float(value_hd_ld):.3f}"
            
            if isinstance(value_hd_hd, list):
                value_hd_hd = ", ".join([f"{float(v):.3f}" if isinstance(v, (int, float)) else str(v) for v in value_hd_hd])
            elif isinstance(value_hd_hd, (int, float)):
                value_hd_hd = f"{float(value_hd_hd):.3f}"
            
            # Add row to table data
            table_data.append([name, value_hd_ld, value_hd_hd])

        ###___________ Custom reorder table rows__________________________
        # List of names to highlight
        names_to_highlight = ["new_metric", "lcmc", "neighbor_dissimilarity", "trustworthiness", "continuity","stress", "kmeans_arand", "distance_to_measure", "silhouette"]
        # Reorder rows in table_data based on names_to_order
        ordered_table_data = [table_data[0]]  # Start with the header row

        # Add rows that match the names_to_order, preserving their order
        ordered_table_data.extend(
            [row for name in names_to_highlight for row in table_data if row[0] == name]
        )

        # Add any remaining rows not included in names_to_order
        ordered_table_data.extend(
            [row for row in table_data if row not in ordered_table_data]
        )

        # Replace the original table_data with the ordered one
        table_data = ordered_table_data

        
        ##################################################################


        # Create table
        table = ax_metrics.table(
            cellText=table_data,
            colLabels=None,
            cellLoc='center',
            loc='center'
        )

        # Adjust table appearance to fill the subplot
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.0, 1.7)  # Scale the table (width, height) to fill the subplot region

        # Format cells to show only outer borders
        for i, row in enumerate(table_data):
            for j, _ in enumerate(row):
                cell = table[(i, j)]  # Access each cell in the table

                # First row: Full border (header row)
                if i == 0:
                    cell.visible_edges = "TBLR"
                    # Set font to bold for header row
                    cell.set_text_props(fontweight="bold", color="black", ha="center")
                # Other rows: Only left and right borders
                elif i == len(table_data) - 1:  # Last row
                    cell.visible_edges = "BLR"  # No top border for the last row
                else:
                    cell.visible_edges = "LR"  # Only left and right borders

                # Format text for first column and other columns
                if j == 0:
                    cell.set_text_props(fontweight="bold", color="black", ha="right")
                else:
                    cell.set_text_props(color="black", ha="center")

        # Make column widths adapt to text size
        table.auto_set_column_width([0, 1, 2])
        ax_metrics.axis('tight')
        ax_metrics.axis('off')

        #####_________________________________________________________________________________
        # List of names to highlight
        # names_to_highlight = ["new_metric", "lcmc", "neighbor_dissimilarity", "trustworthiness", "continuity"]

        # # Highlight rows where the metric name matches
        # for row_idx, row_data in enumerate(table_data):
        #     if row_idx == 0:
        #         continue  # Skip the header row
        #     if row_data[0] in names_to_highlight:  # Check if the metric name matches
        #         for col_idx in range(len(row_data)):  # Iterate through all columns in the row
        #             cell = table[row_idx, col_idx]  # Access the specific cell
        #             cell.set_facecolor("#FFA500")  # Apply peach background color
        #             cell.set_text_props(color="black", weight="bold")  # Set text color to black and bold

                # Highlight rows where the metric name matches
        # for row_idx, row_data in enumerate(table_data):
        #     if row_idx == 0:
        #         continue  # Skip the header row
        #     if row_data[0] in names_to_highlight:  # Check if the metric name matches
        #         for col_idx in range(len(row_data)):  # Iterate through all columns in the row
        #             cell = table[row_idx, col_idx]  # Access the specific cell
        #             cell.set_facecolor("#FFA500")  # Apply peach background color
        #             cell.set_edgecolor("black")  # Optional: Set edge color for better visibility
        #             cell.set_text_props(color="black", weight="bold")  # Set text to black and bold

        # # Hide axes for the table

        # Highlight rows where the metric name matches
        for row_idx, row_data in enumerate(table_data):
            if row_idx == 0:  # Skip the header row
                continue
            if row_data[0] in names_to_highlight:  # Check if the metric name matches
                for col_idx in range(len(row_data)):
                    cell = table[row_idx, col_idx]
                    cell.set_facecolor("#FFA500")  # Set the background color
                    cell.set_alpha(1.0)  # Ensure full opacity
                    cell.set_edgecolor("black")
                    cell.set_text_props(color="black", weight="bold")  # Make text bold

        # table.auto_set_font_size(False)  # Ensure fonts are consistent
        # table.auto_set_column_width(col=list(range(len(table_data[0]))))  # Adjust column widths

        ###__________________Custom reorder__________________________________________________________
        

        ################################################################################################
        
        ##Save the plot only if output_folder is specified
        if output_folder:
            output_path = os.path.join(output_folder, f"jacobian_heatmap_{method_name}_{perplexity}_vs_quality_metrics.png")
            # table.auto_set_font_size(False)
            # table.auto_set_column_width(col=list(range(len(table_data[0]))))
            plt.draw()
            plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
            
            
            plt.close()
            # print(f"Results saved to {output_path}")
        else:
                print("Output folder not specified. Results not saved.")
                plt.show()
                plt.close()
    else:
        raise ValueError("Selected neighbour is not in projection metric dictionary.")



# def spectral_norm_jacob_heatmap_vs_quality_metrics_plot_2Q_metrics(
#     S, c, jacobian_norm, projection_metrics_hd_ld, projection_metrics_hd_hd,
#     perplexity, n_gauss, selected_k, 
#     x_min, x_max, y_min, y_max, clarity,
#     method_name, title_var, output_folder=None):

#     colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#000000']  # for testing

#     if selected_k in projection_metrics_hd_ld:
#         # Extract metrics for the selected k
#         metrics_hd_ld = projection_metrics_hd_ld.get(selected_k)
#         metrics_hd_hd = projection_metrics_hd_hd.get(selected_k)

#         # Exclude specific metrics
#         keys_to_exclude = ['projection_precision_score_common_neig', 'topographic_product', 'scale_normalized_stress',
#                            'non_metric_stress', 'label_trustworthiness', 'label_continuity']
#         metrics_hd_ld = {k: v for k, v in metrics_hd_ld.items() if k not in keys_to_exclude}
#         metrics_hd_hd = {k: v for k, v in metrics_hd_hd.items() if k not in keys_to_exclude}

#         # Create the figure
#         fig = plt.figure(figsize=(25, 10))
#         # Heatmap subplot
#         ax_heatmap = plt.subplot2grid((1, 2), (0, 0))

#         # Plot the Jacobian norm heatmap
#         heatmap = ax_heatmap.imshow(
#             jacobian_norm,
#             extent=(x_min, x_max, y_min, y_max),
#             origin='lower',
#             cmap='hot',
#             alpha=1.0
#         )

#         # Add scatter points for Gaussian clusters
#         for i in range(n_gauss):
#             ax_heatmap.scatter(S[c == i, 0], S[c == i, 1], color=colors[i],
#                                label=f'Gaussian{i + 1}', zorder=3, edgecolor='k')

#         # Add colorbar and labels
#         fig.colorbar(heatmap, ax=ax_heatmap, label='Spectral Norm of Jacobian')
#         ax_heatmap.set_title(f"{title_var} {perplexity} 'clarity_score: {clarity}")
#         ax_heatmap.set_xlabel(f"{method_name} Dimension 1")
#         ax_heatmap.set_ylabel(f"{method_name} Dimension 2")

#         # Combined metrics subplot
#         ax_metrics = plt.subplot2grid((1, 2), (0, 1))

#         # Prepare data for table
#         metric_names = sorted(set(metrics_hd_ld.keys()).union(metrics_hd_hd.keys()))
#         table_data = [["Metric Name", "HD to LD", "HD to HD"]]  # Table header

#         for name in metric_names:
#             value_hd_ld = metrics_hd_ld.get(name, "N/A")
#             value_hd_hd = metrics_hd_hd.get(name, "N/A")
            
#             # Format values for display
#             if isinstance(value_hd_ld, list):
#                 value_hd_ld = ", ".join([f"{float(v):.3f}" if isinstance(v, (int, float)) else str(v) for v in value_hd_ld])
#             elif isinstance(value_hd_ld, (int, float)):
#                 value_hd_ld = f"{float(value_hd_ld):.3f}"
            
#             if isinstance(value_hd_hd, list):
#                 value_hd_hd = ", ".join([f"{float(v):.3f}" if isinstance(v, (int, float)) else str(v) for v in value_hd_hd])
#             elif isinstance(value_hd_hd, (int, float)):
#                 value_hd_hd = f"{float(value_hd_hd):.3f}"
            
#             # Add row to table data
#             table_data.append([name, value_hd_ld, value_hd_hd])

#         # Create table
#         table = ax_metrics.table(
#             cellText=table_data,
#             colLabels=None,
#             cellLoc='center',
#             loc='center'
#         )

#         # Adjust table appearance to fill the subplot
#         table.auto_set_font_size(False)
#         table.set_fontsize(12)
#         table.scale(1.0, 2.0)  # Scale the table (width, height) to fill the subplot region

#         # Format cells to show only outer borders and header row border
#         for i, row in enumerate(table_data):
#             for j, _ in enumerate(row):
#                 cell = table[(i, j)]  # Access each cell in the table

#                 # Header row: full border
#                 if i == 0:
#                     cell.visible_edges = "TBLR"  # Full border for the header row
#                 # Data rows: only left and right borders, no top border for last row
#                 elif i == len(table_data) - 1:
#                     cell.visible_edges = "LR"  # Left and right borders only
#                 else:
#                     cell.visible_edges = "LR"  # Left and right borders only

#                 # Customize appearance for header row
#                 if i == 0:
#                     cell.set_text_props(fontweight="bold", color="black", ha="center")
#                 else:
#                     # Align the first column text to the right
#                     if j == 0:
#                         cell.set_text_props(color="black", ha="right")
#                     else:
#                         cell.set_text_props(color="black", ha="center")

#         # Hide axes for the table
#         ax_metrics.axis('tight')
#         ax_metrics.axis('off')

#         # Save the plot if output folder is provided
#         if output_folder:
#             plt.savefig(f"{output_folder}/heatmap_vs_metrics_{selected_k}.png", bbox_inches='tight')
#         # plt.show()



# def spectral_norm_jacob_heatmap_vs_quality_metrics_plot_2Q_metrics(
#     S, c, jacobian_norm, projection_metrics_hd_ld, projection_metrics_hd_hd,
#     perplexity, n_gauss, selected_k, 
#     x_min, x_max, y_min, y_max, clarity,
#     method_name, title_var, output_folder=None):

#     colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#000000']  # for testing

#     if selected_k in projection_metrics_hd_ld:
#         # Extract metrics for the selected k
#         metrics_hd_ld = projection_metrics_hd_ld.get(selected_k)
#         metrics_hd_hd = projection_metrics_hd_hd.get(selected_k)

#         # Exclude specific metrics
#         keys_to_exclude = ['projection_precision_score_common_neig', 'topographic_product', 'scale_normalized_stress',
#                            'non_metric_stress', 'label_trustworthiness', 'label_continuity']
#         metrics_hd_ld = {k: v for k, v in metrics_hd_ld.items() if k not in keys_to_exclude}
#         metrics_hd_hd = {k: v for k, v in metrics_hd_hd.items() if k not in keys_to_exclude}

#         # Create the figure
#         fig = plt.figure(figsize=(25, 10))
#         # Heatmap subplot
#         ax_heatmap = plt.subplot2grid((1, 2), (0, 0))

#         # Plot the Jacobian norm heatmap
#         heatmap = ax_heatmap.imshow(
#             jacobian_norm,
#             extent=(x_min, x_max, y_min, y_max),
#             origin='lower',
#             cmap='hot',
#             alpha=1.0
#         )

#         # Add scatter points for Gaussian clusters
#         for i in range(n_gauss):
#             ax_heatmap.scatter(S[c == i, 0], S[c == i, 1], color=colors[i],
#                                label=f'Gaussian{i + 1}', zorder=3, edgecolor='k')

#         # Add colorbar and labels
#         fig.colorbar(heatmap, ax=ax_heatmap, label='Spectral Norm of Jacobian')
#         ax_heatmap.set_title(f"{title_var} {perplexity} 'clarity_score: {clarity}")
#         ax_heatmap.set_xlabel(f"{method_name} Dimension 1")
#         ax_heatmap.set_ylabel(f"{method_name} Dimension 2")

#         # Combined metrics subplot
#         ax_metrics = plt.subplot2grid((1, 2), (0, 1))

#         # Format metrics for display
#         metric_text_hd_ld = "\n".join(
#             [f"$\\bf{{{name.replace('_', '\\_')}}}$: {', '.join([f'{float(v):.3f}' if isinstance(v, (int, float)) else str(v) for v in value])}"
#              if isinstance(value, list)
#              else f"$\\bf{{{name.replace('_', '\\_')}}}$: {float(value):.3f}" for name, value in metrics_hd_ld.items()]
#         )

#         metric_text_hd_hd = "\n".join(
#             [f"$\\bf{{{name.replace('_', '\\_')}}}$: {', '.join([f'{float(v):.3f}' if isinstance(v, (int, float)) else str(v) for v in value])}"
#              if isinstance(value, list)
#              else f"$\\bf{{{name.replace('_', '\\_')}}}$: {float(value):.3f}" for name, value in metrics_hd_hd.items()]
#         )

#         ax_metrics.text(0.05, 0.95, "HD to LD", fontweight='bold', fontsize=16, color='blue')
#         ax_metrics.text(0.20, 0.90, metric_text_hd_ld, fontsize=14, va='top', ha='right', linespacing=1.2)
#         ax_metrics.text(0.55, 0.95, "HD to HD", fontweight='bold', fontsize=16, color='blue')
#         ax_metrics.text(0.65, 0.90, metric_text_hd_hd, fontsize=14, va='top', ha='right', linespacing=1.2)

#         ax_metrics.axis("off")

#         # Save the plot if output_folder is specified
#         if output_folder:
#             output_path = os.path.join(output_folder, f"jacobian_heatmap_{method_name}_{perplexity}_vs_quality_metrics.png")
#             plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
#             plt.close()
#         else:
#             plt.show()
#     else:
#         raise ValueError("Selected neighbour is not in projection metric dictionary.")



# def spectral_norm_jacob_heatmap_vs_quality_metrics_plot_2Q_metrics(S,c,jacobian_norm, projection_metrics_hd_ld, projection_metrics_hd_hd,
#                                                         perplexity, n_gauss, selected_k, 
#                                                         x_min, x_max, y_min, y_max, clarity,
#                                                         method_name, title_var, output_folder = None):
    
#     # colors = ['#FFFF00', '#00FFFF', '#000000', '#FF00FF', '#00FF00', '#FFA500', '#800080', '#008080', '#00FF7F', '#FFC0CB', '#008080']
#     colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#000000']  # only for testing


#     if selected_k in projection_metrics_hd_ld:
#         # Extract metrics for the selected k
#         # metrics_hd_ld = projection_metrics_hd_ld[selected_k]
#         metrics_hd_ld = projection_metrics_hd_ld.get(selected_k)

        
#         # metrics_hd_hd = projection_metrics_hd_hd[selected_k]
#         metrics_hd_hd = projection_metrics_hd_hd.get(selected_k)

#         # breakpoint()
#         # List of keys to exclude
#         keys_to_exclude = ['projection_precision_score_common_neig', 'topographic_product', 'scale_normalized_stress',
#                            'non_metric_stress', 'label_trustworthiness', 'label_continuity' ]

#         # Filtered dictionary
#         metrics_hd_ld = dict(filter(lambda item: item[0] not in keys_to_exclude, metrics_hd_ld.items()))
#         metrics_hd_hd = dict(filter(lambda item: item[0] not in keys_to_exclude, metrics_hd_hd.items()))


#         # plt.rcParams['text.usetex'] = False  # Disable LaTeX for simplicity
#         # plt.rcParams['text.usetex'] = False
#         # plt.rcParams['font.family'] = 'sans-serif'
#         # Create a figure with two columns
#         fig, axes = plt.subplots(1, 3, figsize=(20, 7))
#         # Plot the heatmap
#         ax_heatmap = axes[0]

#         # Plot the Jacobian norm heatmap
#         heatmap  = ax_heatmap.imshow(jacobian_norm,
#             extent=(x_min, x_max, y_min, y_max),
#             origin='lower',
#             # cmap='hot',
#             # cmap='hot',
#             cmap='hot',
#             alpha=1.0,
#             # norm=matplotlib.colors.LogNorm()
#         )

#         for i in range(n_gauss):
#             # ax_heatmap.scatter(S[c == i, 0], S[c == i, 1], color=colors[i], label=f'Gaussian{i + 1}', edgecolor=None)
#             ax_heatmap.scatter(S[c == i, 0], S[c == i, 1], color=colors[i],
#                                label=f'Gaussian{i + 1}',
#                                zorder=3, edgecolor='k')

#         # Add colorbar for the heatmap
#         fig.colorbar(heatmap, ax=ax_heatmap, label='Spectral Norm of Jacobian')
#         ax_heatmap.set_title(f"{title_var}  {perplexity} 'clarity_score: {clarity}")
#         ax_heatmap.set_xlabel(f"{method_name} Dimension 1")
#         ax_heatmap.set_ylabel(f"{method_name} Dimension 2")

#         # # Add contour lines for decision boundaries
#         # contour = ax_heatmap.contour(jacobian_norm, levels=10, cmap='hot', alpha=0.7)
#         # ax_heatmap.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

#         # Display the metrics
#         ax_metrics = axes[1]
        
#         # metric_text = "\n".join(
#         # [f"{name}: {', '.join([f'{float(v):.5f}' if isinstance(v, (int, float)) else str(v) for v in value])}" 
#         # if isinstance(value, list) 
#         # else f"{name}: {float(value):.2f}" for name, value in metrics.items()]
#         # )

#         metric_text = "\n".join(
#         [f"$\\bf{{{name.replace('_', '\\_')}}}$: {', '.join([f'{float(v):.3f}' if isinstance(v, (int, float)) else str(v) for v in value])}"
#         if isinstance(value, list)
#         else f"$\\bf{{{name.replace('_', '\\_')}}}$: {float(value):.3f}" for name, value in metrics_hd_ld.items()]
#         )

#         # breakpoint()
#         # ax_metrics.text(0.5, 0.5, metric_text, ha="right", va="center", fontsize=12)
#         ax_metrics.text(0.7, 0.47, metric_text, ha="right", va="center", fontsize=12)
#         # ax_metrics.set_title(f"Quality Metrics (Perplexity {perplexity})")
#         ax_metrics.set_title(f"hd to ld", fontweight='bold', fontsize=14, color='blue')
#         ax_metrics.axis("off")

#         # Display the metrics
#         ax_metrics_2 = axes[2]
        
#         # metric_text = "\n".join(
#         # [f"{name}: {', '.join([f'{float(v):.5f}' if isinstance(v, (int, float)) else str(v) for v in value])}" 
#         # if isinstance(value, list) 
#         # else f"{name}: {float(value):.2f}" for name, value in metrics.items()]
#         # )

#         metric_text = "\n".join(
#         [f"$\\bf{{{name.replace('_', '\\_')}}}$: {', '.join([f'{float(v):.3f}' if isinstance(v, (int, float)) else str(v) for v in value])}"
#         if isinstance(value, list)
#         else f"$\\bf{{{name.replace('_', '\\_')}}}$: {float(value):.3f}" for name, value in metrics_hd_hd.items()]
#         )


#         # ax_metrics_2.text(0.7, 0.5, metric_text, ha="right", va="center", fontsize=12)
#         ax_metrics_2.text(0.7, 0.47, metric_text, ha="right", va="center", fontsize=12)
#         # ax_metrics_2.set_title(f"Quality Metrics (Perplexity {perplexity})")
#         ax_metrics_2.set_title(f"hd to hd",fontweight='bold', fontsize=14, color='blue')
#         ax_metrics_2.axis("off")
#     else:
#         raise ValueError("Selected neighbour is not in projection metric dictionary.")
    
#     # output_path = os.path.join(output_folder, f"jacobian_heatmap_{method_name}_{perplexity}_vs_quality_metrics.png")
#     # breakpoint()
#     # plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
    
#     # Save the plot only if output_folder is specified
#     if output_folder:
#         output_path = os.path.join(output_folder, f"jacobian_heatmap_{method_name}_{perplexity}_vs_quality_metrics.png")

#         plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
#         plt.close()
#         # print(f"Results saved to {output_path}")
#     else:
#         print("Output folder not specified. Results not saved.")
#         plt.close()


def spectral_norm_jacob_heatmap_vs_quality_metrics_plot(S,c,jacobian_norm, projection_metrics, 
                                                        perplexity, n_gauss, selected_k, 
                                                        x_min, x_max, y_min, y_max,
                                                        method_name, title_var, output_folder = None):
    
    # colors = ['#FFFF00', '#00FFFF', '#000000', '#FF00FF', '#00FF00', '#FFA500', '#800080', '#008080', '#00FF7F', '#FFC0CB', '#008080']
    # colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#000000']  # only for testing
    # colors = ['#FFEB3B', '#1E88E5', '#D32F2F', '#8E24AA']


    if selected_k in projection_metrics:
        # Extract metrics for the selected k
        metrics = projection_metrics[selected_k]

        # List of keys to exclude
        keys_to_exclude = ['projection_precision_score_common_neig', 'topographic_product', 'scale_normalized_stress',
                           'non_metric_stress', 'label_trustworthiness', 'label_continuity' ]

        # Filtered dictionary
        metrics = dict(filter(lambda item: item[0] not in keys_to_exclude, metrics.items()))


        # plt.rcParams['text.usetex'] = False  # Disable LaTeX for simplicity
        # plt.rcParams['text.usetex'] = False
        # plt.rcParams['font.family'] = 'sans-serif'
        # Create a figure with two columns
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        # Plot the heatmap
        ax_heatmap = axes[0]

        # Plot the Jacobian norm heatmap
        heatmap  = ax_heatmap.imshow(jacobian_norm,
            extent=(x_min, x_max, y_min, y_max),
            origin='lower',
            # cmap='hot',
            # cmap='hot',
            cmap='hot',
            alpha=1.0,
            # norm=matplotlib.colors.LogNorm()
        )

        for i in range(n_gauss):
            # ax_heatmap.scatter(S[c == i, 0], S[c == i, 1], color=colors[i], label=f'Gaussian{i + 1}', edgecolor=None)
            ax_heatmap.scatter(S[c == i, 0], S[c == i, 1], color=colors[i],
                               label=f'Gaussian{i + 1}',
                               zorder=3, edgecolor='k')

        # Add colorbar for the heatmap
        fig.colorbar(heatmap, ax=ax_heatmap, label='Spectral Norm of Jacobian')
        ax_heatmap.set_title(f"{method_name} {title_var}  {perplexity}")
        ax_heatmap.set_xlabel(f"{method_name} Dimension 1")
        ax_heatmap.set_ylabel(f"{method_name} Dimension 2")

        # # Add contour lines for decision boundaries
        # contour = ax_heatmap.contour(jacobian_norm, levels=10, cmap='hot', alpha=0.7)
        # ax_heatmap.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

        # Display the metrics
        ax_metrics = axes[1]
        
        # metric_text = "\n".join(
        # [f"{name}: {', '.join([f'{float(v):.5f}' if isinstance(v, (int, float)) else str(v) for v in value])}" 
        # if isinstance(value, list) 
        # else f"{name}: {float(value):.2f}" for name, value in metrics.items()]
        # )

        metric_text = "\n".join(
        [f"$\\bf{{{name.replace('_', '\\_')}}}$: {', '.join([f'{float(v):.5f}' if isinstance(v, (int, float)) else str(v) for v in value])}"
        if isinstance(value, list)
        else f"$\\bf{{{name.replace('_', '\\_')}}}$: {float(value):.5f}" for name, value in metrics.items()]
        )


        ax_metrics.text(0.7, 0.5, metric_text, ha="right", va="center", fontsize=12)
        ax_metrics.set_title(f"Quality Metrics (Perplexity {perplexity})")
        ax_metrics.axis("off")
    else:
        raise ValueError("Selected neighbour is not in projection metric dictionary.")
    
    # output_path = os.path.join(output_folder, f"jacobian_heatmap_{method_name}_{perplexity}_vs_quality_metrics.png")
    # breakpoint()
    # plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
    
    # Save the plot only if output_folder is specified
    if output_folder:
        output_path = os.path.join(output_folder, f"jacobian_heatmap_{method_name}_{perplexity}_vs_quality_metrics.png")

        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()
    
    
def plot_jacobian_spectral_norm_heatmap(S, c, jacobian_norm, n_gauss, x_min, x_max, y_min, y_max, method_name, title_var, perplexity, clarity = 0.2, output_path = None, figsize = (10,8) ):
    # fig, ax = plt.subplots(figsize=(10, 8))

    # fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    fig, ax = plt.subplots(figsize= figsize, constrained_layout=True)
    # ax.set_aspect('equal', adjustable='box')  # Ensures square axes

    # Plot the Jacobian norm heatmap
    heatmap = ax.imshow(
        jacobian_norm,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        cmap='hot',
        alpha=1.0
    )

    # Add scatter points for Gaussian clusters
    for i in range(n_gauss):
        ax.scatter(S[c == i, 0], S[c == i, 1], color=colors[i],
                            zorder=3, edgecolor='k', s = 15)

    # Add colorbar and labels
    # fig.colorbar(heatmap, ax=ax, label='Spectral Norm of Jacobian')
    # fig.colorbar(heatmap, ax=ax, shrink=1.0)
    # ax.set_title(f"{title_var} {perplexity} , clarity_score: {clarity}")
    # ax.set_xlabel(f"{method_name} Dimension 1", fontsize=14)
    # ax.set_ylabel(f"{method_name} Dimension 2", fontsize=14)
    # # ax.legend(loc='best', bbox_to_anchor=(1.05, 1), fontsize=14)

    # Remove x and y ticks but keep the box
    # ax = plt.gca()
    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    ax.axis('equal')
    plt.axis("off")

    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}" , dpi=dpi, format=save_format, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()
        

def plot_dimensionality_reduction(S, c, n_gauss, method_name, title_var, perplexity, clarity = 0.2, output_path=None, figsize = (10,10) ):
    # fig, ax = plt.subplots(figsize=(10, 8))
    # plt.figure(figsize=(10, 8))
    
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    # ax.set_aspect('equal', adjustable='box')  # Ensures square axes

    # Add scatter points for Gaussian clusters
    # colors = plt.cm.get_cmap('tab10', n_gauss)  # Use a colormap with n_gauss distinct colors
    for i in range(n_gauss):
        ax.scatter(S[c == i, 0], S[c == i, 1], color=colors[i],
                            zorder=3, edgecolor='k', s=100)

    # Add labels and title
    # plt.set_title(f"{title_var} {perplexity}, clarity_score: {clarity}")
    # plt.set_xlabel(f"{method_name} Dimension 1")
    # plt.set_ylabel(f"{method_name} Dimension 2")

    # Add legend
    # ax.legend()
    # ax.legend(fontsize=14, markerscale=1.5)
    ax.axis('equal')
    plt.axis("off")

    # Remove x and y ticks but keep the box
    # ax = plt.gca()
    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)


    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.png", dpi=300, format="png", bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()


def plot_jacobian_spectral_norm_heatmap_unsupervised(S, jacobian_norm, x_min, x_max, y_min, y_max, 
                                        method_name, title_var, perplexity, clarity = '', output_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot the Jacobian norm heatmap
    heatmap = ax.imshow(
        jacobian_norm,
        extent=(x_min, x_max, y_min, y_max),
        origin='upper',
        cmap='hot',
        alpha=1.0
    )

    # Scatter plot for data points (all in same color since no labels)
    ax.scatter(S[:, 0], S[:, 1], color='black', label='Data Points', zorder=3, edgecolor='k', s=10)

    # Add colorbar and labels
    fig.colorbar(heatmap, ax=ax, label='Spectral Norm of Jacobian')
    ax.set_title(f"{title_var} {perplexity} , clarity_score: {clarity}")
    ax.set_xlabel(f"{method_name} Dimension 1")
    ax.set_ylabel(f"{method_name} Dimension 2")

    # Save the plot only if a path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()

        


def spectral_norm_jacobian_heatmap_plot(S, c, n_gauss, num_grid_points, inverse_model, input_size, output_size,
                                        perplexity ,method_name, title_var, output_folder):

    ## Grid creation for Jacobian estimation
    x_min, x_max = np.min(S[:, 0]), np.max(S[:, 0])
    y_min, y_max = np.min(S[:, 1]), np.max(S[:, 1])
    x_vals = np.linspace(x_min, x_max, num_grid_points)
    y_vals = np.linspace(y_min, y_max, num_grid_points)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    ## Jacobian estimation
    jacobian_norms = np.zeros(len(grid_points))
    for idx, point in enumerate(grid_points):
        point_tensor = torch.tensor(point, dtype=torch.float32, requires_grad=True).view(1, 2)
        jacobian = torch.autograd.functional.jacobian(lambda x: inverse_model(x), point_tensor)
        jacobian_2d = jacobian.view(output_size, input_size)
        jacobian_norms[idx] = torch.linalg.norm(jacobian_2d, ord=2).item()

    jacobian_norms = jacobian_norms.reshape(xx.shape)
    

    # colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#000000']
    # colors = ['#FFEB3B', '#1E88E5', '#D32F2F', '#8E24AA']
    plt.figure(figsize=(10, 8))
    for i in range(n_gauss):
        plt.scatter(S[c == i, 0], S[c == i, 1], color=colors[i], label=f'Gaussian{i + 1}', edgecolor='k')
    plt.imshow(
        jacobian_norms,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        cmap='hot',
        alpha=1
    )
    # plt.colorbar(label='Spectral Norm of Jacobian')
    plt.title(f"{method_name} {title_var}  {perplexity}")
    plt.xlabel(f"{method_name} Dimension 1")
    plt.ylabel(f"{method_name} Dimension 2")
    # plt.legend()
    output_path = os.path.join(output_folder, f"jacobian_heatmap_{method_name}_{perplexity}.png")
    plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
    plt.close()


############________________ Metrics plot __________________________######################################################

def plot_metrics_vs_perplexity(perplex, metrics_dict, method_name,title_var, output_folder):
    """
    General function to plot multiple metrics dynamically for each k.

    Parameters:
    - perplex: List or array of perplexities.
    - metrics_dict: Dictionary where keys are k values and values are dictionaries of metrics.
    - method_name: The name of the Projection method (e.g., T-sne or UMAP).
    - output_folder: Folder where the plot image will be saved.
    """

    k_independent_metrics = ['spearman_rho', 'pearson_r', 'distance_to_measure', 'kmeans_arand',
                             'distance_consistency', 'silhouette', 'kl_divergence', 'label_trustworthiness',
                              'label_continuity', 'non_metric_stress', 'stress', 'scale_normalized_stress', 'average_local_error' ]
    # Iterate over each metric (it will be applied across all k values)
    for metric_name in next(iter(metrics_dict.values())).keys():  # Get the first sub-dictionary (metrics for a specific k)
        plt.figure(figsize=(10, 8))
        
        # Plot all k values for the current metric
        for k, k_metrics in metrics_dict.items():
            if metric_name in k_metrics:
                if metric_name not in k_independent_metrics:
                    plt.plot(perplex, k_metrics[metric_name], marker='o', label=f'k={k}')
                    plt.legend(title="k Values")
                else: 
                    plt.plot(perplex, k_metrics[metric_name], marker='o')   
        
        # Customize the plot
        # plt.title(f"{metric_name} vs Perplexity for {method_name}")
        plt.title(f"{method_name}")        
        plt.xlabel("Perplexities")
        plt.ylabel(metric_name)
        plt.grid(True)
        
        
        # Save the plot to the output folder
        output_path = os.path.join(output_folder, f"metric_vs_perplexity_{metric_name.lower().replace(' ', '_')}_{method_name}.png")
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()


def plot_3D_gaussian(D, c,n_gauss, colors, dataset, output_path = None, perplexity = ''):
        # Create a 3D scatter plot using Plotly
    fig = go.Figure()

    # Loop through each Gaussian to plot points with corresponding color
    for i in range(n_gauss):
        fig.add_trace(go.Scatter3d(
            x=D[c == i, 0], 
            y=D[c == i, 1], 
            z=D[c == i, 2], 
            mode='markers', 
            marker=dict(
                color=colors[i], 
                size=5,
                line=dict(
                    color='black',  # Set the edge color to black (you can change this)
                    width=1         # Set the edge width
                )
            ),
            name=f'Gaussian {i+1}'
        ))

    # Set labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis'
        ),
        title=f'3D Scatter Plot of Data Points from Gaussian Distributions {perplexity}',
        legend=dict(title="Gaussians")
    )

    # Save the plot only if path is specified
    if output_path:
        fig.write_html(output_path)
    else:
        fig.show()
        # fig.close()
        print("Output folder not specified. Results not saved.")
        

def plot_3D_gaussian_no_label(D, c,n_gauss, colors, dataset, output_path,  perplexity= '' ):
        # Create a 3D scatter plot using Plotly
    fig = go.Figure()

    # Loop through each Gaussian to plot points with corresponding color
    # for i in range(n_gauss):
    fig.add_trace(go.Scatter3d(
        x=D[:, 0], 
        y=D[:, 1], 
        z=D[:, 2], 
        mode='markers', 
        # marker=dict(color=colors[i], size=5),
        # name=f'Gaussian {i+1}'
    ))
    # ax.scatter(D[c == i, 0], D[c == i, 1], D[c == i, 2], color=colors[i], alpha=0.7, label=f'Gaussian {i+1}')

    # Set labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis'
        ),
        title= f'Grids projected to 3D. perplexity {perplexity}',
        legend=dict(title="Gaussians")
    )

    # Save the plot only if path is specified
    if output_path:
        fig.write_html(output_path)
    else:
        print("Output folder not specified. Results not saved.")

# def plot_3D_gaussian_no_label_with_projected_grids(girds_dt, orig_dt, c,n_gauss, colors, dataset, output_path,  perplexity= '' ):
#         # Create a 3D scatter plot using Plotly
#     fig = go.Figure()

#     # Loop through each Gaussian to plot points with corresponding color
#     # for i in range(n_gauss):
#     fig.add_trace(go.Scatter3d(
#         x=girds_dt[:, 0], 
#         y=girds_dt[:, 1], 
#         z=girds_dt[:, 2], 
#         mode='markers', 
#         # marker=dict(color=colors[i], size=5),
#         # name=f'Gaussian {i+1}'
#     ))
#     # Loop through each Gaussian to plot points with corresponding color
#     for i in range(n_gauss):
#         fig.add_trace(go.Scatter3d(
#             x=orig_dt[c == i, 0], 
#             y=orig_dt[c == i, 1], 
#             z=orig_dt[c == i, 2], 
#             mode='markers', 
#             marker=dict(
#                 color=colors[i], 
#                 size=5,
#                 line=dict(
#                     color='black',  # Set the edge color to black (you can change this)
#                     width=1         # Set the edge width
#                 )
#             ),
#             name=f'Gaussian {i+1}'
#         ))

#     # Set labels and title
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X-axis',
#             yaxis_title='Y-axis',
#             zaxis_title='Z-axis'
#         ),
#         title= f'Grids projected to 3D. perplexity {perplexity}',
#         legend=dict(title="Gaussians")
#     )

#     # Save the plot only if path is specified
#     if output_path:
#         fig.write_html(output_path)
#     else:
#         print("Output folder not specified. Results not saved.")

# import numpy as np
# import plotly.graph_objects as go

# def plot_3D_gaussian_no_label_with_projected_grids(grids_dt, orig_dt, c, n_gauss, colors, dataset, output_path, perplexity=''):
#     fig = go.Figure()

#     # Reshape the grid points into a structured mesh (assuming it's a structured grid)
#     grid_x = grids_dt[:, 0].reshape((-1, int(np.sqrt(len(grids_dt)))))
#     grid_y = grids_dt[:, 1].reshape((-1, int(np.sqrt(len(grids_dt)))))
#     grid_z = grids_dt[:, 2].reshape((-1, int(np.sqrt(len(grids_dt)))))

#     # Add a surface plot (continuous grid)
#     fig.add_trace(go.Surface(
#         x=grid_x, 
#         y=grid_y, 
#         z=grid_z,
#         colorscale='Viridis',
#         opacity=0.7
#     ))

#     # Add original points as scatter for reference
#     for i in range(n_gauss):
#         fig.add_trace(go.Scatter3d(
#             x=orig_dt[c == i, 0], 
#             y=orig_dt[c == i, 1], 
#             z=orig_dt[c == i, 2], 
#             mode='markers', 
#             marker=dict(color=colors[i], size=5, line=dict(color='black', width=1)),
#             name=f'Gaussian {i+1}'
#         ))

#     # Set labels and title
#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X-axis',
#             yaxis_title='Y-axis',
#             zaxis_title='Z-axis'
#         ),
#         title=f'Grids projected to 3D (Surface). Perplexity {perplexity}',
#         legend=dict(title="Gaussians")
#     )

#     # Save the plot only if path is specified
#     if output_path:
#         fig.write_html(output_path)
#     else:
#         print("Output folder not specified. Results not saved.")

def plot_3D_gaussian_no_label_with_projected_grids(grids_dt, orig_dt, c, n_gauss, colors, dataset, output_path, perplexity=''):
    fig = go.Figure()

    # Create line connections for wireframe structure
    for i in range(len(grids_dt) - 1):
        fig.add_trace(go.Mesh3d(
            x=grids_dt[:, 0],
            y=grids_dt[:, 1],
            z=grids_dt[:, 2],
            opacity=0.5,
            color='blue',
            alphahull=0  # Ensures the convex hull is drawn
        ))

    # Add original points for reference
    for i in range(n_gauss):
        fig.add_trace(go.Scatter3d(
            x=orig_dt[c == i, 0], 
            y=orig_dt[c == i, 1], 
            z=orig_dt[c == i, 2], 
            mode='markers', 
            marker=dict(color=colors[i], size=5, line=dict(color='black', width=1)),
            name=f'Gaussian {i+1}'
        ))

    # Set labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis'
        ),
        title=f'Grids projected to 3D (Wireframe). Perplexity {perplexity}',
        legend=dict(title="Gaussians")
    )

    # Save the plot only if path is specified
    if output_path:
        fig.write_html(output_path)
    else:
        print("Output folder not specified. Results not saved.")





def plot_mean_cluster_distance(distance_matrix, unique_clusters, custom_colors, perplexity= '', output_path=None):
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap="hot", interpolation="nearest")  # Grayscale heatmap
    plt.colorbar(label="Distance")
    plt.title(f"Inter-Cluster and Intra-Cluster Mean Distances {perplexity}")
    plt.xticks(ticks=np.arange(len(unique_clusters)), labels=[f"Cluster {int(i) +1}" for i in unique_clusters], fontsize=12, fontweight="bold")
    plt.yticks(ticks=np.arange(len(unique_clusters)), labels=[f"Cluster {int(i) +1}" for i in unique_clusters], fontsize=12, fontweight="bold")
    # plt.xlabel("Clusters")
    # plt.ylabel("Clusters")

    # Apply custom colors to axis labels
    ax = plt.gca()  # Get current axis
    for xtick, color in zip(ax.get_xticklabels(), custom_colors):
        xtick.set_color(color)
    for ytick, color in zip(ax.get_yticklabels(), custom_colors):
        ytick.set_color(color)

    # Annotate the heatmap with the mean distances from distance_matrix
    for i in range(len(unique_clusters)):  # Loop over rows
        for j in range(len(unique_clusters)):  # Loop over columns
            value = distance_matrix[i, j]  # Get the mean distance value
            plt.text(
                j,  # x-coordinate (column index)
                i,  # y-coordinate (row index)
                f"{value:.2f}",  # Format value to 2 decimal places
                color="black",  # Text color
                ha="center",  # Horizontal alignment
                va="center",  # Vertical alignment
                fontsize=12,  # Font size
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3", alpha=0.6)  # Text box styling
            )
       
    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()
    

def plot_pairwise_cluster_distance(distance_matrix, mean_cluster_distance, label, unique_clusters,colors, fig_title, perplexity = '', output_path=None):
    # Step 3: Calculate the start and end positions for cluster blocks
    num_points_per_cluster = [np.sum(label == i) for i in unique_clusters]
    cumulative_positions = np.cumsum([0] + num_points_per_cluster)  # Start and end positions for each cluster
    tick_positions = cumulative_positions[:-1] + np.diff(cumulative_positions) / 2  # Midpoints for cluster names

    # Step 4: Visualize the full pairwise distance matrix
    plt.figure(figsize=(15, 15))
    # plt.imshow(distance_matrix, cmap='hot', interpolation='nearest')  # Grayscale heatmap
    plt.imshow(distance_matrix, cmap='hot')  # Grayscale heatmap
    # plt.colorbar(label='Distance')
    plt.colorbar()
    # plt.title(f"{fig_title} {perplexity}")

    # # Add cluster names as ticks
    # cluster_names = [f"Cluster {int(i) +1}" for i in unique_clusters]
    # # plt.xticks(tick_positions, cluster_names, ha="right", fontsize=12, fontweight="bold")
    # plt.xticks(tick_positions, cluster_names, fontsize=18, fontweight="bold")
    # plt.yticks(tick_positions, cluster_names, fontsize=18, fontweight="bold")


    # # Set custom colors for axis labels
    # ax = plt.gca()  # Get current axis
    # for xtick, color in zip(ax.get_xticklabels(), colors):
    #     xtick.set_color(color)
    # for ytick, color in zip(ax.get_yticklabels(), colors):
    #     ytick.set_color(color)

    # Remove x and y ticks but keep the box
    ax = plt.gca()  # Get current axis
    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)


    # Step 4: Annotate the heatmap with mean distances for each cluster block
    for i, start_i in enumerate(cumulative_positions[:-1]):
        for j, start_j in enumerate(cumulative_positions[:-1]):
            # # Get the submatrix for the (i, j) block
            # block = distance_matrix[start_i:cumulative_positions[i + 1], start_j:cumulative_positions[j + 1]]
            # # Compute the mean of the block
            # mean_distance = np.mean(block)
            mean_distance = mean_cluster_distance[i,j]
            # breakpoint()
            # Annotate the mean distance at the center of the block
            plt.text(
                (start_j + cumulative_positions[j + 1]) / 2,  # x-coordinate (center of block)
                (start_i + cumulative_positions[i + 1]) / 2,  # y-coordinate (center of block)
                f"{mean_distance:.2f}",  # Format mean distance to 2 decimal places
                color="black",
                ha="center",
                va="center",
                fontsize=25,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3", alpha=0.6)  # Text box styling
            )
    
    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()
    

def plot_pairwise_cluster_distance_v2(distance_matrix, mean_cluster_distance, label, unique_clusters, colors, fig_title, perplexity = '', output_path=None, border_thickness=10, figsize=(10, 10)):
    # Compute cluster boundaries dynamically
    num_points_per_cluster = [np.sum(label == i) for i in unique_clusters]
    cumulative_positions = np.cumsum([0] + num_points_per_cluster)  # Start and end positions for each cluster
    
    # Create the heatmap plot
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_aspect(1)
    ax.axis('equal')
    plt.axis("off")
    im = ax.imshow(distance_matrix, cmap='hot')  # Heatmap visualization
    # plt.colorbar(im, ax=ax, fraction=0.05, pad=0.05) 
    # plt.colorbar(im, ax=ax) 

    # Remove x and y ticks but keep the box
    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines[:].set_visible(False)  # Hide all borders first

    # Draw only the outer left and outer top borders
    for i in range(len(unique_clusters)):
        pad_border_corner = 0
        start = cumulative_positions[i] - pad_border_corner
        end = cumulative_positions[i + 1] - pad_border_corner if i + 1 < len(cumulative_positions) else cumulative_positions[-1] - pad_border_corner

        color = colors[i]  # Assign cluster color

        # Left border (Outer vertical) - precisely aligned
        ax.plot([-0.5, -0.5], [start-0.5, end-0.5], color=color, linewidth=border_thickness, solid_capstyle='butt', clip_on=False)  

        # Top border (Outer horizontal) - precisely aligned
        ax.plot([start-0.5, end-0.5], [-0.5, -0.5], color=color, linewidth=border_thickness, solid_capstyle='butt', clip_on=False)  


    # Annotate mean distances inside cluster blocks
    for i, start_i in enumerate(cumulative_positions[:-1]):
        for j, start_j in enumerate(cumulative_positions[:-1]):
            mean_distance = mean_cluster_distance[i, j]
            center_x = (start_j + cumulative_positions[j + 1]) / 2
            center_y = (start_i + cumulative_positions[i + 1]) / 2
            ax.text(center_x, center_y, f"{mean_distance:.2f}", color="black", ha="center", va="center", fontsize=25,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3", alpha=0.6))

    # Adjust figure to remove empty space
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save or display the plot
    if output_path:
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_highlight_select_edges_on_distance_matrix(distance_matrix, highlight_points, label, unique_clusters, colors, fig_title = None, perplexity = '', output_path=None, border_thickness=10, figsize=(10, 10)):
    
    # Compute cluster boundaries dynamically
    num_points_per_cluster = [np.sum(label == i) for i in unique_clusters]
    cumulative_positions = np.cumsum([0] + num_points_per_cluster)  # Start and end positions for each cluster
    # breakpoint()
    # Create the heatmap plot
    # fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    ax.set_aspect(aspect='auto')
    ax.axis('equal')
    plt.axis("off")
    im = ax.imshow(distance_matrix, aspect='auto', cmap='hot', interpolation="nearest") 

    for row, col in highlight_points:  
        rect_size = 10  #rectangle size

        # Draw a larger rectangle around the selected point
        rect = patches.Rectangle((col - rect_size/2, row - rect_size/2), rect_size, rect_size, linewidth=3, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)

        # Since the matrix is symmetric, highlight both (row, col) and (col, row)
        if row != col:
            rect_sym = patches.Rectangle((row - rect_size/2, col - rect_size/2), rect_size, rect_size, linewidth=3, edgecolor='blue', facecolor='none')
            ax.add_patch(rect_sym)

    # Remove x and y ticks but keep the box
    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines[:].set_visible(False)  # Hide all borders first

    # Draw only the outer left and outer top borders
    for i in range(len(unique_clusters)):
        pad_border_corner = 0
        start = cumulative_positions[i] - pad_border_corner
        end = cumulative_positions[i + 1] - pad_border_corner if i + 1 < len(cumulative_positions) else cumulative_positions[-1] - pad_border_corner

        color = colors[i]  # Assign cluster color

        # Left border (Outer vertical) - precisely aligned
        ax.plot([-0.5, -0.5], [start-0.5, end-0.5], color=color, linewidth=border_thickness, solid_capstyle='butt', clip_on=False)  

        # Top border (Outer horizontal) - precisely aligned
        ax.plot([start-0.5, end-0.5], [-0.5, -0.5], color=color, linewidth=border_thickness, solid_capstyle='butt', clip_on=False)  


    # Save or display the plot
    if output_path:
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_highlight_select_edges_on_distance_matrix_v2(distance_matrix, highlight_points, label, unique_clusters, colors, fig_title = None, perplexity = '', output_path=None, border_thickness=10, figsize=(10, 10)):
    
        # Sort indices based on class labels
    sorted_indices = np.argsort(label)

    # Reorder dataset and labels
    # X_sorted = X[sorted_indices]
    y_sorted = label[sorted_indices]

    # 🔹 Reorder the Pairwise Distance Matrix
    # reindexed_matrix = distance_matrix[np.ix_(sorted_indices, sorted_indices)]

    # 🔹 Reorder BOTH rows and columns using sorted_indices
    reindexed_matrix = distance_matrix[sorted_indices, :][:, sorted_indices]

    # 🔹 Update the Highlighted Points (Find New Positions)
    # Convert old indices to new positions
    new_highlighted_positions = np.searchsorted(sorted_indices, highlight_points)


    ###_________________________________________
    
    # Compute cluster boundaries dynamically
    # num_points_per_cluster = [np.sum(label == i) for i in unique_clusters]
    num_points_per_cluster = [np.sum(y_sorted == i) for i in unique_clusters]
    cumulative_positions = np.cumsum([0] + num_points_per_cluster)  # Start and end positions for each cluster
    # breakpoint()
    # Create the heatmap plot
    # fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    # ax.set_aspect(aspect='auto')
    # ax.axis('equal')
    plt.axis("off")
    # im = ax.imshow(distance_matrix, aspect='auto', cmap='hot', interpolation="none")  # Heatmap visualization
    im = ax.imshow(reindexed_matrix, aspect='auto', cmap='hot', interpolation="none")  # Heatmap visualization
    # **Corrected Highlighting of Selected Points**
    # for row, col in highlight_points:  
    # for row, col in new_highlighted_positions:  
    #     rect_size = 10  # Increase rectangle size

    #     # Draw a larger rectangle around the selected point
    #     rect = patches.Rectangle((col - rect_size/2, row - rect_size/2), rect_size, rect_size, linewidth=3, edgecolor='blue', facecolor='none')
    #     ax.add_patch(rect)
    #     # breakpoint()
    #     # Since the matrix is symmetric, highlight both (row, col) and (col, row)
    #     if row != col:
    #         rect_sym = patches.Rectangle((row - rect_size/2, col - rect_size/2), rect_size, rect_size, linewidth=3, edgecolor='blue', facecolor='none')
    #         ax.add_patch(rect_sym)

    # # Remove x and y ticks but keep the box
    # ax.set_xticks([])  
    # ax.set_yticks([])
    # ax.spines[:].set_visible(False)  # Hide all borders first

    # Draw only the outer left and outer top borders
    for i in range(len(unique_clusters)):
        pad_border_corner = 0
        start = cumulative_positions[i] - pad_border_corner
        end = cumulative_positions[i + 1] - pad_border_corner if i + 1 < len(cumulative_positions) else cumulative_positions[-1] - pad_border_corner

        color = colors[i]  # Assign cluster color

        # Left border (Outer vertical) - precisely aligned
        ax.plot([-0.5, -0.5], [start-0.5, end-0.5], color=color, linewidth=border_thickness, solid_capstyle='butt', clip_on=False)  

        # Top border (Outer horizontal) - precisely aligned
        ax.plot([start-0.5, end-0.5], [-0.5, -0.5], color=color, linewidth=border_thickness, solid_capstyle='butt', clip_on=False)  


    # Save or display the plot
    if output_path:
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



def plot_hd_to_ld_clust_distance_difference(distance_matrix_hd, ld_distance_matrix, perplexity = '', output_path = None):
    # Symmetrize the  distance matrix
    final_distance_matrix_orig = (distance_matrix_hd + distance_matrix_hd.T) / 2
    final_distance_matrix = (ld_distance_matrix + ld_distance_matrix.T) / 2

    # Ensure diagonal is zero
    np.fill_diagonal(final_distance_matrix_orig, 0)
    np.fill_diagonal(final_distance_matrix, 0)

    # # Flatten the upper triangles of the matrices
    # original_distances_flat = squareform(final_distance_matrix_orig)
    # low_dim_distances_flat = squareform(final_distance_matrix)

    # Compute absolute difference matrix
    # difference_matrix = np.abs(final_distance_matrix_orig - final_distance_matrix)
    difference_matrix = np.abs(distance_matrix_hd - ld_distance_matrix)

    # Visualize with a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(difference_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Absolute Difference')
    plt.title(f'Distance Differences, perplexity: {perplexity}')
    plt.xlabel('Points')
    plt.ylabel('Points')

    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()


def plot_spectral_norm_vs_centroid_decision_boundaries(data, label,num_classes ,grid_points,  near_centroid_labels, centroids, jacob_norm,
                                                   x_min, x_max, y_min, y_max,
                                                   perplexity, clarity, colors, output_path = None ):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the scatter plot (t-SNE data with clusters and centers)
    for i in range(num_classes):
        ax.scatter(data[label == i, 0], data[label == i, 1], color=colors[i], alpha=0.7, label=f'Cluster {i+1}', edgecolor = 'k')
        ax.scatter(grid_points[near_centroid_labels == i, 0], grid_points[near_centroid_labels == i, 1], alpha=0.05, color=colors[i])

    # Mark cluster centers with 'X'
    ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centers')

    # Overlay the Jacobian spectral norm as a heatmap
    jacob_im = ax.imshow(
        jacob_norm,
        extent=(x_min, x_max, y_min, y_max),
        origin='lower',
        cmap='hot',
        alpha=1.0  # Adjust alpha to make sure both plots are visible
    )

    # Add colorbar for the heatmap
    cbar = plt.colorbar(jacob_im, ax=ax)
    cbar.set_label('Spectral Norm of Jacobian')

    # Set labels, title, and legend for the combined plot
    ax.set_xlabel(f'Dimension 1')
    ax.set_ylabel(f'Dimension 2')
    ax.set_title(f'perplexity: {perplexity} , clarity_score: {clarity}')
    # Position the legend outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1), title="Cluster Labels")
    plt.tight_layout()  # Adjust layout to prevent overlapping

    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()


def plot_proj_inverse_proj_metric_differences(data, output_path):
    df_diff = pd.DataFrame(data)

    # Pivot the DataFrame for heatmap
    heatmap_data = df_diff.pivot(index="metric", columns="perplexity", values="difference")
    # Normalize the differences for each metric independently
    heatmap_data_normalized = heatmap_data.apply(
        # lambda x: (x - x.min()) / (x.max() - x.min()), axis=1
        lambda x: hybrid_normalization(x)
    )

    metrics_optimal_one = ['new_metric', 'lcmc', 'kmeans_arand', 'distance_consistency','spearman_rho','pearson_r', 
                           'trustworthiness', 'continuity',  'ca_trustworthiness', 'ca_continuity', 'neighborhood_hit',
                           'mrre_false', 'mrre_missing', 'silhouette', 'steadiness', 'cohesiveness']
    metrics_optimal_zero= [ 'neighbor_dissimilarity',  'stress', 'distance_to_measure', 'projection_precision_score',   
                           'kl_divergence',  'procrustes', 'average_local_error']
    quality_metrics_order = metrics_optimal_one + metrics_optimal_zero
     # Reorder rows
    heatmap_data_normalized = heatmap_data_normalized.reindex(quality_metrics_order)
    # Plot heatmap
    plt.figure(figsize=(17, 6))
    sns.heatmap(heatmap_data_normalized, annot=False, cmap="cividis", cbar_kws={"label": "Absolute Difference"})
    plt.title("Metrics Differences Between HD to LD and HD to LD")
    plt.xlabel("Perplexities")
    # plt.ylabel("Metric")

    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()
    
def plot_metrics_perplexity_heatmap(data, perplexities, output_path, title = ''):
    metric_names = list(data.get(5).keys())

    ########
    metrics_optimal_one = ['new_metric', 'lcmc', 'kmeans_arand', 'distance_consistency','spearman_rho','pearson_r', 
                           'trustworthiness', 'continuity',  'ca_trustworthiness', 'ca_continuity', 'neighborhood_hit',
                           'mrre_false', 'mrre_missing', 'silhouette', 'steadiness', 'cohesiveness']
    metrics_optimal_zero= [ 'neighbor_dissimilarity',  'stress', 'distance_to_measure', 'projection_precision_score',   
                           'kl_divergence',  'procrustes', 'average_local_error']
    quality_metrics_order = metrics_optimal_one + metrics_optimal_zero

    #############

    # Normalize the values for each metric
    normalized_data = {}
    for metric in metric_names:
        values = [data[5][metric][i] for i in range(len(perplexities))]
        # normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))  # Min-max normalization
        # Apply Hybrid Normalization
        normalized_values = hybrid_normalization(values)  # Use hybrid normalization
        # Adjust metrics that are optimal at zero
        if metric in metrics_optimal_zero:
            normalized_values = 1 - normalized_values
        normalized_data[metric] = normalized_values
    
    # Create DataFrame for the heatmap
    heatmap_data = pd.DataFrame(normalized_data, index=perplexities).T
    heatmap_data.columns = [f"{p}" for p in perplexities]

    # Reorder rows
    heatmap_data = heatmap_data.reindex(quality_metrics_order)

    # Plot the heatmap
    plt.figure(figsize=(15, 6))
    sns.heatmap(
        heatmap_data,
        annot=False,  # Optionally add `annot=True` to show values
        cmap="cividis",  # Colormap; change to suit your preferences
        # cbar_kws={"label": "Normalized Metric Value"},
        cbar_kws={"label": "Absolute Difference"}
    )
    plt.title(f"Normalized Metric Values Across Perplexities for {title}")
    plt.xlabel("Perplexity")
    # plt.ylabel("Metrics")
    plt.tight_layout()

    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()


def plot_metrics_vs_rate_change_perplexity_heatmap(data, perplexities, output_path, title=''):
    # Define metric names and optimal lists
    metric_names = list(data.get(5).keys())
    metrics_optimal_one = ['new_metric', 'lcmc', 'kmeans_arand', 'distance_consistency','spearman_rho','pearson_r', 
                           'trustworthiness', 'continuity', 'ca_trustworthiness', 'ca_continuity', 'neighborhood_hit',
                           'mrre_false', 'mrre_missing', 'silhouette', 'steadiness', 'cohesiveness']
    metrics_optimal_zero = ['neighbor_dissimilarity', 'stress', 'distance_to_measure', 'projection_precision_score',   
                           'kl_divergence', 'procrustes', 'average_local_error']
    quality_metrics_order = metrics_optimal_one + metrics_optimal_zero

    # Normalize the metric values
    normalized_data = {}
    rate_of_change_data = {}  # To store the rate of change for each metric
    
    for metric in metric_names:
        # Extract metric values for all perplexities
        values = [data[5][metric][i] for i in range(len(perplexities))]
        
        # Min-max normalization of metric values
        # normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
        # Apply Hybrid Normalization
        normalized_values = hybrid_normalization(values)  # Use hybrid normalization
        
        # Adjust metrics that are optimal at zero
        if metric in metrics_optimal_zero:
            normalized_values = 1 - normalized_values
        
        normalized_data[metric] = normalized_values
        
        # Calculate the rate of change (difference between consecutive values)
        rate_of_change = np.diff(normalized_values)
        rate_of_change_data[metric] = np.concatenate([[0], rate_of_change])  # Pad with 0 for first value to keep same length
    
    # Create DataFrame for the heatmap of metric values
    heatmap_data = pd.DataFrame(normalized_data, index=perplexities).T
    heatmap_data.columns = [f"{p}" for p in perplexities]

    # Create DataFrame for the heatmap of rate of change values
    rate_of_change_df = pd.DataFrame(rate_of_change_data, index=perplexities).T
    rate_of_change_df.columns = [f"{p}" for p in perplexities]
    
    # Reorder rows based on the quality metrics
    heatmap_data = heatmap_data.reindex(quality_metrics_order)
    rate_of_change_df = rate_of_change_df.reindex(quality_metrics_order)

    # Create a figure with two subplots: one for metric values and one for rate of change
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot the heatmap for Metric Values
    sns.heatmap(
        heatmap_data,
        annot=False,  # Optionally add `annot=True` to show values
        cmap="cividis",  # Colormap for the metric values
        cbar_kws={"label": "Normalized Metric Value"},
        ax=axes[0]
    )
    axes[0].set_title(f"Normalized Metric Values Across Perplexities for {title}")
    axes[0].set_xlabel("Perplexity")
    axes[0].set_ylabel("Metric")

    # Plot the heatmap for Rate of Change
    sns.heatmap(
        rate_of_change_df,
        annot=False,  # Optionally add `annot=True` to show values
        cmap="cividis",  # Colormap for rate of change
        cbar_kws={"label": "Rate of Change (Normalized)"},
        ax=axes[1]
    )
    axes[1].set_title(f"Rate of Change of Metrics Across Perplexities for {title}")
    axes[1].set_xlabel("Perplexity")
    axes[1].set_ylabel("Metric")

    # Adjust layout for clarity
    plt.tight_layout()

    # Save the plot only if output_path is provided
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()


def plot_metrics_perplexity_with_combined(data, perplexities, output_path, title=''):
    metric_names = list(data.get(5).keys())
    metrics_optimal_one = ['new_metric', 'lcmc', 'kmeans_arand', 'distance_consistency','spearman_rho','pearson_r', 
                           'trustworthiness', 'continuity', 'ca_trustworthiness', 'ca_continuity', 'neighborhood_hit',
                           'mrre_false', 'mrre_missing', 'silhouette', 'steadiness', 'cohesiveness']
    metrics_optimal_zero = ['neighbor_dissimilarity', 'stress', 'distance_to_measure', 'projection_precision_score',   
                           'kl_divergence', 'procrustes', 'average_local_error']
    quality_metrics_order = metrics_optimal_one + metrics_optimal_zero

    # Normalize the metric values and calculate rate of change
    normalized_data = {}
    rate_of_change_data = {}
    combined_product_data = {}  # Store the combined product of metric value and rate of change

    for metric in metric_names:
        # Extract metric values for all perplexities
        values = [data[5][metric][i] for i in range(len(perplexities))]
        
        # Min-max normalization of metric values
        # normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
        # Apply Hybrid Normalization
        normalized_values = hybrid_normalization(values)  # Use hybrid normalization
        
        # Adjust metrics that are optimal at zero
        if metric in metrics_optimal_zero:
            normalized_values = 1 - normalized_values
        
        normalized_data[metric] = normalized_values
        
        # Calculate the rate of change (difference between consecutive values)
        rate_of_change = np.diff(normalized_values)
        rate_of_change_data[metric] = np.concatenate([[0], rate_of_change])  # Pad with 0 for first value
        
        # Calculate the product of metric value and rate of change
        combined_product = normalized_values * rate_of_change_data[metric]
        combined_product_data[metric] = combined_product
    
    # Create DataFrame for the heatmap of combined metric values and rate of change
    combined_product_df = pd.DataFrame(combined_product_data, index=perplexities).T
    combined_product_df.columns = [f"{p}" for p in perplexities]

    # Reorder rows based on the quality metrics
    combined_product_df = combined_product_df.reindex(quality_metrics_order)

    # Plot the heatmap for combined product of metric value and rate of change
    plt.figure(figsize=(15, 6))
    sns.heatmap(
        combined_product_df,
        annot=False,  # Optionally add `annot=True` to show values
        cmap="cividis",  # Colormap for the combined product
        cbar_kws={"label": "Metric Value * Rate of Change"},
    )
    plt.title(f"Combined Metric Values and Rate of Change Across Perplexities for {title}")
    plt.xlabel("Perplexity")
    plt.ylabel("Metric")
    plt.tight_layout()

    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()


# Plot compactness per cluster
def plot_relative_compactness(cluster_ratios, output_path ):
    clusters = list(cluster_ratios.keys())
    ratios = list(cluster_ratios.values())

    plt.figure(figsize=(10, 8))
    plt.bar(clusters, ratios, color=['green' if r < 1 else 'red' for r in ratios])
    plt.axhline(y=1, color='black', linestyle='--', label="No Change (Ratio = 1)")
    
    plt.xlabel("Cluster ID")
    plt.ylabel("Relative Compactness Ratio (2D / 3D)")
    plt.title("Relative Compactness of Clusters (t-SNE Effect)")
    plt.legend()

    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()

def plot_relative_change(relative_change_matrix,output_path ):
    plt.figure(figsize=(15, 6))
    sns.heatmap(
        relative_change_matrix,
        annot=False,  # Optionally add `annot=True` to show values
        cmap="hot",  # Colormap for the combined product
        # cbar_kws={"label": "Metric Value * Rate of Change"},
    )
    plt.title("Relative Change in Distances (2D vs 3D)")
    plt.xlabel("Cluster Index")
    plt.ylabel("Cluster Index")

    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()


def plot_bar_absolute_change_per_cluster(absolute_distance, unique_cluster, output_path):
    # Plot bar charts for relative change
    # clusters = range(1, inter_cluster_3D.shape[0] + 1)

    plt.figure(figsize=(10, 8))
    plt.bar(np.sort(unique_cluster), absolute_distance, color='blue', label='Inter-Cluster')
    # plt.bar(clusters, row_sum_intra, color='red', alpha=0.7, label='Intra-Cluster')
    plt.xlabel("Cluster Index")
    plt.ylabel("Absolute Sum of Distance Changes")
    plt.title("Absolute Change in Cluster Distances (2D vs. 3D)")
    plt.legend()
    plt.xticks(unique_cluster, labels=[f"Cluster {i}" for i in unique_cluster])
    # plt.show()
    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()


# def plot_normalized_distances_with_custom_colors(pairwise_dict, color_list, output_path):
#     num_clusters = len(pairwise_dict)

#     # Create a new figure
#     fig, ax = plt.subplots(figsize=(10, 8))

#     # Iterate through each cluster to plot its normalized distances
#     for y, (cluster, distances) in enumerate(pairwise_dict.items(), start=1):
#         sorted_items = list(distances.items())  # Convert the sub-dictionary to a list of tuples
#         segments = np.cumsum([item[1] for item in sorted_items])  # Cumulative sum of normalized distances
#         segments = np.insert(segments, 0, 0)  # Insert 0 at the start for the first segment
#         # breakpoint()
#         # Plot each line segment for this cluster with the corresponding color from the sub-cluster keys
#         for i, (sub_cluster, distance) in enumerate(sorted_items):
#             # Choose color based on sub-cluster key
#             segment_color = color_list[sub_cluster -1]  # Avoid out-of-bounds errors

#             # Plot segment
#             ax.plot([segments[i], segments[i+1]], [y, y], color=segment_color, lw=distance * 10)

#             # Add value on top of the line segment
#             mid_x = (segments[i] + segments[i+1]) / 2
#             # breakpoint()
#             ax.text(mid_x, y + 0.1, f"{distance:.2f}", color=segment_color, fontsize=12, ha='center')

#     # Adding labels and title
#     ax.set_xlabel('Normalized Distance')
#     ax.set_ylabel('Cluster')
#     ax.set_title('Normalized Pairwise Distances Between Clusters')
#     ax.set_ylim(0.5, num_clusters + 0.5)
#     ax.set_xlim(0, 1)

#     # Set y-ticks to show only cluster numbers, color them based on cluster index
#     ax.set_yticks(range(1, num_clusters + 1))  
#     ax.set_yticklabels(['Cluster '+str(i) for i in range(1, num_clusters + 1)])
    
#     # Color the y-tick labels based on cluster index
#     for i, tick in enumerate(ax.get_yticklabels()):
#         tick.set_color(color_list[i % len(color_list)])

#     # Save the plot only if path is specified
#     if output_path:
#         plt.savefig(output_path, dpi=dpi, format=save_format, bbox_inches="tight")
#         plt.close()
#     else:
#         print("Output folder not specified. Results not saved.")
#         plt.show()
#         plt.close()


import matplotlib.pyplot as plt
import numpy as np

def plot_normalized_distances_with_custom_colors(pairwise_dict, color_list, perplexity, title, output_path, figsize = {10,8}):
    num_clusters = len(pairwise_dict)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 8))
    # fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    # Small gap at the start of the x-axis
    # x_start_gap = 0.02  # Shift the x-axis slightly to avoid cutting off zero-length segments

    # Iterate through each cluster to plot its normalized distances
    for y, (cluster, distances) in enumerate(pairwise_dict.items(), start=1):
        sorted_items = list(distances.items())  # Convert the sub-dictionary to a list of tuples
        segments = np.cumsum([item[1] for item in sorted_items])  # Cumulative sum of normalized distances
        segments = np.insert(segments, 0, 0)  # Insert 0 at the start for the first segment

        # Ensure zero-length segments are visible by adjusting x positions slightly
        # segments = np.clip(segments, x_start_gap, 1)  # Prevent zero values from collapsing

        # Plot each line segment for this cluster with the corresponding color from the sub-cluster keys
        for i, (sub_cluster, distance) in enumerate(sorted_items):
            segment_color = color_list[sub_cluster - 1]  # Choose color based on sub-cluster key

            # Plot segment
            ax.plot([segments[i], segments[i+1]], [y, y], color=segment_color, lw=max(distance * 20, 1.0))  # Ensure visible line

            # Add value on top of the line segment
            mid_x = (segments[i] + segments[i+1]) / 2
            rounded_distance = round(distance, 2)
            ax.text(mid_x, y + 0.1, f"{rounded_distance:.2f}", color=segment_color, fontsize=12, ha='center')


    # Add vertical color lines on the y-axis for each cluster
    for i in range(1, num_clusters + 1):
        ax.plot([-0.1, -0.05], [i, i], color=color_list[i - 1], lw=30, solid_capstyle='round')  # Short colored line


    # Adding labels and title
    # ax.set_xlabel('Distance', fontsize=14)  # Larger x-axis label
    # ax.set_ylabel('Cluster', fontsize=14)
    # ax.set_title(f'Relative distance distortion {title} , {perplexity}', fontsize=16)
    ax.set_ylim(0.5, num_clusters + 0.5)
    ax.set_xlim(-0.1, 1)  # Start x-axis with a gap

    # # Set y-ticks to show only cluster numbers, color them based on cluster index
    # ax.set_yticks(range(1, num_clusters + 1))
    # ax.set_yticklabels(['Cluster ' + str(i) for i in range(1, num_clusters + 1)], fontsize=14)  # Larger y-tick labels

    # Remove default y-ticks (cluster numbers)
    ax.set_yticks([])
    # ax.axis('equal')
    plt.axis("off")

    # Color the y-tick labels based on cluster index
    for i, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(color_list[i % len(color_list)])

    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()


def plot_delaunay_triangulation(tri, embedding, c,n_gauss,  title="Delaunay Triangulation of t-SNE Output", 
                                point_color=colors, tri_color='blue', alpha=0.7, figsize=(10,8), output_path=None):
    """
    Plots the Delaunay triangulation of a 2D embedding.
    
    Parameters:
        tri (ndarray): Delaunay triangulation. A (M,3) array containing the traingle vertices.
        title (str): Title of the plot.
        point_color (list): Colors of the scatter points.
        tri_color (str): Color of the triangulation lines.
        alpha (float): Transparency of the triangulation lines.
        figsize (tuple): Figure size (width, height).
        output_path (str or None): If provided, saves the figure to this path.
        
    Returns:
        None
    """
   
    # Create figure
    plt.figure(figsize=figsize, constrained_layout=True)
    
    ax = plt.gca()
    # Plot triangulation
    ax.triplot(embedding[:, 0], embedding[:, 1], tri.simplices, color=tri_color, alpha=alpha, linewidth=0.8, linestyle='-')

    # Plot scatter points
    # ax.scatter(embedding[:, 0], embedding[:, 1], color=point_color, edgecolors='black', marker='o', s=15, label="Data Points")
    # Add scatter points for Gaussian clusters
    for i in range(n_gauss):
        ax.scatter(embedding[c == i, 0], embedding[c == i, 1], color=point_color[i],
                            zorder=3, edgecolor='k', s=50)

    # Labels & Title
    # plt.xlabel("t-SNE Dimension 1", fontsize=12, fontweight='bold')
    # plt.ylabel("t-SNE Dimension 2", fontsize=12, fontweight='bold')
    # plt.title(title, fontsize=14, fontweight='bold', pad=10)

    # Grid and legend
    # plt.grid(True, linestyle="--", alpha=0.5)
    # plt.legend(loc="upper right", fontsize=10)
    ax.axis('equal')
    plt.axis("off")

    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        # plt.close()

def validate_edge_length(edge_length):
    if edge_length is None or edge_length == 0 or edge_length == [] or edge_length == {}:
        raise ValueError("Error: edge_length cannot be None, zero, or empty.")
    
    # If it's a list or an array, check if all elements are zero
    if isinstance(edge_length, (list, tuple)) and all(el == 0 for el in edge_length):
        raise ValueError("Error: edge_length contains only zeros.")
    
def plot_triangulation_with_distortion_max_min(low_dm_emb, c,n_gauss, triang_t_sne, dist_ratio_area_normalized, highlight_triangles= [], bLogNorm = True, figsize=(10, 8),
                                       cmap_name="hot",point_color = colors, alpha=0.4, title="Triangles Colored by Distortion Ratio", output_path = None):
    """
    Plots a Delaunay triangulation of 2D embedded data, coloring each triangle 
    based on its distortion ratio.

    Parameters:
        low_dm_emb (ndarray): 2D coordinates of the low-dimensional embedding.
        triang_t_sne (Triangulation): Delaunay triangulation object.
        dist_ratio_area_normalized (array): Normalized distortion values for each triangle.
        cmap_name (str, optional): Name of the colormap to use (default: "viridis").
        alpha (float, optional): Transparency of triangle colors (default: 0.6).
        title (str, optional): Title of the plot (default: "Triangles Colored by Distortion Ratio").

    Returns:
        None
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    # fig, ax = plt.subplots(figsize=figsize)

    ax.axis('equal')
    plt.axis("off")
    # Plot triangulation as edges
    # ax.triplot(low_dm_emb[:, 0], low_dm_emb[:, 1], triang_t_sne.triangles, color='blue', alpha=0.7)
    ax.triplot(low_dm_emb[:, 0], low_dm_emb[:, 1], triang_t_sne.triangles, color='k', alpha=0.7)

    # Scatter plot the points
    # for i in range(n_gauss):
    #     ax.scatter(low_dm_emb[c == i, 0], low_dm_emb[c == i, 1], color=point_color[i],
    #                         label=f'Gaussian{i + 1}', zorder=3, edgecolor='k', s=50)


    if bLogNorm :
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=np.min(dist_ratio_area_normalized[dist_ratio_area_normalized > 0]), vmax=np.max(dist_ratio_area_normalized))
    else:
        norm = plt.Normalize(vmin=min(dist_ratio_area_normalized), vmax=max(dist_ratio_area_normalized))
    
    cmap = plt.get_cmap(cmap_name)
    
    # Color each triangle based on distortion ratio
    color_color = []
    for i, t in enumerate(triang_t_sne.triangles):
        triangle_coords = low_dm_emb[t]  # Triangle vertex coordinates
        color_tri = cmap(norm(dist_ratio_area_normalized[i]))  # Get color for this triangle

        ax.fill(triangle_coords[:, 0], triangle_coords[:, 1], color=color_tri, edgecolor='k', alpha=alpha)

        # # Check if the current triangle is in the highlight list
        # if i in highlight_triangles:
        #     ax.fill(triangle_coords[:, 0], triangle_coords[:, 1], 
        #             color=color_tri, edgecolor='red', linewidth=2.5, alpha=0.8)  # Highlighted triangles
        # else:
        #     ax.fill(triangle_coords[:, 0], triangle_coords[:, 1], 
        #             color=color_tri, edgecolor='k', alpha=alpha)  # Regular triangles

        color_color.append(color_tri)

    # Set title and labels
    # ax.set_title(title)
    # ax.set_xlabel("t-SNE Dimension 1")
    # ax.set_ylabel("t-SNE Dimension 2")

    

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # fig.colorbar(sm, ax=ax)

    # if bLogNorm:
    #     tick_values = np.logspace(np.log10(norm.vmin), np.log10(norm.vmax), num=5)  
    #     cbar = fig.colorbar(sm, ax=ax, ticks=tick_values)
    #     cbar.ax.set_yticklabels([f"{t:.1e}" for t in tick_values])
    #     # **Disable minor ticks to remove extra lines**
    #     cbar.ax.yaxis.set_minor_locator(ticker.NullLocator())  
    # else:
    #     tick_values = np.linspace(min(dist_ratio_area_normalized), max(dist_ratio_area_normalized), num=5)
    #     fig.colorbar(sm, ax=ax, ticks=tick_values)

    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()

def plot_triangulation_colored_edges(triangle, embedding, c, n_gauss, edge_weights, 
                                title="Delaunay Triangulation with Colored Edges", 
                                point_color=colors, cmap="hot", alpha=0.7, bLogNorm = True, figsize=(8,6), black_clust_points=False, output_path=None):
    """
    Plots the Delaunay triangulation of a 2D embedding with edges colored by their weights.

    Parameters:
        tri (Delaunay): Delaunay triangulation object.
        embedding (ndarray): 2D coordinates of the data points.
        c (ndarray): Cluster labels for coloring points.
        n_gauss (int): Number of clusters.
        edge_weights (ndarray): Weights for each edge in the triangulation.
        title (str): Title of the plot.
        point_color (list or None): Colors of the scatter points.
        cmap (str): Colormap for edge coloring.
        alpha (float): Transparency of the triangulation lines.
        figsize (tuple): Figure size (width, height).
        output_path (str or None): If provided, saves the figure to this path.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=figsize)

    ########____Black background changes____________________
    fig.patch.set_facecolor('black')  # Set figure background
    ax.set_facecolor('black')

    # Change tick labels and spines to white for visibility
    ax.tick_params(colors='white')  
    for spine in ax.spines.values():
        spine.set_color('white') 
    
    ########____Black background changes end____________________

    cmap = plt.get_cmap(cmap)
    norm = LogNorm(vmin=np.min(edge_weights[edge_weights > 0]), vmax=np.max(edge_weights))

    # Create an array to store edges
    tri_edges = np.zeros((triangle.shape[0], 3, 2), dtype=int)

    # Loop through triangles and store their edges
    for i, tri_nodes in enumerate(triangle):
        tri_edges[i] = [
            (tri_nodes[0], tri_nodes[1]),  # Edge 1
            (tri_nodes[1], tri_nodes[2]),  # Edge 2
            (tri_nodes[2], tri_nodes[0])   # Edge 3
        ]
    tri_edges = np.array(list(tri_edges))  # Convert set to array

    # Loop through each triangle and its edges
    for i in range(tri_edges.shape[0]):
        for j in range(tri_edges.shape[1]):
            
            # Get the start and end points of the edge
            edge_start = embedding[tri_edges[i, j, 0]]  # Index to get the coordinate
            edge_end = embedding[tri_edges[i, j, 1]]    # Index to get the coordinate
            
            # Get the weight for the current edge
            edge_weight = edge_weights[i, j]
            # validate_edge_length(edge_weight)
            # Map the weight to a color
            color = cmap(norm(edge_weight))
            # breakpoint()
            # Plot the edge
            ax.plot([edge_start[0], edge_end[0]], [edge_start[1], edge_end[1]], color=color)

    if black_clust_points:
        ax.scatter(embedding[:, 0], embedding[:, 1], color='k', edgecolor='k', s=50)
    else:
        # Scatter plot the points
        for i in range(n_gauss):
            ax.scatter(embedding[c == i, 0], embedding[c == i, 1], color=point_color[i], 
                    label=f'Gaussian{i + 1}', edgecolor='k', s=50)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # fig.colorbar(sm, ax=ax)

    if bLogNorm:
        tick_values = np.logspace(np.log10(norm.vmin), np.log10(norm.vmax), num=5)  
        cbar = fig.colorbar(sm, ax=ax, ticks=tick_values)
        cbar.ax.set_yticklabels([f"{t:.1e}" for t in tick_values])
        # **Disable minor ticks to remove extra lines**
        cbar.ax.yaxis.set_minor_locator(ticker.NullLocator()) 

    ########____Black background changes____________________

    cbar.ax.yaxis.set_tick_params(color='white')  # Change tick color
    cbar.ax.yaxis.set_ticklabels([f"{t:.1e}" for t in tick_values], color='white')  # Set white tick labels

    ########____Black background changes end____________________


    # Labels and Title
    # ax.set_title(title, fontsize=14)
    ax.set_xlim(embedding[:, 0].min(), embedding[:, 0].max())
    ax.set_ylim(embedding[:, 1].min(), embedding[:, 1].max())

    # Remove x and y ticks but keep the box
    # ax = plt.gca()
    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Save or show plot
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_triangulation_colored_edges_v2(triangle, embedding, c, n_gauss, edge_weights, 
                                     title="Delaunay Triangulation with Colored Edges", 
                                     point_color=colors, cmap="hot", alpha=0.7, bLogNorm=True, figsize=(8,6), 
                                     black_clust_points=False, output_path=None):
    """
    Plots the Delaunay triangulation of a 2D embedding with edges colored by their weights.

    Parameters:
        tri (Delaunay): Delaunay triangulation object.
        embedding (ndarray): 2D coordinates of the data points.
        c (ndarray): Cluster labels for coloring points.
        n_gauss (int): Number of clusters.
        edge_weights (ndarray): Weights for each edge in the triangulation.
        title (str): Title of the plot.
        point_color (list or None): Colors of the scatter points.
        cmap (str): Colormap for edge coloring.
        alpha (float): Transparency of the triangulation lines.
        figsize (tuple): Figure size (width, height).
        output_path (str or None): If provided, saves the figure to this path.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=figsize)

    ########____Black background changes____________________
    fig.patch.set_facecolor('black')  # Set figure background
    ax.set_facecolor('black')

    # Change tick labels and spines to white for visibility
    ax.tick_params(colors='white')  
    for spine in ax.spines.values():
        spine.set_color('white') 
    
    ########____Black background changes end____________________

    cmap = plt.get_cmap(cmap)

    # Check if LogNorm is used or normal normalization is required
    if bLogNorm:
        norm = LogNorm(vmin=np.min(edge_weights[edge_weights > 0]), vmax=np.max(edge_weights))
    else:
        # Normal Min-Max normalization
        edge_weights_min = np.min(edge_weights)
        edge_weights_max = np.max(edge_weights)
        norm = plt.Normalize(vmin=edge_weights_min, vmax=edge_weights_max)

    # Create an array to store edges
    tri_edges = np.zeros((triangle.shape[0], 3, 2), dtype=int)

    # Loop through triangles and store their edges
    for i, tri_nodes in enumerate(triangle):
        tri_edges[i] = [
            (tri_nodes[0], tri_nodes[1]),  # Edge 1
            (tri_nodes[1], tri_nodes[2]),  # Edge 2
            (tri_nodes[2], tri_nodes[0])   # Edge 3
        ]
    tri_edges = np.array(list(tri_edges))  # Convert set to array

    # Loop through each triangle and its edges
    for i in range(tri_edges.shape[0]):
        for j in range(tri_edges.shape[1]):
            
            # Get the start and end points of the edge
            edge_start = embedding[tri_edges[i, j, 0]]  # Index to get the coordinate
            edge_end = embedding[tri_edges[i, j, 1]]    # Index to get the coordinate
            
            # Get the weight for the current edge
            edge_weight = edge_weights[i, j]
            # Map the weight to a color
            color = cmap(norm(edge_weight))

            # Plot the edge
            ax.plot([edge_start[0], edge_end[0]], [edge_start[1], edge_end[1]], color=color)

    if black_clust_points:
        ax.scatter(embedding[:, 0], embedding[:, 1], color='k', edgecolor='k', s=50)
    else:
        # Scatter plot the points
        for i in range(n_gauss):
            ax.scatter(embedding[c == i, 0], embedding[c == i, 1], color=point_color[i], 
                    label=f'Gaussian{i + 1}', edgecolor='k', s=50)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if bLogNorm:
        # For LogNorm, use log-spaced ticks
        tick_values = np.logspace(np.log10(norm.vmin), np.log10(norm.vmax), num=5)  
        cbar = fig.colorbar(sm, ax=ax, ticks=tick_values)
        cbar.ax.set_yticklabels([f"{t:.1e}" for t in tick_values])
        # **Disable minor ticks to remove extra lines**
        cbar.ax.yaxis.set_minor_locator(ticker.NullLocator()) 
    else:
        # For normal normalization, use linearly spaced ticks
        tick_values = np.linspace(norm.vmin, norm.vmax, num=5)
        cbar = fig.colorbar(sm, ax=ax, ticks=tick_values)
        cbar.ax.set_yticklabels([f"{t:.1e}" for t in tick_values])
        
    ########____Black background changes____________________

    cbar.ax.yaxis.set_tick_params(color='white')  # Change tick color
    cbar.ax.yaxis.set_ticklabels([f"{t:.1e}" for t in tick_values], color='white')  # Set white tick labels

    ########____Black background changes end____________________

    # Labels and Title
    ax.set_xlim(embedding[:, 0].min(), embedding[:, 0].max())
    ax.set_ylim(embedding[:, 1].min(), embedding[:, 1].max())

    # Remove x and y ticks but keep the box
    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Save or show plot
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()  

    
def plot_triangulation_colored_edges_v3(triangle, embedding, c, n_gauss, edge_weights, 
                                         title="Delaunay Triangulation with Colored Edges", 
                                         point_color=colors, cmap="hot", alpha=0.7, figsize=(8,6), 
                                         black_clust_points=False, output_path=None):
    """
    Plots the Delaunay triangulation of a 2D embedding with edges colored by their weights.

    Parameters:
        triangle (ndarray): Delaunay triangulation object.
        embedding (ndarray): 2D coordinates of the data points.
        c (ndarray): Cluster labels for coloring points.
        n_gauss (int): Number of clusters.
        edge_weights (ndarray): Weights for each edge in the triangulation.
        title (str): Title of the plot.
        point_color (list or None): Colors of the scatter points.
        cmap (str): Colormap for edge coloring.
        alpha (float): Transparency of the triangulation lines.
        figsize (tuple): Figure size (width, height).
        output_path (str or None): If provided, saves the figure to this path.

    Returns:
        None
    """

    fig, ax = plt.subplots(figsize=figsize)

    ########____Black background changes____________________
    fig.patch.set_facecolor('black')  # Set figure background
    ax.set_facecolor('black')

    # Change tick labels and spines to white for visibility
    ax.tick_params(colors='white')  
    for spine in ax.spines.values():
        spine.set_color('white') 
    ########____Black background changes end____________________

    cmap = plt.get_cmap(cmap)  # Get the colormap

    # Create an array to store edges
    tri_edges = np.zeros((triangle.shape[0], 3, 2), dtype=int)

    # Loop through triangles and store their edges
    for i, tri_nodes in enumerate(triangle):
        tri_edges[i] = [
            (tri_nodes[0], tri_nodes[1]),  # Edge 1
            (tri_nodes[1], tri_nodes[2]),  # Edge 2
            (tri_nodes[2], tri_nodes[0])   # Edge 3
        ]
    tri_edges = np.array(list(tri_edges))  # Convert set to array

    # Loop through each triangle and its edges
    for i in range(tri_edges.shape[0]):
        for j in range(tri_edges.shape[1]):
            
            # Get the start and end points of the edge
            edge_start = embedding[tri_edges[i, j, 0]]  # Index to get the coordinate
            edge_end = embedding[tri_edges[i, j, 1]]    # Index to get the coordinate
            
            # Get the weight for the current edge (which is already normalized between 0 and 1)
            edge_weight = edge_weights[i, j]
            
            # Map the weight to a color using the colormap
            color = cmap(edge_weight)

            # Plot the edge
            ax.plot([edge_start[0], edge_end[0]], [edge_start[1], edge_end[1]], color=color)

    if black_clust_points:
        ax.scatter(embedding[:, 0], embedding[:, 1], color='k', edgecolor='k', s=50)
    else:
        # Scatter plot the points
        for i in range(n_gauss):
            ax.scatter(embedding[c == i, 0], embedding[c == i, 1], color=point_color[i], 
                    label=f'Gaussian{i + 1}', edgecolor='k', s=50)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])

    # Linearly spaced ticks for the colorbar
    tick_values = np.linspace(0, 1, num=5)
    cbar = fig.colorbar(sm, ax=ax, ticks=tick_values)
    cbar.ax.set_yticklabels([f"{t:.1f}" for t in tick_values])

    ########____Black background changes____________________
    cbar.ax.yaxis.set_tick_params(color='white')  # Change tick color
    cbar.ax.yaxis.set_ticklabels([f"{t:.1f}" for t in tick_values], color='white')  # Set white tick labels
    ########____Black background changes end____________________

    # Labels and Title
    ax.set_xlim(embedding[:, 0].min(), embedding[:, 0].max())
    ax.set_ylim(embedding[:, 1].min(), embedding[:, 1].max())

    # Remove x and y ticks but keep the box
    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Save or show plot
    if output_path:
        plt.savefig(f"{output_path}.png", dpi=300, format='png', bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()   



def plot_histogram_values(data, bins = 20, output_path=None):
    # Plot histogram
    counts, bin_edges, patches = plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)

    # Add value labels on top of each bar
    for count, bin_edge in zip(counts, bin_edges[:-1]):
        if count > 0:  # Avoid labeling bins with zero count
            plt.text(bin_edge + (bin_edges[1] - bin_edges[0]) / 2, count, str(int(count)), 
                    ha='center', va='bottom', fontsize=10, color='black')

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Array Values")
        # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()




##________________ U Vector analysis plot___________________________

def plot_U_vector_field(U1, U2,U_matrices, X, Y, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Normalize vectors for clarity
    scale_factor = np.linalg.norm(U1, axis=0).max()  # Scale by max norm
    # breakpoint()
    U1 /= scale_factor
    U2 /= scale_factor
    # breakpoint()
    
    magnitude = np.sqrt(U1**2 + U2**2)
    # Plot the quiver (vector field)
    ax.quiver(X, Y, U1, U1, color='r', angles='xy', scale_units='xy', scale=0.2)
    plt.contourf(X, Y, magnitude, levels=50, cmap='hot')
    plt.colorbar(label='Vector Magnitude')

    # magnitude = np.sqrt(U1**2 + U2**2)
    # normalized_magnitude = magnitude / magnitude.max()
    # quiver_plot = ax.quiver(X, Y, U1, U2, normalized_magnitude, cmap='plasma', angles='xy', scale_units='xy', scale=0.1)

    # plt.colorbar(quiver_plot, ax=ax, label="Vector Magnitude")    
    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    # ax.set_title("SVD U-matrix (First Output Dimension)")
    
    
    # Save the plot only if path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        print("Output folder not specified. Results not saved.")
        plt.show()
        plt.close()

from scipy.ndimage import gaussian_filter

def plot_U_vector_field_3(U1, U2, X, Y, output_path, save_format="png", dpi=300, arrow_factor=5):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Normalize vectors for clarity
    scale_factor = np.linalg.norm(U1, axis=0).max()  # Scale by max norm
    U1 /= scale_factor
    U2 /= scale_factor
    
    # **Scale the arrows' length dynamically based on magnitude**
    magnitude = np.sqrt(U1**2 + U2**2)
    max_magnitude = magnitude.max() if magnitude.max() > 0 else 1  # Avoid division by zero
    normalized_magnitude = magnitude / max_magnitude
    
    # **Adjust arrow length based on the normalized magnitude**
    U1_scaled = U1 * (arrow_factor * normalized_magnitude)
    U2_scaled = U2 * (arrow_factor * normalized_magnitude)
    
    # **Plot the quiver (vector field)**
    quiver_plot = ax.quiver(X, Y, U1_scaled, U2_scaled, color='r', angles='xy', scale_units='xy', scale=1, width=0.003, alpha=0.8)

    # **Optional: Add labels, title, and axis control**
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("SVD U-matrix (First Output Dimension)")

    # Save the plot only if the output path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()



def plot_U_vector_field_updated(U_matrices, X,Y, output_path, save_format="png", dpi=300, bdensity_reduce = False):
    
    # Extract the first coloumn of U matrix (associated with largest singluar value) for each grid point
    first_col_U = U_matrices[:, :, 0, :]  # Shape will be (200, 200, 3)

    # breakpoint()
    # Extract U1 and U2 components from the first column of U matrix
    U1 = first_col_U[:, :, 0]  
    U2 = first_col_U[:, :, 1] 

    # breakpoint()
    # breakpoint()
    # # Normalize the vectors
    magnitude = np.sqrt(U1**2 + U2**2)
    # Normalize magnitude **per row** (row-wise normalization)
    # magnitude_row_norm = magnitude / magnitude.max(axis=1, keepdims=True)  # Normalize within each row
    magnitude_row_norm = magnitude / magnitude.max() # Normalize globally

    # U1 /= magnitude
    # U2 /= magnitude
    # breakpoint()

    # Normalize vectors
    # scale_factor = np.linalg.norm(U1, axis=0).max()  # Scale by max norm
    # U1 /= scale_factor
    # U2 /= scale_factor

    # breakpoint()
    U1 = U1/np.max(np.abs(U1))
    U2 = U2/np.max(np.abs(U2))

    # breakpoint()


    # scaling_factor = 0.05  # Adjust for better visualization
    # U1_scaled = U1 * (1 + magnitude_row_norm * scaling_factor)  # Ensure nonzero length
    # U2_scaled = U2 * (1 + magnitude_row_norm * scaling_factor)
    
    # breakpoint()
    
    # Create the quiver plot
    fig, ax = plt.subplots(figsize=(8, 8))
    plt_every = 5
    # quive = ax.quiver(X, Y, U1, U2, magnitude_row_norm, cmap ='plasma' , scale=0.9, angles='xy', scale_units='xy')
    # quive_1= ax.quiver(X[::plt_every, ::plt_every], Y[::plt_every, ::plt_every], U1[::plt_every, ::plt_every], U2[::plt_every, ::plt_every], scale=0.2, angles='xy', scale_units='xy', color = 'b')
    # quive = ax.quiver(X, Y, U1_scaled, U2_scaled, angles='xy', scale_units='xy', color = 'b')
    
    if bdensity_reduce:
        quive_1= ax.quiver(X[::plt_every, ::plt_every], Y[::plt_every, ::plt_every], U1[::plt_every, ::plt_every], U2[::plt_every, ::plt_every], scale=0.05, angles='xy', scale_units='xy', color = 'b', headaxislength=0, headlength=0)
    else:
        quive_1= ax.quiver(X, Y, U1, U2, scale=0.8, angles='xy', scale_units='xy', color = 'b', headaxislength=0, headlength=0)

    # Add colorbar to show magnitude
    # cbar = plt.colorbar(quive)
    # cbar.set_label("Magnitude")

    # # Add labels and title
    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    # ax.set_title("Quiver Plot of First Row of U Matrix")

    # Remove x and y ticks but keep the box
    # ax = plt.gca()
    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Save the plot only if the output path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()

def  plot_interpolation(n_gauss, embedding, c, point_color, interpolations, x_min, x_max, y_min, y_max, bscatter_plot = False, background_color = 'green',  output_path = None, figsize=(10, 10)):
    
# Plot the Delaunay triangulation
    # fig, ax = plt.subplots(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    # ax.set_facecolor('black')  # Set background color of plot area to black
    # Define padding values
    # x_pad = 0.1 * (x_max - x_min)  
    # y_pad = 0.1 * (y_max - y_min) 
    # breakpoint()
    heatmap = ax.imshow(interpolations,
            # extent=(x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad),
            extent=(x_min, x_max, y_min, y_max),
            origin='lower',
            cmap='hot',
            alpha=1.0,
            # interpolation='nearest'
        )
    # Set axis limits explicitly to enforce the gap
    # ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_xlim(x_min, x_max )
    # ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_ylim(y_min, y_max)
    ax.axis('equal')
    plt.axis("off")

    # Ensure NaN values appear as background color
    heatmap.cmap.set_bad(background_color)
    # plt.triplot(low_dm_emb[:, 0], low_dm_emb[:, 1], tri_delaunay.simplices, color='white', alpha=0.6)

    
    if bscatter_plot:
        # for i in range(n_gauss):
        for ind, i in enumerate(np.unique(c)):
            ax.scatter(embedding[c == i, 0], embedding[c == i, 1], color=point_color[int(ind)], 
                    edgecolor='k', s=15, zorder = 5)
    
    # fig.colorbar(heatmap, ax=ax)
    # Remove x and y ticks but keep the box
    # ax = plt.gca()
    ax.set_xticks([])  
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

     # Save the plot only if the output path is specified
    if output_path:
        plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        # plt.close()

    
# from skimage.draw import line as rasterize_line
# from matplotlib.collections import LineCollection

# def blend_colors(colors, method="average"):
#     """ Blend a list of colors using the chosen method. """
#     colors = np.array(colors)  # Convert to NumPy array (N, 3)
    
#     if method == "average":
#         return np.mean(colors, axis=0)  # Mean blending
#     elif method == "max":
#         return np.max(colors, axis=0)  # Max blending
#     elif method == "min":
#         return np.min(colors, axis=0)  # Min blending
#     else:
#         raise ValueError("Invalid blending method. Choose 'average', 'max', or 'min'.")
# import numpy as np
# from skimage.draw import line as rasterize_line

# def rasterize_lines(line_segments, colors, image_size, blend_method="average"):
#     H, W = image_size
#     img = np.zeros((H, W, 3), dtype=np.float32)  # RGB image
#     count = np.zeros((H, W), dtype=np.float32)   # Overlapping line counter

#     for i, (p1, p2) in enumerate(line_segments):
#         # Convert floating point coordinates to integers for rasterization
#         x1, y1 = map(int, p1)
#         x2, y2 = map(int, p2)
        
#         # Get pixel coordinates of the line
#         rr, cc = rasterize_line(y1, x1, y2, x2)

#         # Ensure the indices are within image bounds
#         valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
#         rr, cc = rr[valid], cc[valid]

#         # Accumulate color contributions
#         img[rr, cc] += colors[i][:3]  # Ensure RGB color (ignore alpha if exists)
#         count[rr, cc] += 1  # Track the number of lines per pixel

#     # Avoid division by zero
#     mask = count > 0

#     # Expand count's shape to match (H, W, 3) for broadcasting
#     img[mask] /= count[mask][:, None]  # Broadcasting fix

#     return img



# def plot_fully_connected_points(embedding, D, c,n_gauss, point_color=colors, output_path = None):
#     # Generate all possible edges
#     edges = list(itertools.combinations(range(len(embedding)), 2))  # All point pairs
#     # Define edge colors based on some property (e.g., Euclidean distance)
#     edge_weights = [np.linalg.norm(D[i] - D[j]) for i, j in edges]

#     # breakpoint()

#     # Normalize and map to colormap
#     cmap = plt.get_cmap("hot")
#     norm = plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
#     edge_colors = [cmap(norm(w))[:3]  for w in edge_weights]

#     # Create line segments
#     line_segments = [(embedding[i], embedding[j]) for i, j in edges]

#     # Rasterize and blend
#     image_size = (500, 500)  # Image resolution
#     rasterized_img = rasterize_lines(line_segments, edge_colors, image_size, blend_method="average")

#     # Plot rasterized image
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.imshow(rasterized_img, extent=[embedding[:, 0].min(), embedding[:, 0].max(),
#                                       embedding[:, 1].min(), embedding[:, 1].max()], origin="lower")

#     # Scatter plot the points
#     for i in range(n_gauss):
#         ax.scatter(embedding[c == i, 0], embedding[c == i, 1], color=point_color[i], edgecolor='k', s=50)

#     plt.title("Rasterized Fully Connected t-SNE with Blended Edge Colors")
#     ax.set_xticks([]), ax.set_yticks([])

#     # Save or show the plot
#     if output_path:
#         plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
#         plt.close()
#     else:
#         plt.show()
#         plt.close()


#     #     # Plot
#     # fig, ax = plt.subplots(figsize=(8, 6))
#     # # Scatter plot the points
#     # for i in range(n_gauss):
#     #     ax.scatter(embedding[c == i, 0], embedding[c == i, 1], color=point_color[i], 
#     #              edgecolor='k', s=50)

#     # # Add colored edges
#     # lc = LineCollection(line_segments, colors=edge_colors, linewidths=0.5)
#     # ax.add_collection(lc)


#     # # Colorbar
#     # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     # sm.set_array([])
#     # plt.colorbar(sm, ax=ax, label="Edge Distance")

#     # plt.title("Fully Connected t-SNE Output with Edge Colors")

#     # ax.set_xticks([]), ax.set_yticks([])
#     # ax.set_frame_on(False)

    

#     #  # Save the plot only if the output path is specified
#     # if output_path:
#     #     plt.savefig(f"{output_path}.{save_format}", dpi=dpi, format=save_format, bbox_inches="tight")
#     #     plt.close()
#     # else:
#     #     plt.show()
#     #     plt.close()


import numpy as np
import matplotlib.pyplot as plt
import itertools
from skimage.draw import line as rasterize_line
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from matplotlib import cm

# def rasterize_lines(line_segments, colors, image_size, blend_method="max"):
#     H, W = image_size
#     img = np.zeros((H, W, 3), dtype=np.float32)  # RGB image
#     count = np.zeros((H, W), dtype=np.float32)   # Tracks number of overlapping lines per pixel

#     for i, (p1, p2) in enumerate(line_segments):
#         # Convert floating point coordinates to integers for rasterization
#         x1, y1 = map(int, p1)
#         x2, y2 = map(int, p2)
#         # y1, x1 = map(int, p1)
#         # y2, x2 = map(int, p2)
        
#         # Get pixel coordinates of the line
#         rr, cc = rasterize_line(y1, x1, y2, x2)

#         # Ensure indices are within bounds
#         valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
#         rr, cc = rr[valid], cc[valid]

#         # # Invert the y-coordinates to match image coordinate system
#         # rr = img.shape[0] - rr - 1

#         # Add colors to the raster image
#         img[rr, cc] += colors[i][:3]  # Ensure RGB
#         count[rr, cc] += 1  # Track overlap count

        

#     # Normalize color blending
#     mask = count > 0  # Pixels where lines exist

#     # Normalize based on the count of overlaps
#     if blend_method == "average":
#         img[mask] /= count[mask][:, None]  # Correct broadcasting for average
    
#     elif blend_method == "max":
#         # Perform max blending correctly for each channel (R, G, B)
#         for c in range(3):  # Iterate over each channel (R, G, B)
#             img[mask, c] = np.maximum(img[mask, c], colors[i][c])  # Take the max across channels

#     elif blend_method == "min":
#         # Perform min blending correctly for each channel (R, G, B)
#         for c in range(3):  # Iterate over each channel (R, G, B)
#             img[mask, c] = np.minimum(img[mask, c], colors[i][c])  # Take the min across channels

#     # Normalize the image based on the min and max values across the entire image
#     min_val = np.min(img)
#     max_val = np.max(img)

#     # If the image is not already in the range [0, 1], normalize it
#     if max_val > min_val:
#         img = (img - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    
#     return img


##____________________ The above works for average__________________________________

# def rasterize_lines(line_segments, colors, image_size, blend_method="max"):
#     H, W = image_size
#     # img = np.zeros((H, W, 3), dtype=np.float32)  # RGB image initialized to black
#     if blend_method == "max":
#         img = np.zeros((H, W, 3), dtype=np.float32)  # RGB image initialized to black
#     elif blend_method == "min":
#         img = np.zeros((H, W, 3), dtype=np.float32)  # RGB image initialized to white
#     elif blend_method == "average":
#         img = np.zeros((H, W, 3), dtype=np.float32)  # RGB image initialized to white
    
#     count = np.zeros((H, W), dtype=np.float32)   # Tracks number of overlapping lines per pixel
#     # breakpoint()
#     for i, (p1, p2) in enumerate(line_segments):
#         x1, y1 = map(int, p1)
#         x2, y2 = map(int, p2)

#         # Get pixel coordinates of the line using Bresenham's algorithm
#         rr, cc = rasterize_line(y1, x1, y2, x2)
#         # breakpoint()
#         # Ensure indices are within bounds
#         valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
#         rr, cc = rr[valid], cc[valid]

#         # breakpoint()

#         # Process each pixel for the current line based on the blend method
#         for j in range(len(rr)):
#             if blend_method == "average":
#                 img[rr[j], cc[j]] += colors[i][:3]
#                 count[rr[j], cc[j]] += 1
#             elif blend_method == "max":
#                 if np.linalg.norm(colors[i][:3]) >= np.linalg.norm(img[rr[j], cc[j]]):
#                     img[rr[j], cc[j]] = colors[i][:3]  # Assign the full RGB color
                    
#             elif blend_method == "min":
#                 if np.linalg.norm(colors[i][:3]) < np.linalg.norm(img[rr[j], cc[j]]):
#                     img[rr[j], cc[j]] = colors[i][:3]  # Assign the full RGB color
#                 # Check if the current pixel is black (initialized value)
#                 if np.all(img[rr[j], cc[j]] == [0, 0, 0]) or np.linalg.norm(colors[i][:3]) < np.linalg.norm(img[rr[j], cc[j]]):
#                     img[rr[j], cc[j]] = colors[i][:3]  # Assign the full RGB color

#     # breakpoint()
#     # Normalize the image based on the number of overlaps for "average" blending
#     if blend_method == "average":
#         mask = count > 0  # Only normalize pixels that had overlapping lines
#         img[mask] /= count[mask][:, None]  # Normalize correctly using broadcasting for average

#     # Normalize the image based on the min and max values across only valid pixels
#     valid_pixels = count > 0  # Valid pixels are those that had at least one line
#     if np.any(valid_pixels):
#         min_val = np.min(img[valid_pixels])
#         max_val = np.max(img[valid_pixels])

#         # Normalize only the valid pixels to the range [0, 1]
#         if max_val > min_val:
#             img[valid_pixels] = (img[valid_pixels] - min_val) / (max_val - min_val)
#     else:
#         print("Warning: No valid pixels found for normalization.")

#     return img

import numpy as np

def rasterize_lines(line_segments, colors, image_size, blend_method="max"):
    H, W = image_size

    # # Initialize the image based on the blend method
    # if blend_method == "max":
    #     img = np.zeros((H, W, 3), dtype=np.float32)  # Black background
    # else:  # min and average methods
    #     # img = np.ones((H, W, 3), dtype=np.float32) * 255  # White background
    #     img = np.zeros((H, W, 3), dtype=np.float32) # Black background

    img = np.zeros((H, W, 3), dtype=np.float32)  # Black background
    count = np.zeros((H, W), dtype=np.float32)  # To track overlapping lines
    visited = np.zeros((H, W), dtype=bool)

    for i, (p1, p2) in enumerate(line_segments):
        x1, y1 = map(int, p1)
        x2, y2 = map(int, p2)

        # Get pixel coordinates of the line
        rr, cc = rasterize_line(y1, x1, y2, x2)

        # Ensure indices are within bounds
        valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc = rr[valid], cc[valid]

        visited[rr, cc] = True

        # Extract current pixel values
        current_pixels = img[rr, cc]
        new_color = colors[i][:3]

        if blend_method == "average":
            img[rr, cc] += new_color
            count[rr, cc] += 1

        elif blend_method == "max":
            mask = np.linalg.norm(new_color) >= np.linalg.norm(current_pixels, axis=1)
            img[rr[mask], cc[mask]] = new_color  # Apply new color only where needed

        elif blend_method == "min":
            new_color_norm = np.linalg.norm(new_color)
            current_norms = np.linalg.norm(current_pixels, axis=1)

            # Find where the new color should replace the current pixel
            mask = (current_norms > new_color_norm) | np.all(current_pixels == [0,0,0], axis=1)
            img[rr[mask], cc[mask]] = new_color  # Apply the minimum color efficiently

    
    # 
    # img[~mask] = 1  # Set not used pixels to white background
    # Normalize for "average" blending
    if blend_method == "average":
        mask = count > 0
        img[mask] /= count[mask][:, None]
        img[~mask] = 1  # Set not used pixels to white background
    
    if blend_method in ["min", "max"]:
        # breakpoint()
        img[~visited] = 1  # Set not used pixels to white background
    
    # if blend_method == "average":
    #     mask = count > 0
    #     img[mask] /= count[mask][:, None]  # Compute the true average

    #     # img[mask] = (img[mask] - img[mask].min()) / (img[mask].max() - img[mask].min()) * 255
    #     img[~mask] = 255  # Set not used pixels to white background

    # Normalize image contrast
    valid_pixels = count > 0
    if np.any(visited):
        min_val = np.min(img[visited])
        max_val = np.max(img[visited])

        if max_val > min_val:
            img[visited] = (img[visited] - min_val) / (max_val - min_val)

    return img


def plot_fully_connected_points_optimized(embedding, D, c, n_gauss, point_color, blend_methods=['average'], bscatter_plot = False, output_path=None, image_size=(500, 500), figsize = (10,10)):
    """
    Plots a fully connected graph using rasterized edges with different blending approaches.

    Parameters:
    - embedding: 2D coordinates of points
    - D: Original data for edge weight computation
    - c: Cluster labels
    - n_gauss: Number of clusters
    - point_color: List of colors for points
    - blend_methods: List of blending approaches to use (e.g., ['average', 'max', 'min'])
    - output_path: File path prefix for saving plots
    - image_size: Size of the output image
    - figsize: Figure size
    """
    # edges = list(itertools.combinations(range(len(embedding)), 2))
    # edge_weights_old = np.array([np.linalg.norm(D[i] - D[j]) for i, j in edges])  # Compute edge lengths once

    edge_weights_full = squareform(pdist(D))  # Only once!
    edge_weights = edge_weights_full[np.triu_indices(len(D), k=1)]


    # Normalize edge colors
    cmap_w = cm.hot
    norm_weights = Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())
    edge_colors = cmap_w(norm_weights(edge_weights))

    # Convert embedding to image coordinates
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)
    scale_x = (image_size[1] - 1) / (max_x - min_x)
    scale_y = (image_size[0] - 1) / (max_y - min_y)

    # embedding_scaled_old = np.array([
    #     [(x - min_x) * scale_x, (y - min_y) * scale_y]
    #     for x, y in embedding
    # ])

    embedding_scaled = (embedding - [min_x, min_y]) * [scale_x, scale_y]

    # breakpoint()
    # Prepare line segments
    # line_segments_old = np.array([(embedding_scaled[i], embedding_scaled[j]) for i, j in edges])

    # Get upper triangle indices (i < j)
    i_idx, j_idx = np.triu_indices(len(embedding_scaled), k=1)
    # Stack them into line segments of shape (num_edges, 2, 2)
    line_segments = np.stack([embedding_scaled[i_idx], embedding_scaled[j_idx]], axis=1)
    # breakpoint()

    # Loop over selected blending methods
    for blend_method in blend_methods:
        print(f'Running blending: {blend_method} ')
        # Rasterize the image with blended edges
        rasterized_img = rasterize_lines(line_segments, edge_colors, image_size, blend_method=blend_method)

        # breakpoint()
        # Normalize for the color bar
        norm = Normalize(vmin=rasterized_img.min(), vmax=rasterized_img.max())
        cmap = cm.hot  # Change colormap if needed

        
        # Create the plot
        # fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        # ax.set_facecolor('black')  # Set background color of plot area to black
        # ax.set_aspect(1)

        im = ax.imshow(rasterized_img, cmap=cmap, norm=norm, origin="lower")
        

        # breakpoint()

        # Remove axis ticks
        ax.set_xticks([]), ax.set_yticks([])

        # # Scatter plot points on top of rasterized image
        if bscatter_plot:
            # for i in range(n_gauss):
            for ind, i in enumerate(np.unique(c)):
                ax.scatter(embedding_scaled[c == i, 0], embedding_scaled[c == i, 1], color=point_color[int(ind)], 
                        edgecolor='k', s=15, zorder = 5)
        
        # Create colorbar
        # plt.colorbar(im, ax=ax, orientation='vertical')
        # plt.colorbar(im, ax=ax)

        # fixed_x_min=-10, fixed_x_max=10, fixed_y_min=-10, fixed_y_max=10
        # Set fixed axis limits
        # ax.set_xlim(fixed_x_min, fixed_x_max)
        # ax.set_ylim(fixed_y_min, fixed_y_max)

        ax.axis('equal')
        plt.axis("off")
        

        # Save or display
        if output_path:
            plt.savefig(f"{output_path}_{blend_method}.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()


def plot_fully_connected_points(embedding, D, c, n_gauss, point_color, blend_methods=['average'], output_path=None, image_size=(500, 500), figsize = (10,10)):
    """
    Plots a fully connected graph using rasterized edges with different blending approaches.

    Parameters:
    - embedding: 2D coordinates of points
    - D: Original data for edge weight computation
    - c: Cluster labels
    - n_gauss: Number of clusters
    - point_color: List of colors for points
    - blend_methods: List of blending approaches to use (e.g., ['average', 'max', 'min'])
    - output_path: File path prefix for saving plots
    - image_size: Size of the output image
    - figsize: Figure size
    """
    edges = list(itertools.combinations(range(len(embedding)), 2))
    edge_weights = np.array([np.linalg.norm(D[i] - D[j]) for i, j in edges])  # Compute edge lengths once

    # Normalize edge colors
    cmap_w = cm.hot
    norm_weights = Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())
    edge_colors = cmap_w(norm_weights(edge_weights))

    # Convert embedding to image coordinates
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)
    scale_x = (image_size[1] - 1) / (max_x - min_x)
    scale_y = (image_size[0] - 1) / (max_y - min_y)

    embedding_scaled = np.array([
        [(x - min_x) * scale_x, (y - min_y) * scale_y]
        for x, y in embedding
    ])

    # # Determine padding limits
    # x_range, y_range = max_x - min_x, max_y - min_y
    # x_min, x_max = min_x - x_range * padding, max_x + x_range * padding
    # y_min, y_max = min_y - y_range * padding, max_y + y_range * padding

    # Prepare line segments
    line_segments = np.array([(embedding_scaled[i], embedding_scaled[j]) for i, j in edges])

    # Loop over selected blending methods
    for blend_method in blend_methods:
        print(f'Running blending: {blend_method} ')
        # Rasterize the image with blended edges
        rasterized_img = rasterize_lines(line_segments, edge_colors, image_size, blend_method=blend_method)

        # breakpoint()
        # Normalize for the color bar
        norm = Normalize(vmin=rasterized_img.min(), vmax=rasterized_img.max())
        cmap = cm.hot  # Change colormap if needed

        
        # Create the plot
        # fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        # ax.set_facecolor('black')  # Set background color of plot area to black
        # ax.set_aspect(1)

        im = ax.imshow(rasterized_img, cmap=cmap, norm=norm, origin="lower")
        ax.axis('equal')
        plt.axis("off")

        # breakpoint()

        # Remove axis ticks
        ax.set_xticks([]), ax.set_yticks([])

        # Scatter plot points on top of rasterized image
        for i in range(n_gauss):
            ax.scatter(
                embedding_scaled[c == i, 0], embedding_scaled[c == i, 1],
                color=point_color[i], edgecolor='k', s=50
            )
        
        # Create colorbar
        # plt.colorbar(im, ax=ax, orientation='vertical')
        # plt.colorbar(im, ax=ax)

        # fixed_x_min=-10, fixed_x_max=10, fixed_y_min=-10, fixed_y_max=10
        # Set fixed axis limits
        # ax.set_xlim(fixed_x_min, fixed_x_max)
        # ax.set_ylim(fixed_y_min, fixed_y_max)

        

        # Save or display
        if output_path:
            plt.savefig(f"{output_path}_{blend_method}.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()




# def plot_fully_connected_points(embedding, D, c, n_gauss, point_color, blend_method = 'average' , output_path=None, image_size=(500, 500), figsize = (10, 10)):
#     # Generate all possible edges
#     edges = list(itertools.combinations(range(len(embedding)), 2))
#     edge_weights = [np.linalg.norm(D[i] - D[j]) for i, j in reversed(edges)]

#     # Normalize edge colors
#     cmap_w = plt.get_cmap("hot")
#     norm_weights = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
#     edge_colors = [cmap_w(norm_weights(w)) for w in edge_weights]

#     # Convert embedding to image coordinates
#     min_x, min_y = np.min(embedding, axis=0)
#     max_x, max_y = np.max(embedding, axis=0)
#     scale_x = (image_size[1] - 1) / (max_x - min_x)
#     scale_y = (image_size[0] - 1) / (max_y - min_y)

#     embedding_scaled = np.array([
#         [(x - min_x) * scale_x, (y - min_y) * scale_y]
#         for x, y in embedding
#     ])

#     # Create line segments with rescaled coordinates
#     line_segments = [(embedding_scaled[i], embedding_scaled[j]) for i, j in reversed(edges)]

#     # Rasterize the image with blended edges
#     rasterized_img = rasterize_lines(line_segments, edge_colors, image_size, blend_method=blend_method)

#     # Normalize for the color bar
#     norm = Normalize(vmin=rasterized_img.min(), vmax=rasterized_img.max())
#     cmap = cm.hot  # You can change this to any colormap you prefer
    
#     # Create the plot
#     fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
#     # fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
#     # ax.set_aspect(1)

#     im = ax.imshow(rasterized_img,cmap=cmap, norm=norm, origin= "lower")
    
#     # Create the colorbar
#     cbar = plt.colorbar(im, ax=ax, orientation='vertical')
#     # cbar.set_label("Pixel Intensity")  # Label for colorbar
    
#     # Remove axis ticks
#     ax.set_xticks([]), ax.set_yticks([])

#     # Scatter plot points on top of rasterized image
#     for i in range(n_gauss):
#         ax.scatter(
#             embedding_scaled[c == i, 0], embedding_scaled[c == i, 1],
#             color=point_color[i], edgecolor='k', s=50
#         )

#     # Save the plot
#     if output_path:
#         plt.savefig(f"{output_path}_{blend_method}.png", dpi=300, bbox_inches="tight")
#         # plt.savefig(f"{output_path}_{blend_method}.png", dpi=300)
#         plt.close()
#     else:
#         plt.show()
#         plt.close()


def plot_fully_connected_points_edges(embedding, D, c, n_gauss, point_color, edge_sorts=['normal', 'ascending', 'descending'], output_path=None, image_size=(500, 500), figsize=(10,10)):
    """
    Plots a fully connected graph with edges color-coded by distance, allowing multiple sorting options.

    Parameters:
    - embedding: 2D coordinates of points
    - D: Original data for edge weight computation
    - c: Cluster labels
    - n_gauss: Number of clusters
    - point_color: List of colors for points
    - edge_sorts: List of sorting options to apply ('normal', 'ascending', 'descending')
    - output_path: File path prefix for saving plots
    - image_size: Size of the output image
    - figsize: Figure size
    """
    edges = list(itertools.combinations(range(len(embedding)), 2))
    edge_weights = np.array([np.linalg.norm(D[i] - D[j]) for i, j in edges])  # Compute edge weights once

    # Sorting methods
    sort_methods = {
        'normal': np.arange(len(edge_weights)),  # No sorting
        'ascending': np.argsort(edge_weights),   # Shortest edges first
        'descending': np.argsort(edge_weights)[::-1]  # Longest edges first
    }

    # Loop over selected sorting orders
    for edge_sort in edge_sorts:
        if edge_sort not in sort_methods:
            print(f"Skipping invalid sort option: {edge_sort}")
            continue

        sorted_edge_indices = sort_methods[edge_sort]
        sorted_edges = [edges[i] for i in sorted_edge_indices]
        sorted_edge_weights = edge_weights[sorted_edge_indices]

        # Normalize weights for color mapping
        edge_weights_norm = (sorted_edge_weights - sorted_edge_weights.min()) / (sorted_edge_weights.max() - sorted_edge_weights.min())
        cmap = cm.hot
        edge_colors = cmap(edge_weights_norm)

        # Prepare edge segments
        edge_segments = np.array([(embedding[i], embedding[j]) for i, j in sorted_edges])

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        # fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        # ax.set_aspect(1)

        # Draw edges
        lc = LineCollection(edge_segments, colors=edge_colors, alpha=0.7, linewidths=0.5)
        ax.add_collection(lc)

        # Scatter plot points
        for i in range(n_gauss):
            ax.scatter(
                embedding[c == i, 0], embedding[c == i, 1],
                color=point_color[i], edgecolor='k', s=50
            )

        # Remove axis labels
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis('equal')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.05)

        # Save or display
        if output_path:
            plt.savefig(f"{output_path}_{edge_sort}.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()