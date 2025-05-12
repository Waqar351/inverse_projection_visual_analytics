import matplotlib.pyplot as plt
from matplotlib.table import Table

# Test data
table_data = [
    ["Metric", "Value"],
    ["Accuracy", "0.95"],
    ["Precision", "0.90"],
    ["Recall", "0.85"]
]
names_to_highlight = ["Accuracy", "Recall"]

# Create a simple figure and axis
fig, ax = plt.subplots(figsize=(6, 3))
ax.axis("tight")
ax.axis("off")

# Add the table
table = ax.table(
    cellText=table_data,
    cellLoc="center",
    loc="center"
)

# Iterate through rows and apply facecolor and bold text
for row_idx, row_data in enumerate(table_data):
    if row_idx == 0:  # Skip the header row
        continue
    if row_data[0] in names_to_highlight:
        for col_idx in range(len(row_data)):
            cell = table[row_idx, col_idx]
            cell.set_facecolor("#FFA500")  # Set the background color
            cell.set_text_props(color="black", weight="bold")  # Make text bold

# Display the plot
plt.show()
