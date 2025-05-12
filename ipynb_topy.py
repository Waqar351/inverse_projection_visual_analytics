import nbformat
from nbconvert import PythonExporter

# Load the notebook
with open("Jacobian_numerical_symbolic_testing.ipynb", "r") as f:
    notebook = nbformat.read(f, as_version=4)

# Convert to Python script
exporter = PythonExporter()
source, _ = exporter.from_notebook_node(notebook)

# Save the Python script
with open("your_notebook.py", "w") as f:
    f.write(source)
