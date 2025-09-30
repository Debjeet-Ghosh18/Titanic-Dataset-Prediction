import nbformat
from nbconvert import PythonExporter
import pandas as pd
import pickle

# Path to your notebook
notebook_path = "notebooks/Titanic Dataset.ipynb"

# Load notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Convert notebook to python code
exporter = PythonExporter()
python_code, _ = exporter.from_notebook_node(nb)

# Run the code in isolated namespace
namespace = {}
exec(python_code, namespace)

# Get dataframe
df = namespace.get("df")
if df is None:
    raise ValueError("DataFrame 'df' not found in notebook. Make sure df is defined inside Titanic Dataset.ipynb")

# Prepare X and y
X = df[["Pclass", "Sex", "Age"]]
y = df["Survived"]

# Encode 'Sex'
X = pd.get_dummies(X, columns=["Sex"], drop_first=True)

# ✅ Save dictionary with X and y
with open("pickle_files/notebook.pkl", "wb") as f:
    pickle.dump({"X": X, "y": y}, f)

print("✅ Pickle created: pickle_files/notebook.pkl")
