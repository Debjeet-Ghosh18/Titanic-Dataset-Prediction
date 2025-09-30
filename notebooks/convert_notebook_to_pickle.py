import nbformat
from nbconvert import PythonExporter
import pandas as pd
import pickle

# Load the notebook
notebook_path = "notebooks/Titanic Dataset.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Convert notebook to Python code
exporter = PythonExporter()
python_code, _ = exporter.from_notebook_node(nb)

# Execute notebook code in a separate namespace
namespace = {}
exec(python_code, namespace)

# Get dataframe
df = namespace.get("df")
if df is None:
    raise ValueError("DataFrame 'df' not found in the notebook.")

# Prepare features and target
X = df[["Pclass", "Sex", "Age"]]
y = df["Survived"]

# Encode categorical column 'Sex'
X = pd.get_dummies(X, columns=["Sex"], drop_first=True)

# ✅ Save both X and y together
pickle_file_path = "pickle_files/notebook.pkl"
with open(pickle_file_path, "wb") as f:
    pickle.dump({"X": X, "y": y}, f)

print("✅ Corrected pickle saved at:", pickle_file_path)
