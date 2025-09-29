import streamlit as st
import pickle

# Load the pickle file
pickle_path = "pickle_files/notebook.pkl"
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# Ensure the pickle contains X and y
if not isinstance(data, dict) or "X" not in data or "y" not in data:
    st.error("Pickle file is not in the expected format.")
else:
    X = data["X"]
    y = data["y"]

    st.title("Titanic Dataset Prediction Example")

    st.subheader("Input Features (X):")
    st.dataframe(X.head())

    st.subheader("Target (y):")
    st.dataframe(y.head())
