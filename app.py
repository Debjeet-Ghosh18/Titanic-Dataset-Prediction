import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load pickle file
with open("pickle_files/notebook.pkl", "rb") as f:
    data = pickle.load(f)

X = data["X"]
y = data["y"]

# Train a simple model (Logistic Regression)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details to predict survival:")

# Inputs
pclass = st.selectbox("Pclass (Ticket Class)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)

# Prepare input
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Age": [age],
    "Sex": [sex]
})

# One-hot encode 'Sex' same as training
input_data = pd.get_dummies(input_data, columns=["Sex"], drop_first=True)

# Ensure same columns as training
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[X.columns]  # Reorder columns

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Survived! (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Did not survive. (Probability: {prob:.2f})")
