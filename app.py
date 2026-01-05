import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.title("🫀 Heart Disease Prediction")

model = load_model("heart_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

inputs = []
labels = [
    "Age", "Sex", "Chest Pain", "BP", "Cholesterol",
    "FBS > 120", "EKG", "Max HR", "Exercise Angina",
    "ST Depression", "Slope", "Vessels", "Thallium"
]

for label in labels:
    inputs.append(st.number_input(label, value=0.0))

if st.button("Predict"):
    data = np.array([inputs])
    data = scaler.transform(data)

    prob = model.predict(data)[0][0]
    result = "Heart Disease" if prob > 0.5 else "No Heart Disease"

    st.success(f"Result: {result}")
    st.info(f"Probability: {prob:.2f}")
