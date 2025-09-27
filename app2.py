# app.py

import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle

# --- Page Configuration ---
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Predictor")
st.write("Predict house prices using Linear Regression and Decision Tree models.")

# --- Load Models Safely ---
@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as f:
        return cloudpickle.load(f)

LR_model = load_model("LR_model.pkl")
DT_model = load_model("DT_model.pkl")

# --- User Input ---
st.sidebar.header("Enter House Details")

n_citi = st.sidebar.number_input("Number of Cities", min_value=0, step=1, value=1)
bed = st.sidebar.number_input("Number of Bedrooms", min_value=0, step=1, value=3)
bath = st.sidebar.number_input("Number of Bathrooms", min_value=0, step=1, value=2)
sqft = st.sidebar.number_input("Square Footage", min_value=0, step=10, value=1500)
citi = st.sidebar.text_input("City", value="Los Angeles")
street = st.sidebar.text_input("Street", value="Main St")

input_data = pd.DataFrame({
    'n_citi': [n_citi],
    'bed': [bed],
    'bath': [bath],
    'sqft': [sqft],
    'citi': [citi],
    'street': [street]
})

# --- Prediction ---
if st.button("Predict Price"):
    pred_lr = LR_model.predict(input_data)[0]
    pred_dt = DT_model.predict(input_data)[0]

    st.success(f"Linear Regression Prediction: ${pred_lr:,.2f}")
    st.success(f"Decision Tree Prediction: ${pred_dt:,.2f}")

# --- Optional: Compare Models ---
st.info("Linear Regression tends to generalize better for larger datasets.")
st.info("Decision Tree may overfit smaller datasets but can capture non-linear patterns.")
