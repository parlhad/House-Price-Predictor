import streamlit as st
import pandas as pd
import joblib
import os

# --- APP CONFIG ---
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = None
    try:
        if os.path.exists("model.joblib"):
            model = joblib.load("model.joblib")
        else:
            st.warning("⚠️ Model file not found. Upload model.joblib.")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
    return model

model = load_model()

# --- APP HEADER ---
st.title("🏠 House Price Predictor")
st.markdown("Provide property details below and get an instant price prediction.")

# --- USER INPUT GRID (4x2) ---
CITY_STREET_MAP = {
    "New York": ["Broadway", "5th Avenue", "Wall Street"],
    "Los Angeles": ["Sunset Blvd", "Rodeo Drive", "Hollywood Blvd"]
}

st.subheader("Property Details")

# --- FIRST ROW (City, Street, Sqft, Bedrooms) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    city = st.selectbox("City", options=list(CITY_STREET_MAP.keys()))
with col2:
    street = st.selectbox("Street", options=CITY_STREET_MAP[city])
with col3:
    sqft = st.number_input("Area (sqft)", 100, 10000, 1500, step=50)
with col4:
    bed = st.number_input("Bedrooms", 1, 10, 3)

# --- SECOND ROW (Bathrooms, Parking, Mainroad, Basement) ---
col5, col6, col7, col8 = st.columns(4)
with col5:
    bath = st.number_input("Bathrooms", 1, 10, 2)
with col6:
    parking = st.number_input("Parking Spots", 0, 10, 2)
with col7:
    mainroad = st.selectbox("Mainroad Access", ["Yes", "No"])
with col8:
    basement = st.selectbox("Basement", ["Yes", "No"])

# --- PREDICT BUTTON ---
predict_button = st.button("🔮 Predict Price")

# --- PREDICTION LOGIC ---
if predict_button:
    if model is None:
        st.error("❌ Model not loaded.")
    else:
        try:
            # Build input DataFrame exactly like training features
            user_input = pd.DataFrame([{
                "citi": city,
                "street": street,
                "sqft": sqft,
                "bed": bed,
                "bath": bath,
                "n_citi": parking,
                "mainroad": 1 if mainroad == "Yes" else 0,
                "basement": 1 if basement == "Yes" else 0
            }])

            # Directly pass raw input to pipeline
            price = model.predict(user_input)[0]

            st.subheader("💰 Predicted House Price:")
            st.success(f"₹ {price:,.0f}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)
