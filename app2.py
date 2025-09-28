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

# --- MODEL & ASSETS LOADING ---
@st.cache_resource
def load_model_assets():
    """Load model and model columns from .joblib files safely."""
    model, model_columns = None, None
    try:
        if os.path.exists("model.joblib") and os.path.exists("model_columns.joblib"):
            model = joblib.load("model.joblib")
            model_columns = joblib.load("model_columns.joblib")
        else:
            st.warning("⚠️ Model files not found. Please upload model.joblib and model_columns.joblib.")
    except Exception as e:
        st.error(f"❌ Error loading model assets: {e}")
    return model, model_columns

model, model_columns = load_model_assets()

# --- APP HEADER ---
st.title("🏠 House Price Predictor")
st.markdown("Provide property details in the sidebar and get an instant price prediction.")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Property Details")

    CITY_STREET_MAP = {
        "New York": ["Broadway", "5th Avenue", "Wall Street"],
        "Los Angeles": ["Sunset Blvd", "Rodeo Drive", "Hollywood Blvd"]
    }

    city = st.selectbox("City", options=list(CITY_STREET_MAP.keys()))
    street = st.selectbox("Street", options=CITY_STREET_MAP[city])

    sqft = st.number_input("Area (sqft)", 100, 10000, 1500, step=50)
    bed = st.number_input("Bedrooms", 1, 10, 3)
    bath = st.number_input("Bathrooms", 1, 10, 2)
    parking = st.number_input("Parking Spots", 0, 10, 2)

    mainroad = st.selectbox("Mainroad Access", ["Yes", "No"])
    basement = st.selectbox("Basement", ["Yes", "No"])

    predict_button = st.button("🔮 Predict Price")

# --- PREDICTION LOGIC ---
if predict_button:
    if model is None or model_columns is None:
        st.error("❌ Model not available. Please check your deployment files.")
    else:
        try:       # Match training column names
user_input = {
    'citi': [city],
    'street': [street],
    'sqft': [area],
    'bed': [bedrooms],
    'bath': [bathrooms],
    'n_citi': [parking],   # assuming 'n_citi' was parking or similar feature
    'mainroad': [1 if mainroad == "Yes" else 0],
    'basement': [1 if basement == "Yes" else 0]
}
input_df = pd.DataFrame(user_input)


            # If model expects 'n_citi', create a numeric encoding
            if "n_citi" in model_columns:
                city_map = {c: i for i, c in enumerate(CITY_STREET_MAP.keys())}
                input_df["n_citi"] = input_df["citi"].map(city_map).fillna(0).astype(int)

            # Align columns with model
            final_df = pd.DataFrame(columns=model_columns)
            final_df = pd.concat([final_df, input_df], ignore_index=True).fillna(0)
            final_df = final_df[model_columns]  # ensure correct order

            # Make prediction
            price = model.predict(final_df)[0]

            st.subheader("💰 Predicted House Price:")
            st.success(f"₹ {price:,.0f}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)  # show traceback for debugging
