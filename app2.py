import streamlit as st
import pandas as pd
import joblib

# --- MODEL & ASSETS LOADING ---
@st.cache_resource
def load_model_assets():
    """Load model and model columns from .joblib files."""
    try:
        with open('model.joblib', 'rb') as f:
            model = joblib.load(f)
        with open('model_columns.joblib', 'rb') as f:
            model_columns = joblib.load(f)
        return model, model_columns
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

model, model_columns = load_model_assets()

# --- APP CONFIG ---
st.set_page_config(page_title="🏠 House Price Predictor", layout="wide")
st.title("🏠 House Price Predictor")
st.markdown("Enter house details in the sidebar to predict the price.")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Property Details")

    # Location dependent dropdown example
    CITY_STREET_MAP = {
        "New York": ["Broadway", "5th Avenue", "Wall Street"],
        "Los Angeles": ["Sunset Blvd", "Rodeo Drive", "Hollywood Blvd"]
    }
    city = st.selectbox("City", options=list(CITY_STREET_MAP.keys()))
    street = st.selectbox("Street", options=CITY_STREET_MAP[city])

    # Numerical inputs
    area = st.number_input("Area (sqft)", 100, 10000, 1500, step=50)
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)
    parking = st.number_input("Parking Spots", 0, 10, 2)

    # Categorical features
    mainroad = st.selectbox("Mainroad Access", options=["Yes", "No"])
    basement = st.selectbox("Basement", options=["Yes", "No"])

    # Predict button
    predict_button = st.button("Predict Price")

# --- PREDICTION LOGIC ---
if model is not None and model_columns is not None and predict_button:
    user_input = {
        'City': [city],
        'Street': [street],
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'parking': [parking],
        'mainroad': [1 if mainroad=="Yes" else 0],
        'basement': [1 if basement=="Yes" else 0]
    }

    input_df = pd.DataFrame(user_input)

    try:
        # Align input with model columns
        final_df = pd.DataFrame(columns=model_columns)
        final_df = pd.concat([final_df, input_df]).fillna(0)
        final_df = final_df[model_columns]

        # Make prediction
        price = model.predict(final_df)[0]
        st.subheader("Predicted House Price:")
        st.success(f"💰 ₹ {price:,.0f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

elif predict_button:
    st.error("Model or columns not loaded. Check your model files.")
