import streamlit as st
import pandas as pd
import joblib
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="SoCal Real Estate Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching the Model ---
# SIMPLIFIED: We only need to load the single pipeline file.
@st.cache_resource
def load_model():
    """Loads the trained pipeline model from disk."""
    try:
        # CHANGED: Load the correct model file name.
        model = joblib.load('final_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'final_model.joblib' is in the same directory.")
        return None

# Load the model, which is the entire pipeline
pipeline = load_model()

# --- UI Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #F5F5F5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            font-size: 16px;
            font-weight: bold;
        }
        .stMetric {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            border-radius: 12px;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)


# --- Main Application ---
st.title("🏡 SoCal Real Estate Price Predictor")
st.markdown("Enter the property details to get an estimated market value based on our trained model.")
st.markdown("---")


# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Enter Property Details")

    # --- CHANGED: Input fields now match the training data columns ---
    n_citi = st.number_input("City Code (n_citi)", min_value=0, value=152, step=1, help="Enter the numerical code for the city.")
    sqft = st.slider("Area (sqft)", min_value=500, max_value=10000, value=1500, step=50)
    bed = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
    bath = st.slider("Bathrooms", min_value=1.0, max_value=8.0, value=2.0, step=0.5)

    st.markdown("---")

    # For categorical features, text input is more flexible
    citi = st.text_input("City Name", placeholder="e.g., Imperial, CA")
    street = st.text_input("Street Address", placeholder="e.g., 2304 Clark Road")

    predict_button = st.button("Predict Price", use_container_width=True)


# --- Prediction Logic and Display ---
if predict_button and pipeline is not None:
    # 1. Create a dictionary from user inputs
    # The keys MUST match the column names of the original DataFrame (X)
    input_data = {
        'n_citi': n_citi,
        'bed': bed,
        'bath': bath,
        'sqft': sqft,
        'citi': citi,
        'street': street
    }

    # 2. Convert to a DataFrame
    input_df = pd.DataFrame([input_data])

    # 3. Make the prediction
    # SIMPLIFIED: The pipeline handles all preprocessing (scaling, encoding) automatically.
    # We no longer need get_dummies or reindex.
    with st.spinner('Calculating...'):
        time.sleep(1)
        prediction = pipeline.predict(input_df)

    # 4. Display the result
    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://i.imgur.com/J2g4Vha.png", width=200)

    with col2:
        st.subheader("Predicted Property Value")
        # Format the prediction as currency
        formatted_price = f"${prediction[0]:,.0f}"
        st.metric(label="Estimated Price", value=formatted_price)
        st.success("The prediction is based on the features provided. Market conditions can influence the final price.")

    st.balloons()

elif not pipeline:
    st.warning("Please make sure the model file is loaded correctly before trying to predict.")
