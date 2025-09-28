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
    "Los Angeles": ["Sunset Blvd", "Rodeo Drive", "Hollywood Blvd"],
    "Salton City, CA": ["1317 Van Buren Avenue"],
    "Brawley, CA": ["124 C Street W"],
    "Imperial, CA": ["2304 Clark Road"],
    "Calexico, CA": ["755 Brawley Avenue"],
    "Gorman, CA": ["2207 R Carrillo Court"],
    "Frazier Park, CA": ["1100 CAMILIA Street"],
    "Rosamond, CA": ["803 Chaparral Court"],
    "Kernville, CA": ["2306 Lark Court"],
    "Tehachapi, CA": ["38833 Gorman Post Road"],
    "Arvin, CA": ["8072 Cuddy Valley Road"],
    "California City, CA": ["818 155th Street W"],
    "Bakersfield, CA": ["12869 Sierra Way"],
    "Delano, CA": ["230 Gaskell Road"],
    "Lebec, CA": ["11265 Steinhoff Road"],
    "Pine Mountain Club, CA": ["12471 Boy Scout Camp Road"],
    "Inyokern, CA": ["0 Oak Creek rd"],
    "Mojave, CA": ["226 Spruce Street"],
    "Boron, CA": ["632 Grove Street"],
    "Stallion Springs, CA": ["9225 Rea Avenue"],
    "Ridgecrest, CA": ["6005 N charmain"],
    "Keene, CA": ["3405 San Carlos"],
    "Caliente, CA": ["19800 Nonie Court"],
    "Bear Valley Springs, CA": ["11001 Meacham Road"],
    "Wofford Heights, CA": ["3933 Encino"],
    "Studio City, CA": ["347 Cypress Street"],
    "Glendale, CA": ["3305 Reeder Avenue"],
    "Eagle Rock, CA": ["2001 WILLIAM F. HALSEY Avenue"],
    "Rancho Palos Verdes, CA": ["1835 Kavalier Ct"],
    "Woodland Hills, CA": ["3409 Glendower Street"],
    "Montebello, CA": ["2013 College Drive"],
    "Los Angeles, CA": ["8308 Ailanthus Court"],
    "Culver City, CA": ["11801 Pasture Avenue"],
    "Long Beach, CA": ["8315 Hemlock Court"],
    "Walnut, CA": ["2131 OAK Street"],
    "Rowland Heights, CA": ["608 Poinsettia Avenue"],
    "Porter Ranch, CA": ["21413 Yerba Boulevard"],
    "Newhall, CA": ["2822 Dixie Street"],
    "Covina, CA": ["9224 Evelyn Avenue"],
    "Santa Monica, CA": ["2504 Lebec Oaks Road"],
    "San Marino, CA": ["16521 Mil Potrero"],
    "Agoura Hills, CA": ["200 Oak Street"],
    "Stevenson Ranch, CA": ["21110 S. Charlene Place"],
    "La Canada Flintridge, CA": ["15808 Mil Potrero"],
    "Arcadia, CA": ["14700 Tumbleweed Place"],
    "Pasadena, CA": ["7108 Lakeview Drive"],
    "El Segundo, CA": ["4400 Vista Mesa Drive"],
    "El Monte, CA": ["2309 Symonds Drive"],
    "Torrance, CA": ["228 W D Street"],
    "Altadena, CA": ["1942 El Rey Street"],
    "Encino, CA": ["3633 Capri Court"],
    "Downey, CA": ["21410 Brook Drive"],
    "Sherman Oaks, CA": ["1456 BRADFORD Avenue"],
    "Monrovia, CA": ["2737 Hempstead Lane"]
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
