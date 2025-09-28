import streamlit as st
import pandas as pd
import joblib

# --- APP CONFIG ---
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    """Load the prediction model from file."""
    try:
        model = joblib.load("model.joblib")
        return model
    except FileNotFoundError:
        st.error("⚠️ **Model file not found.** Please make sure 'model.joblib' is in the same folder as your app.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

# --- DATA ---
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

# --- APP HEADER ---
st.title("🏠 House Price Predictor")

# --- ADD IMAGE HERE ---
st.image("Screenshot_2024-03-20_at_2.51.52_PM.png", caption="The Rising Cost of Living", use_column_width=True)

st.markdown("Enter the property details below to get an instant price prediction.")

# --- USER INPUTS ---
st.subheader("📍 Location")
col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("City", options=list(CITY_STREET_MAP.keys()))
with col2:
    street = st.selectbox("Street", options=CITY_STREET_MAP[city])

st.subheader("📐 Property Size & Rooms")
sqft = st.slider("Area (sqft)", min_value=100, max_value=10000, value=1500, step=50)

col3, col4, col5 = st.columns(3)
with col3:
    bed = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
with col4:
    bath = st.slider("Bathrooms", min_value=1, max_value=10, value=2)
with col5:
    parking = st.slider("Parking Spots", min_value=0, max_value=10, value=2)

st.subheader("✨ Additional Features")
col6, col7 = st.columns(2)
with col6:
    mainroad = st.toggle("Mainroad Access", value=True)
with col7:
    basement = st.toggle("Has a Basement", value=False)

st.divider()

# --- PREDICTION ---
predict_button = st.button("🔮 Predict Price", use_container_width=True)

if predict_button:
    if model is None:
        st.error("❌ **Prediction failed.** The model is not loaded.")
    else:
        try:
            user_input = pd.DataFrame([{
                "citi": city,
                "street": street,
                "sqft": sqft,
                "bed": bed,
                "bath": bath,
                "n_citi": parking,
                "mainroad": 1 if mainroad else 0,
                "basement": 1 if basement else 0
            }])

            price = model.predict(user_input)[0]

            st.subheader("💰 Predicted House Price:")
            st.success(f"**₹ {price:,.0f}**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
