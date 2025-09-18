import pandas as pd
import joblib
import streamlit as st
import time

# --- Load cleaned dataset ---
data = pd.read_csv("cleaned_fixed.csv")

# --- Extract features (exclude target Price) ---
feature_columns = data.columns.drop('Price')

# --- Load original dataset to extract City/District ---
original = pd.read_csv("all_results.csv")
original[['City', 'District']] = original['property_location'].str.split(',', expand=True)
original['City'] = original['City'].str.strip()
original['District'] = original['District'].str.strip()
city_to_district = original[['City', 'District']].drop_duplicates().set_index('City')['District'].to_dict()

# --- Streamlit UI ---
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

st.markdown("""
<style>
.stApp { background-color: #1e1e1e; color: #ffffff; }
h1 { color: #00bfa5; }
.stButton>button { background-color: #00bfa5; color: black; height:3em; width:100%; font-size:16px; font-weight:bold; }
.big-font { font-size: 32px; font-weight: bold; color: #ffeb3b; }
.range-font { font-size: 24px; font-weight: semi-bold; color: #03a9f4; }
.stSelectbox, .stNumberInput { color: black; }
</style>
""", unsafe_allow_html=True)

st.title("🏠 House Price Prediction")

# City selectbox
cities = sorted(list(city_to_district.keys()))
city = st.selectbox("Select City", cities)
district = city_to_district[city]
st.write(f"District: {district}")

# Property type selectbox
property_types = [col.replace('Property_type_', '') for col in feature_columns if col.startswith('Property_type_')]
property_type = st.selectbox("Property Type", property_types)

# Furnished selectbox
furnished_options = [col.replace('Is_furnished_', '') for col in feature_columns if col.startswith('Is_furnished_')]
furnished = st.selectbox("Furnished Status", furnished_options)

# Numeric inputs
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1)
size = st.number_input("Size (sqm)", min_value=10, max_value=1000, value=100)

# Predict button
if st.button("Predict Price"):
    with st.spinner('Predicting price... 🏠'):
        time.sleep(1)  # simulate loading

        # Prepare input row with **only feature columns**
        input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

        # Set numeric values
        input_data.at[0, 'Bedrooms'] = bedrooms
        input_data.at[0, 'Bathrooms'] = bathrooms
        input_data.at[0, 'Size'] = size

        # Set city
        city_col = f"City_{city}"
        if city_col in input_data.columns:
            input_data.at[0, city_col] = 1

        # Set property type
        property_col = f"Property_type_{property_type}"
        if property_col in input_data.columns:
            input_data.at[0, property_col] = 1

        # Set furnished
        furnished_col = f"Is_furnished_{furnished}"
        if furnished_col in input_data.columns:
            input_data.at[0, furnished_col] = 1

        # Load model
        model = joblib.load("rf_model_finetuned.pkl")
        price = model.predict(input_data)[0]

        # Rounded value
        rounded_price = round(price / 1000) * 1000

        # Hybrid range: max of 5k$ or 2% of price
        range_value = round(max(5000, rounded_price * 0.02) / 1000) * 1000

        lower = max(0, rounded_price - range_value)
        upper = rounded_price + range_value


        # Display results
        st.markdown(f"<div class='big-font'>🏠 Predicted Price: ${rounded_price:,.0f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='range-font'>💰 Estimated Range: ${lower:,.0f} - ${upper:,.0f}</div>", unsafe_allow_html=True)
