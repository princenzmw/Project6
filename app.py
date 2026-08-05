import streamlit as st
import pandas as pd
import joblib


# 1. Cache the model loading to prevent CPU throttling
@st.cache_resource
def load_artifacts():
    model = joblib.load("kenya_tea_model.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le


# Load artifacts into memory (runs only once)
model, le = load_artifacts()

# 2. Define the County Means Dictionary for Dynamic Autofill
# These values represent the average metrics for each county based on the dataset
COUNTY_MEANS = {
    "Baringo": {
        "altitude_m": 1760.87,
        "annual_rainfall_mm": 798.29,
        "temp_max_c": 29.53,
        "temp_min_c": 15.96,
        "soil_ph": 7.6,
        "humidity_pct": 58.05,
        "soil_nitrogen_pct": 0.67,
        "organic_carbon_pct": 1.05,
        "slope_degrees": 9.91,
    },
    "Bomet": {
        "altitude_m": 1831.74,
        "annual_rainfall_mm": 1438.91,
        "temp_max_c": 24.03,
        "temp_min_c": 14.09,
        "soil_ph": 5.08,
        "humidity_pct": 73.34,
        "soil_nitrogen_pct": 1.87,
        "organic_carbon_pct": 2.93,
        "slope_degrees": 7.6,
    },
    "Bungoma": {
        "altitude_m": 1630.93,
        "annual_rainfall_mm": 1510.55,
        "temp_max_c": 25.49,
        "temp_min_c": 14.96,
        "soil_ph": 5.93,
        "humidity_pct": 71.95,
        "soil_nitrogen_pct": 1.4,
        "organic_carbon_pct": 1.94,
        "slope_degrees": 7.22,
    },
    "Elgeyo Marakwet": {
        "altitude_m": 2087.65,
        "annual_rainfall_mm": 1150.81,
        "temp_max_c": 21.55,
        "temp_min_c": 11.17,
        "soil_ph": 5.8,
        "humidity_pct": 61.71,
        "soil_nitrogen_pct": 1.29,
        "organic_carbon_pct": 1.93,
        "slope_degrees": 19.31,
    },
    "Embu": {
        "altitude_m": 1520.21,
        "annual_rainfall_mm": 1286.81,
        "temp_max_c": 24.45,
        "temp_min_c": 14.84,
        "soil_ph": 5.65,
        "humidity_pct": 67.7,
        "soil_nitrogen_pct": 1.36,
        "organic_carbon_pct": 2.18,
        "slope_degrees": 10.84,
    },
    "Garissa": {
        "altitude_m": 278.93,
        "annual_rainfall_mm": 260.57,
        "temp_max_c": 37.35,
        "temp_min_c": 24.04,
        "soil_ph": 8.47,
        "humidity_pct": 37.51,
        "soil_nitrogen_pct": 0.25,
        "organic_carbon_pct": 0.26,
        "slope_degrees": 1.35,
    },
    "Homa Bay": {
        "altitude_m": 1388.72,
        "annual_rainfall_mm": 1243.94,
        "temp_max_c": 29.4,
        "temp_min_c": 18.37,
        "soil_ph": 6.33,
        "humidity_pct": 67.68,
        "soil_nitrogen_pct": 0.83,
        "organic_carbon_pct": 1.36,
        "slope_degrees": 6.09,
    },
    "Isiolo": {
        "altitude_m": 889.01,
        "annual_rainfall_mm": 452.37,
        "temp_max_c": 33.73,
        "temp_min_c": 20.65,
        "soil_ph": 7.96,
        "humidity_pct": 43.25,
        "soil_nitrogen_pct": 0.42,
        "organic_carbon_pct": 0.42,
        "slope_degrees": 4.59,
    },
    "Kajiado": {
        "altitude_m": 1442.07,
        "annual_rainfall_mm": 444.03,
        "temp_max_c": 30.66,
        "temp_min_c": 15.54,
        "soil_ph": 7.8,
        "humidity_pct": 46.46,
        "soil_nitrogen_pct": 0.48,
        "organic_carbon_pct": 0.72,
        "slope_degrees": 6.5,
    },
    "Kakamega": {
        "altitude_m": 1599.0,
        "annual_rainfall_mm": 1828.8,
        "temp_max_c": 26.33,
        "temp_min_c": 15.82,
        "soil_ph": 5.59,
        "humidity_pct": 77.45,
        "soil_nitrogen_pct": 1.65,
        "organic_carbon_pct": 2.21,
        "slope_degrees": 5.63,
    },
    "Kericho": {
        "altitude_m": 1948.71,
        "annual_rainfall_mm": 1786.85,
        "temp_max_c": 23.13,
        "temp_min_c": 13.81,
        "soil_ph": 4.88,
        "humidity_pct": 77.2,
        "soil_nitrogen_pct": 2.27,
        "organic_carbon_pct": 3.65,
        "slope_degrees": 6.94,
    },
    "Kiambu": {
        "altitude_m": 1895.8,
        "annual_rainfall_mm": 1259.11,
        "temp_max_c": 24.51,
        "temp_min_c": 14.09,
        "soil_ph": 5.63,
        "humidity_pct": 67.25,
        "soil_nitrogen_pct": 1.39,
        "organic_carbon_pct": 2.11,
        "slope_degrees": 9.17,
    },
    "Kilifi": {
        "altitude_m": 262.21,
        "annual_rainfall_mm": 1000.69,
        "temp_max_c": 31.6,
        "temp_min_c": 23.76,
        "soil_ph": 7.14,
        "humidity_pct": 72.91,
        "soil_nitrogen_pct": 0.54,
        "organic_carbon_pct": 0.96,
        "slope_degrees": 3.72,
    },
    "Kirinyaga": {
        "altitude_m": 1809.75,
        "annual_rainfall_mm": 1428.45,
        "temp_max_c": 22.72,
        "temp_min_c": 12.99,
        "soil_ph": 5.11,
        "humidity_pct": 71.1,
        "soil_nitrogen_pct": 1.67,
        "organic_carbon_pct": 2.58,
        "slope_degrees": 9.61,
    },
    "Kisii": {
        "altitude_m": 1662.75,
        "annual_rainfall_mm": 1643.52,
        "temp_max_c": 25.52,
        "temp_min_c": 14.98,
        "soil_ph": 5.26,
        "humidity_pct": 74.5,
        "soil_nitrogen_pct": 1.54,
        "organic_carbon_pct": 2.41,
        "slope_degrees": 13.56,
    },
    "Kisumu": {
        "altitude_m": 1358.52,
        "annual_rainfall_mm": 1152.45,
        "temp_max_c": 29.38,
        "temp_min_c": 18.35,
        "soil_ph": 6.34,
        "humidity_pct": 67.54,
        "soil_nitrogen_pct": 0.88,
        "organic_carbon_pct": 1.28,
        "slope_degrees": 3.77,
    },
    "Kitui": {
        "altitude_m": 1015.7,
        "annual_rainfall_mm": 686.53,
        "temp_max_c": 30.98,
        "temp_min_c": 20.3,
        "soil_ph": 7.66,
        "humidity_pct": 49.32,
        "soil_nitrogen_pct": 0.44,
        "organic_carbon_pct": 0.66,
        "slope_degrees": 7.86,
    },
    "Kwale": {
        "altitude_m": 275.99,
        "annual_rainfall_mm": 1070.8,
        "temp_max_c": 31.05,
        "temp_min_c": 23.24,
        "soil_ph": 6.86,
        "humidity_pct": 74.91,
        "soil_nitrogen_pct": 0.65,
        "organic_carbon_pct": 1.08,
        "slope_degrees": 4.23,
    },
    "Laikipia": {
        "altitude_m": 2016.95,
        "annual_rainfall_mm": 675.67,
        "temp_max_c": 25.96,
        "temp_min_c": 12.58,
        "soil_ph": 7.55,
        "humidity_pct": 52.9,
        "soil_nitrogen_pct": 0.7,
        "organic_carbon_pct": 1.11,
        "slope_degrees": 4.67,
    },
    "Lamu": {
        "altitude_m": 47.71,
        "annual_rainfall_mm": 837.22,
        "temp_max_c": 33.33,
        "temp_min_c": 25.31,
        "soil_ph": 8.03,
        "humidity_pct": 80.32,
        "soil_nitrogen_pct": 0.41,
        "organic_carbon_pct": 0.71,
        "slope_degrees": 1.44,
    },
    "Machakos": {
        "altitude_m": 1452.38,
        "annual_rainfall_mm": 743.22,
        "temp_max_c": 27.62,
        "temp_min_c": 16.96,
        "soil_ph": 6.89,
        "humidity_pct": 53.34,
        "soil_nitrogen_pct": 0.71,
        "organic_carbon_pct": 1.01,
        "slope_degrees": 12.21,
    },
    "Makueni": {
        "altitude_m": 1176.96,
        "annual_rainfall_mm": 605.4,
        "temp_max_c": 29.82,
        "temp_min_c": 18.35,
        "soil_ph": 7.46,
        "humidity_pct": 50.84,
        "soil_nitrogen_pct": 0.55,
        "organic_carbon_pct": 0.75,
        "slope_degrees": 8.38,
    },
    "Mandera": {
        "altitude_m": 368.68,
        "annual_rainfall_mm": 229.6,
        "temp_max_c": 39.19,
        "temp_min_c": 26.76,
        "soil_ph": 8.54,
        "humidity_pct": 37.31,
        "soil_nitrogen_pct": 0.18,
        "organic_carbon_pct": 0.25,
        "slope_degrees": 1.69,
    },
    "Marsabit": {
        "altitude_m": 928.59,
        "annual_rainfall_mm": 362.49,
        "temp_max_c": 31.21,
        "temp_min_c": 18.54,
        "soil_ph": 7.88,
        "humidity_pct": 42.68,
        "soil_nitrogen_pct": 0.39,
        "organic_carbon_pct": 0.49,
        "slope_degrees": 8.49,
    },
    "Meru": {
        "altitude_m": 1313.81,
        "annual_rainfall_mm": 1183.04,
        "temp_max_c": 27.94,
        "temp_min_c": 17.08,
        "soil_ph": 5.77,
        "humidity_pct": 64.46,
        "soil_nitrogen_pct": 1.36,
        "organic_carbon_pct": 1.87,
        "slope_degrees": 15.17,
    },
    "Migori": {
        "altitude_m": 1424.17,
        "annual_rainfall_mm": 1232.72,
        "temp_max_c": 27.44,
        "temp_min_c": 16.66,
        "soil_ph": 5.85,
        "humidity_pct": 67.99,
        "soil_nitrogen_pct": 1.0,
        "organic_carbon_pct": 1.64,
        "slope_degrees": 6.62,
    },
    "Mombasa": {
        "altitude_m": 46.45,
        "annual_rainfall_mm": 981.38,
        "temp_max_c": 32.14,
        "temp_min_c": 24.49,
        "soil_ph": 7.42,
        "humidity_pct": 78.5,
        "soil_nitrogen_pct": 0.51,
        "organic_carbon_pct": 1.11,
        "slope_degrees": 2.5,
    },
    "Murang'a": {
        "altitude_m": 1728.6,
        "annual_rainfall_mm": 1431.1,
        "temp_max_c": 25.26,
        "temp_min_c": 14.81,
        "soil_ph": 5.22,
        "humidity_pct": 70.69,
        "soil_nitrogen_pct": 1.66,
        "organic_carbon_pct": 2.73,
        "slope_degrees": 11.62,
    },
    "Nairobi": {
        "altitude_m": 1729.31,
        "annual_rainfall_mm": 963.85,
        "temp_max_c": 25.65,
        "temp_min_c": 13.92,
        "soil_ph": 6.18,
        "humidity_pct": 63.33,
        "soil_nitrogen_pct": 0.91,
        "organic_carbon_pct": 1.55,
        "slope_degrees": 4.19,
    },
    "Nakuru": {
        "altitude_m": 2123.68,
        "annual_rainfall_mm": 938.03,
        "temp_max_c": 24.02,
        "temp_min_c": 12.97,
        "soil_ph": 7.01,
        "humidity_pct": 58.85,
        "soil_nitrogen_pct": 1.01,
        "organic_carbon_pct": 1.41,
        "slope_degrees": 7.92,
    },
    "Nandi": {
        "altitude_m": 1798.03,
        "annual_rainfall_mm": 1619.86,
        "temp_max_c": 23.54,
        "temp_min_c": 13.06,
        "soil_ph": 5.01,
        "humidity_pct": 75.16,
        "soil_nitrogen_pct": 2.03,
        "organic_carbon_pct": 2.83,
        "slope_degrees": 8.52,
    },
    "Narok": {
        "altitude_m": 2220.23,
        "annual_rainfall_mm": 837.59,
        "temp_max_c": 23.53,
        "temp_min_c": 11.74,
        "soil_ph": 7.0,
        "humidity_pct": 54.97,
        "soil_nitrogen_pct": 0.82,
        "organic_carbon_pct": 1.12,
        "slope_degrees": 6.03,
    },
    "Nyamira": {
        "altitude_m": 1697.49,
        "annual_rainfall_mm": 1528.57,
        "temp_max_c": 24.01,
        "temp_min_c": 14.12,
        "soil_ph": 5.13,
        "humidity_pct": 72.7,
        "soil_nitrogen_pct": 1.63,
        "organic_carbon_pct": 2.3,
        "slope_degrees": 11.54,
    },
    "Nyandarua": {
        "altitude_m": 2535.35,
        "annual_rainfall_mm": 1112.71,
        "temp_max_c": 18.98,
        "temp_min_c": 9.01,
        "soil_ph": 5.76,
        "humidity_pct": 68.75,
        "soil_nitrogen_pct": 1.56,
        "organic_carbon_pct": 2.52,
        "slope_degrees": 9.04,
    },
    "Nyeri": {
        "altitude_m": 2065.8,
        "annual_rainfall_mm": 1562.36,
        "temp_max_c": 22.07,
        "temp_min_c": 12.2,
        "soil_ph": 4.96,
        "humidity_pct": 72.92,
        "soil_nitrogen_pct": 2.12,
        "organic_carbon_pct": 3.12,
        "slope_degrees": 13.53,
    },
    "Samburu": {
        "altitude_m": 1226.51,
        "annual_rainfall_mm": 518.69,
        "temp_max_c": 32.25,
        "temp_min_c": 21.43,
        "soil_ph": 7.97,
        "humidity_pct": 38.46,
        "soil_nitrogen_pct": 0.44,
        "organic_carbon_pct": 0.55,
        "slope_degrees": 7.37,
    },
    "Siaya": {
        "altitude_m": 1308.68,
        "annual_rainfall_mm": 1236.43,
        "temp_max_c": 28.16,
        "temp_min_c": 17.44,
        "soil_ph": 6.35,
        "humidity_pct": 67.98,
        "soil_nitrogen_pct": 0.95,
        "organic_carbon_pct": 1.41,
        "slope_degrees": 3.87,
    },
    "Taita Taveta": {
        "altitude_m": 1379.14,
        "annual_rainfall_mm": 886.47,
        "temp_max_c": 28.76,
        "temp_min_c": 18.76,
        "soil_ph": 6.56,
        "humidity_pct": 61.78,
        "soil_nitrogen_pct": 0.92,
        "organic_carbon_pct": 1.21,
        "slope_degrees": 16.29,
    },
    "Tana River": {
        "altitude_m": 173.97,
        "annual_rainfall_mm": 482.45,
        "temp_max_c": 32.77,
        "temp_min_c": 24.05,
        "soil_ph": 8.23,
        "humidity_pct": 62.63,
        "soil_nitrogen_pct": 0.44,
        "organic_carbon_pct": 0.6,
        "slope_degrees": 2.49,
    },
    "Tharaka Nithi": {
        "altitude_m": 1305.88,
        "annual_rainfall_mm": 971.2,
        "temp_max_c": 27.06,
        "temp_min_c": 17.46,
        "soil_ph": 6.2,
        "humidity_pct": 62.45,
        "soil_nitrogen_pct": 0.8,
        "organic_carbon_pct": 1.11,
        "slope_degrees": 15.28,
    },
    "Trans Nzoia": {
        "altitude_m": 2235.68,
        "annual_rainfall_mm": 1130.67,
        "temp_max_c": 20.84,
        "temp_min_c": 10.67,
        "soil_ph": 6.22,
        "humidity_pct": 66.11,
        "soil_nitrogen_pct": 1.21,
        "organic_carbon_pct": 2.14,
        "slope_degrees": 4.33,
    },
    "Turkana": {
        "altitude_m": 1069.11,
        "annual_rainfall_mm": 221.17,
        "temp_max_c": 37.23,
        "temp_min_c": 25.19,
        "soil_ph": 8.2,
        "humidity_pct": 34.73,
        "soil_nitrogen_pct": 0.19,
        "organic_carbon_pct": 0.27,
        "slope_degrees": 4.4,
    },
    "Uasin Gishu": {
        "altitude_m": 2136.96,
        "annual_rainfall_mm": 1083.75,
        "temp_max_c": 21.1,
        "temp_min_c": 10.07,
        "soil_ph": 6.32,
        "humidity_pct": 63.57,
        "soil_nitrogen_pct": 1.21,
        "organic_carbon_pct": 1.97,
        "slope_degrees": 4.77,
    },
    "Vihiga": {
        "altitude_m": 1679.07,
        "annual_rainfall_mm": 1672.55,
        "temp_max_c": 25.75,
        "temp_min_c": 14.72,
        "soil_ph": 5.44,
        "humidity_pct": 74.57,
        "soil_nitrogen_pct": 1.55,
        "organic_carbon_pct": 2.09,
        "slope_degrees": 9.65,
    },
    "Wajir": {
        "altitude_m": 326.52,
        "annual_rainfall_mm": 221.34,
        "temp_max_c": 37.31,
        "temp_min_c": 23.39,
        "soil_ph": 8.32,
        "humidity_pct": 36.25,
        "soil_nitrogen_pct": 0.21,
        "organic_carbon_pct": 0.24,
        "slope_degrees": 1.52,
    },
    "West Pokot": {
        "altitude_m": 1765.1,
        "annual_rainfall_mm": 1067.54,
        "temp_max_c": 25.62,
        "temp_min_c": 13.82,
        "soil_ph": 6.35,
        "humidity_pct": 58.08,
        "soil_nitrogen_pct": 0.9,
        "organic_carbon_pct": 1.3,
        "slope_degrees": 17.52,
    },
}

# 3. Define the UI Header and UX layout
st.set_page_config(page_title="Kenya Crop Classifier", layout="centered")
st.title("🌱 Kenya Multi-Crop Suitability Classifier")
st.markdown(
    "Select your county to automatically populate average environmental and soil parameters, "
    "or manually enter your specific farm's data below."
)

# 4. County Selection and Dynamic Autofill
st.markdown("### Location")
counties = list(COUNTY_MEANS.keys())

# Defaulting to Nairobi for local testing convenience
default_index = counties.index("Nairobi") if "Nairobi" in counties else 0
selected_county = st.selectbox(
    "Select County for Autofill", counties, index=default_index
)

# Fetch the dictionary of average values for the chosen county
defaults = COUNTY_MEANS[selected_county]

st.markdown("### Environmental Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    altitude = st.number_input(
        "Altitude (m)", min_value=0.0, value=float(defaults["altitude_m"])
    )
    rainfall = st.number_input(
        "Annual Rainfall (mm)",
        min_value=0.0,
        value=float(defaults["annual_rainfall_mm"]),
    )
    max_temp = st.number_input(
        "Max Temp (°C)", min_value=0.0, value=float(defaults["temp_max_c"])
    )

with col2:
    min_temp = st.number_input(
        "Min Temp (°C)", min_value=0.0, value=float(defaults["temp_min_c"])
    )
    soil_ph = st.number_input(
        "Soil pH", min_value=0.0, max_value=14.0, value=float(defaults["soil_ph"])
    )
    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(defaults["humidity_pct"]),
    )

with col3:
    nitrogen = st.number_input(
        "Soil Nitrogen (%)", min_value=0.0, value=float(defaults["soil_nitrogen_pct"])
    )
    carbon = st.number_input(
        "Organic Carbon (%)", min_value=0.0, value=float(defaults["organic_carbon_pct"])
    )
    slope = st.number_input(
        "Slope (degrees)", min_value=0.0, value=float(defaults["slope_degrees"])
    )

# 5. Prediction Execution
if st.button("Evaluate Suitability", type="primary"):

    # Structure inputs as a DataFrame with the EXACT feature names used in training
    input_data = pd.DataFrame(
        [
            [
                altitude,
                rainfall,
                max_temp,
                min_temp,
                soil_ph,
                humidity,
                nitrogen,
                carbon,
                slope,
            ]
        ],
        columns=[
            "altitude_m",
            "annual_rainfall_mm",
            "temp_max_c",
            "temp_min_c",
            "soil_ph",
            "humidity_pct",
            "soil_nitrogen_pct",
            "organic_carbon_pct",
            "slope_degrees",
        ],
    )

    # Predict directly using the unscaled DataFrame
    prediction_encoded = model.predict(input_data)
    prediction_label = le.inverse_transform(prediction_encoded)[0]

    # Output Result
    st.divider()
    if prediction_label == "Highly Suitable":
        st.success(f"**Result:** {prediction_label} ✅")
    elif prediction_label == "Moderately Suitable":
        st.warning(f"**Result:** {prediction_label} ⚠️")
    else:
        st.error(f"**Result:** {prediction_label} ❌")
