import streamlit as st
import numpy as np
import joblib


# 1. Cache the model loading to prevent CPU throttling
@st.cache_resource
def load_artifacts():
    model = joblib.load("kenya_tea_model.pkl")
    scaler = joblib.load("crop_scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, scaler, le


# Load artifacts into memory (runs only once)
model, scaler, le = load_artifacts()

# 2. Define the UI Header and UX layout
st.set_page_config(page_title="Kenya Crop Classifier", layout="centered")
st.title("🌱 Kenya Multi-Crop Suitability Classifier")
st.markdown(
    "Enter the environmental and soil parameters below to determine crop suitability."
)

# 3. Create input columns for a cleaner UI
col1, col2, col3 = st.columns(3)

with col1:
    altitude = st.number_input("Altitude (m)", min_value=0.0, value=1500.0)
    rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1000.0)
    max_temp = st.number_input("Max Temp (°C)", min_value=0.0, value=25.0)

with col2:
    min_temp = st.number_input("Min Temp (°C)", min_value=0.0, value=15.0)
    soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.0)
    humidity = st.number_input(
        "Humidity (%)", min_value=0.0, max_value=100.0, value=65.0
    )

with col3:
    nitrogen = st.number_input("Soil Nitrogen (%)", min_value=0.0, value=1.2)
    carbon = st.number_input("Organic Carbon (%)", min_value=0.0, value=1.8)
    slope = st.number_input("Slope (degrees)", min_value=0.0, value=8.0)

# 4. Prediction Execution
if st.button("Evaluate Suitability", type="primary"):
    # Structure inputs exactly as they were during training
    input_data = np.array(
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
        ]
    )

    # Scale inputs
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction_encoded = model.predict(scaled_data)
    prediction_label = le.inverse_transform(prediction_encoded)[0]

    # Output Result
    st.divider()
    if prediction_label == "Highly Suitable":
        st.success(f"**Result:** {prediction_label} ✅")
    elif prediction_label == "Moderately Suitable":
        st.warning(f"**Result:** {prediction_label} ⚠️")
    else:
        st.error(f"**Result:** {prediction_label} ❌")
