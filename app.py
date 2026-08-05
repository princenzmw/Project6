import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier


# 1. Train the model dynamically on boot and cache it in memory
@st.cache_resource
def build_and_train_model():
    # Load dataset (Ensure this CSV is in your GitHub repo)
    df = pd.read_csv("kenya_multicrop_suitability.csv")

    categorical_features = ["county"]
    numeric_features = [
        "altitude_m",
        "annual_rainfall_mm",
        "temp_max_c",
        "temp_min_c",
        "soil_ph",
        "humidity_pct",
        "soil_nitrogen_pct",
        "organic_carbon_pct",
        "slope_degrees",
    ]

    X = df[categorical_features + numeric_features]
    y = df["suitable_tea"]

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Build preprocessing and model pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=7,
                    min_samples_split=5,
                    min_samples_leaf=3,
                    random_state=42,
                ),
            ),
        ]
    )

    # Train the pipeline in memory
    pipeline.fit(X, y_encoded)
    return pipeline, le


# Load artifacts into memory (trains in milliseconds, runs only once)
model, le = build_and_train_model()

# 2. Define the UI Header and UX layout
st.set_page_config(page_title="Kenya Crop Classifier", layout="centered")
st.title("🌱 Kenya Multi-Crop Suitability Classifier")
st.markdown(
    "Enter the location and environmental parameters below to determine crop suitability."
)

# 3. Create input columns
st.markdown("### Location")
county = st.selectbox(
    "County",
    [
        "Baringo",
        "Bomet",
        "Bungoma",
        "Elgeyo Marakwet",
        "Embu",
        "Garissa",
        "Homa Bay",
        "Isiolo",
        "Kajiado",
        "Kakamega",
        "Kericho",
        "Kiambu",
        "Kilifi",
        "Kirinyaga",
        "Kisii",
        "Kisumu",
        "Kitui",
        "Kwale",
        "Laikipia",
        "Lamu",
        "Machakos",
        "Makueni",
        "Mandera",
        "Marsabit",
        "Meru",
        "Migori",
        "Mombasa",
        "Murang'a",
        "Nairobi",
        "Nakuru",
        "Nandi",
        "Narok",
        "Nyamira",
        "Nyandarua",
        "Nyeri",
        "Samburu",
        "Siaya",
        "Taita Taveta",
        "Tana River",
        "Tharaka Nithi",
        "Trans Nzoia",
        "Turkana",
        "Uasin Gishu",
        "Vihiga",
        "Wajir",
        "West Pokot",
    ],
)

st.markdown("### Environmental Parameters")
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

    # Structure inputs as a DataFrame matching the training data structure
    input_data = pd.DataFrame(
        [
            [
                county,
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
            "county",
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

    # Predict directly using the pipeline (automatically handles One-Hot Encoding & Scaling)
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
