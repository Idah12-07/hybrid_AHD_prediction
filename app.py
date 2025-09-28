
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load deployable object (contains 'model' and ordered 'feature_names')
try:
    deploy = joblib.load("ahd_model_C_hybrid_fixed.pkl")
    model = deploy['model']
    feature_names = deploy['feature_names']
    model_loaded = True
except FileNotFoundError:
    st.error("Model file 'ahd_model_C_hybrid_fixed.pkl' not found. Please ensure it's uploaded.")
    model_loaded = False

st.set_page_config(page_title="AHD Detection", layout="wide", page_icon="🧠")
st.title("🧠 Advanced HIV Disease (AHD) Detection")
st.markdown("""
This tool helps clinicians assess the risk of **Advanced HIV Disease (AHD)**  
based on patient details such as age, weight, CD4 count, viral load, and treatment history.  
""")

st.sidebar.header("📝 Patient Information")

if model_loaded:
    # Input fields (same as training features)
    age = st.sidebar.number_input("Age at Reporting", min_value=0, max_value=120, value=35)
    weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=60.0)
    height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=220, value=165)
    cd4 = st.sidebar.number_input("Latest CD4 Count", min_value=0, max_value=2000, value=350)
    vl = st.sidebar.number_input("Latest Viral Load (copies/ml)", min_value=0, max_value=10000000, value=1000)
    months_rx = st.sidebar.slider("Months of Prescription", 0, 6, 3)
    who_stage = st.sidebar.selectbox("Last WHO Stage", [1, 2, 3, 4])
    cd4_risk = st.sidebar.selectbox("CD4 Risk Category", ["Severe", "Moderate", "Normal", "Unknown"]) # Added Unknown
    sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
    st.sidebar.markdown("---")

    # Derived fields (exact same transformations as training)
    bmi = weight / ((height / 100) ** 2) if (height and height > 0) else 0
    cd4_missing = 0 if cd4 > 0 else 1
    vl_missing = 0 if vl > 0 else 1
    vl_suppressed = 1 if vl < 1000 else 0

    # Handle CD4 Risk Category one-hot encoding, including 'Unknown'
    cd4_risk_Severe = 1 if cd4_risk == "Severe" else 0
    cd4_risk_Moderate = 1 if cd4_risk == "Moderate" else 0
    cd4_risk_Normal = 1 if cd4_risk == "Normal" else 0
    # Assuming 'Unknown' is handled by the missing flag or not explicitly one-hot encoded if only Severe/Moderate/Normal are in features

    Last_WHO_Stage_2 = 1 if who_stage == 2 else 0
    Last_WHO_Stage_3 = 1 if who_stage == 3 else 0
    Last_WHO_Stage_4 = 1 if who_stage == 4 else 0
    Sex_M = 1 if sex.lower().startswith("m") else 0

    # Default missingness flags (same names that training script used)
    Active_in_PMTCT_Missing = 0
    Cacx_Screening_Missing = 0
    Refill_Date_Missing = 0

    # Build feature-vector as a dictionary to map directly to feature names
    input_data_dict = {
        'Age at reporting': age,
        'Weight': weight,
        'Height': height,
        'BMI': bmi,
        'Latest CD4 Result': cd4,
        'CD4_Missing': cd4_missing,
        'Last VL Result': vl,
        'VL_Suppressed': vl_suppressed,
        'VL_Missing': vl_missing,
        'Months of Prescription': months_rx,
        'cd4_risk_Moderate': cd4_risk_Moderate,
        'cd4_risk_Normal': cd4_risk_Normal,
        'cd4_risk_Severe': cd4_risk_Severe,
        'Last_WHO_Stage_2': Last_WHO_Stage_2,
        'Last_WHO_Stage_3': Last_WHO_Stage_3,
        'Last_WHO_Stage_4': Last_WHO_Stage_4,
        'Active_in_PMTCT_Missing': Active_in_PMTCT_Missing,
        'Cacx_Screening_Missing': Cacx_Screening_Missing,
        'Refill_Date_Missing': Refill_Date_Missing,
        'Sex_M': Sex_M
    }

    # Create DataFrame, ensuring correct order and types
    # Use the feature_names from the loaded model object to ensure correct column order
    X_input = pd.DataFrame([input_data_dict])[feature_names].astype(float)


    # Prediction
    if st.sidebar.button("🔍 Predict AHD Risk"):
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][1]  # prob of class 1 (AHD)

        # Display results
        st.subheader("📌 Prediction Result")
        st.metric("AHD Risk", "Yes" if int(pred) == 1 else "No")
        st.metric("Risk Probability", f"{proba:.2%}")

        # Risk interpretation
        st.progress(proba)
        if proba > 0.75:
            st.error("⚠️ High Risk – Consider immediate clinical review.")
        elif proba > 0.45:
            st.warning("🟠 Moderate Risk – Monitor closely.")
        else:
            st.success("🟢 Low Risk – Continue routine care.")

        # (optional) show the feature row so clinicians see what was fed
        with st.expander("Input features (used for prediction)"):
            st.write(X_input.T)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>© 2025 | Built with ❤️ by <b>Idah Anyango</b></div>", unsafe_allow_html=True)

