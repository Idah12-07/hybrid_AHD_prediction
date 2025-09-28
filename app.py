
import streamlit as st
import numpy as np
import pandas as pd
import joblib
# Load model
try:
    deploy = joblib.load("ahd_model_C_hybrid_fixed.pkl")
    model = deploy['model']
    feature_names = deploy['feature_names']
    model_loaded = True
except FileNotFoundError:
    st.error("Model file 'ahd_model_C_hybrid_fixed.pkl' not found.")
    model_loaded = False

# Page config
st.set_page_config(page_title="AHD Detection", layout="wide", page_icon="🧠")
st.title("🧠 Advanced HIV Disease (AHD) Detection")
st.markdown("""
This tool helps clinicians assess the risk of **Advanced HIV Disease (AHD)**  
based on patient details such as age, weight, CD4 count, viral load, and treatment history.  
""")

# Sidebar: Demo profile selector
st.sidebar.markdown("### 🧑‍⚕️ Demo Patient Profiles")
demo_choice = st.sidebar.selectbox("Choose a demo profile", [
    "None (Manual Entry)",
    "Low Risk – Stable Patient",
    "Moderate Risk – Monitoring Needed",
    "High Risk – Immediate Review"
])

# Default values
age = st.sidebar.number_input("Age at Reporting", min_value=0, max_value=120, value=35)
weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=60.0)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=220, value=165)
cd4 = st.sidebar.number_input("Latest CD4 Count", min_value=0, max_value=2000, value=350)
vl = st.sidebar.number_input("Latest Viral Load (copies/ml)", min_value=0, max_value=10000000, value=1000)
months_rx = st.sidebar.slider("Months of Prescription", 0, 6, 3)
who_stage = st.sidebar.selectbox("Last WHO Stage", [1, 2, 3, 4])
cd4_risk = st.sidebar.selectbox("CD4 Risk Category", ["Severe", "Moderate", "Normal", "Unknown"])
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
st.sidebar.markdown("---")

# Apply demo profile if selected
if demo_choice != "None (Manual Entry)":
    if demo_choice == "Low Risk – Stable Patient":
        age, weight, height, cd4, vl, months_rx, who_stage, cd4_risk, sex = (
            28, 65.0, 168, 850, 50, 3, 1, "Normal", "Female"
        )
    elif demo_choice == "Moderate Risk – Monitoring Needed":
        age, weight, height, cd4, vl, months_rx, who_stage, cd4_risk, sex = (
            35, 60.0, 165, 400, 1200, 2, 3, "Moderate", "Male"
        )
    elif demo_choice == "High Risk – Immediate Review":
        age, weight, height, cd4, vl, months_rx, who_stage, cd4_risk, sex = (
            42, 55.0, 160, 150, 5000, 1, 4, "Severe", "Male"
        )

# Derived fields
bmi = weight / ((height / 100) ** 2) if height > 0 else 0
cd4_missing = 0 if cd4 > 0 else 1
vl_missing = 0 if vl > 0 else 1
vl_suppressed = 1 if vl < 1000 else 0
cd4_risk_Severe = int(cd4_risk == "Severe")
cd4_risk_Moderate = int(cd4_risk == "Moderate")
cd4_risk_Normal = int(cd4_risk == "Normal")
Last_WHO_Stage_2 = int(who_stage == 2)
Last_WHO_Stage_3 = int(who_stage == 3)
Last_WHO_Stage_4 = int(who_stage == 4)
Sex_M = int(sex.lower().startswith("m"))

# Missing flags
Active_in_PMTCT_Missing = 0
Cacx_Screening_Missing = 0
Refill_Date_Missing = 0

# Input vector
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

X_input = pd.DataFrame([input_data_dict])[feature_names].astype(float)

# Threshold slider
st.markdown("### 🎚️ Risk Threshold Adjustment")
threshold = st.slider("Set AHD Risk Threshold", 0.0, 1.0, 0.45, 0.01)
st.caption(f"Current threshold: {threshold:.2f}")

# Prediction
if model_loaded and st.sidebar.button("🔍 Predict AHD Risk"):
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    st.subheader("📌 Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AHD Risk", "Yes" if int(pred) == 1 else "No")
    with col2:
        st.metric("Risk Probability", f"{proba:.2%}")
    st.progress(proba)

    if proba > threshold:
        st.error("⚠️ Above Threshold – Consider clinical review.")
    else:
        st.success("🟢 Below Threshold – Continue routine care.")

    # Patient summary replaces raw feature table
    with st.expander("🧑‍⚕️ Patient Summary"):
        st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:15px; border-radius:10px;'>
        <b>Age:</b> {age} years &nbsp;&nbsp;
        <b>Weight:</b> {weight} kg &nbsp;&nbsp;
        <b>Height:</b> {height} cm (BMI: {bmi:.1f})  
        <br><b>Latest CD4 Count:</b> {cd4}  
        <b>Viral Load:</b> {vl} copies/ml ({'Suppressed' if vl_suppressed else 'Unsuppressed'})  
        <br><b>WHO Stage:</b> {who_stage}  
        <b>Months of Prescription:</b> {months_rx}  
        <b>CD4 Risk Category:</b> {cd4_risk}  
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>© 2025 | Built with ❤️ by <b>Idah Anyango</b></div>", unsafe_allow_html=True)

