# app.py — Streamlit UI for Tourism Package Purchase Prediction
import streamlit as st
import pandas as pd
import pickle
from huggingface_hub import hf_hub_download
import joblib

# Download and load model and encoders from Hugging Face
model_path = hf_hub_download(
    repo_id="shawsushant/tourism-package-prediction-model",
    filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

encoders_path = hf_hub_download(
    repo_id="shawsushant/tourism-package-prediction",
    repo_type="dataset",
    filename="encoders.pkl"
)
with open(encoders_path, "rb") as f:
    encoders = pickle.load(f)

# App title and description
st.title("Tourism Package Prediction")
st.write("""
Welcome to Tourism Package Prediction : This application predicts whether a customer will purchase the Wellness Tourism Package
based on their profile and interaction details. Please enter the details below.
""")

# User inputs
age                = st.number_input("Age", min_value=18, max_value=80, value=35)
monthly_income     = st.number_input("Monthly Income (₹)", min_value=5000, max_value=100000, value=25000, step=1000)
duration_pitch     = st.number_input("Duration of Pitch (min)", min_value=5, max_value=60, value=20)
num_followups      = st.number_input("Number of Follow-ups", min_value=1, max_value=6, value=3)
num_persons        = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2)
num_children       = st.number_input("Number of Children Visiting (<5 yrs)", min_value=0, max_value=3, value=0)
num_trips          = st.number_input("Average Trips per Year", min_value=1, max_value=20, value=3)
pitch_satisfaction = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
preferred_star     = st.number_input("Preferred Property Stars (3-5)", min_value=3, max_value=5, value=3)
city_tier          = st.selectbox("City Tier", [1, 2, 3])
type_of_contact    = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
occupation         = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender             = st.selectbox("Gender", ["Male", "Female"])
marital_status     = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
designation        = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
product_pitched    = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
passport           = st.selectbox("Has Passport", ["No", "Yes"])
own_car            = st.selectbox("Owns a Car", ["No", "Yes"])

#  EncodING categorical inputs using the saved LabelEncoders 
def encode(col, value):
    le = encoders[col]
    if value in le.classes_:
        return int(le.transform([value])[0])
    return 0

# Assemble input DataFrame with correct dtypes
input_data = pd.DataFrame([{
    'Age':                      int(age),
    'TypeofContact':            encode('TypeofContact', type_of_contact),
    'CityTier':                 int(city_tier),
    'DurationOfPitch':          float(duration_pitch),
    'Occupation':               encode('Occupation', occupation),
    'Gender':                   encode('Gender', gender),
    'NumberOfPersonVisiting':   int(num_persons),
    'NumberOfFollowups':        float(num_followups),
    'ProductPitched':           encode('ProductPitched', product_pitched),
    'PreferredPropertyStar':    float(preferred_star),
    'MaritalStatus':            encode('MaritalStatus', marital_status),
    'NumberOfTrips':            float(num_trips),
    'Passport':                 1 if passport == 'Yes' else 0,
    'PitchSatisfactionScore':   int(pitch_satisfaction),
    'OwnCar':                   1 if own_car == 'Yes' else 0,
    'NumberOfChildrenVisiting': float(num_children),
    'Designation':              encode('Designation', designation),
    'MonthlyIncome':            float(monthly_income),
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Customer will purchase the package" if prediction == 1 else "Customer will not purchase the package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
