# app.py — Streamlit UI for Tourism Package Purchase Prediction
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="centered"
)

# ── Load model from Hugging Face Model Hub ─────────────────────────────
@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="shawsushant/tourism-package-prediction-model",
        filename="best_tourism_model_v1.joblib"
    )
    return joblib.load(path)

model = load_model()

# ── UI ─────────────────────────────────────────────────────────────────
st.title("✈️ Wellness Tourism Package Predictor")
st.markdown(
    "Predict whether a customer is likely to purchase the **Wellness Tourism Package**."
)
st.divider()

# ── Sidebar: collect user inputs ───────────────────────────────────────
st.sidebar.header("Customer Details")

age                    = st.sidebar.slider("Age", 18, 80, 35)
type_of_contact        = st.sidebar.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier              = st.sidebar.selectbox("City Tier", [1, 2, 3])
occupation             = st.sidebar.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender                 = st.sidebar.selectbox("Gender", ["Male", "Female"])
marital_status         = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
designation            = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
product_pitched        = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
monthly_income         = st.sidebar.number_input("Monthly Income (₹)", 5000, 100000, 25000, 1000)
num_persons            = st.sidebar.slider("Persons Visiting", 1, 5, 2)
num_children           = st.sidebar.slider("Children (<5 yrs) Visiting", 0, 3, 0)
num_trips              = st.sidebar.slider("Avg Trips per Year", 1, 20, 3)
num_followups          = st.sidebar.slider("Number of Follow-ups", 1, 6, 3)
pitch_satisfaction     = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
duration_pitch         = st.sidebar.slider("Duration of Pitch (min)", 5, 60, 20)
preferred_star         = st.sidebar.slider("Preferred Property Stars", 3, 5, 3)
passport               = st.sidebar.checkbox("Has Passport", value=False)
own_car                = st.sidebar.checkbox("Owns a Car", value=False)

# ── Label encoding maps (must match prep.py encoding) ──────────────────
CONTACT_MAP  = {"Company Invited": 0, "Self Inquiry": 1}
OCC_MAP      = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
GENDER_MAP   = {"Female": 0, "Male": 1}
MARITAL_MAP  = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
DESIG_MAP    = {"AVP": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}
PRODUCT_MAP  = {"Basic": 0, "Deluxe": 1, "King": 2, "Standard": 3, "Super Deluxe": 4}

input_dict = {
    "Age":                      age,
    "TypeofContact":            CONTACT_MAP.get(type_of_contact, 0),
    "CityTier":                 city_tier,
    "DurationOfPitch":          duration_pitch,
    "Occupation":               OCC_MAP.get(occupation, 0),
    "Gender":                   GENDER_MAP.get(gender, 1),
    "NumberOfPersonVisiting":   num_persons,
    "NumberOfFollowups":        num_followups,
    "ProductPitched":           PRODUCT_MAP.get(product_pitched, 0),
    "PreferredPropertyStar":    preferred_star,
    "MaritalStatus":            MARITAL_MAP.get(marital_status, 1),
    "NumberOfTrips":            num_trips,
    "Passport":                 int(passport),
    "PitchSatisfactionScore":   pitch_satisfaction,
    "OwnCar":                   int(own_car),
    "NumberOfChildrenVisiting": num_children,
    "Designation":              DESIG_MAP.get(designation, 1),
    "MonthlyIncome":            monthly_income,
}

# ── Predict ────────────────────────────────────────────────────────────
if st.button("🔮 Predict", use_container_width=True):
    input_df = pd.DataFrame([input_dict])
    pred     = model.predict(input_df)[0]
    proba    = model.predict_proba(input_df)[0][1]

    st.divider()
    if pred == 1:
        st.success(f"✅ **Likely to Purchase** — Probability: {proba:.1%}")
        st.balloons()
    else:
        st.warning(f"❌ **Unlikely to Purchase** — Probability: {proba:.1%}")
    st.caption(f"Model confidence: {max(proba, 1 - proba):.1%}")
