import streamlit as st
import pandas as pd
import joblib
import gzip
import pickle
import numpy as np

# ---- Set Page Config ----
st.set_page_config(page_title="Medicare - Best Provider Finder", layout="wide")

# ---- Custom Background Style ----
st.markdown(
    """
    <style>
        body {
            background-color: #e3f2fd; /* Light blue */
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 24px;
            border: none;
        }
        .stButton>button:hover {color: black;}
        .stButton>button:active {
            background-color: white !important;
            color: #007bff !important;
            border: 2px solid #007bff !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Load Models & Encoders ----
@st.cache_resource
def load_models():
    with gzip.open("inpatient_model.pkl.gz", "rb") as f:
        inpatient_model = joblib.load(f)
    with gzip.open("outpatient_model.pkl.gz", "rb") as f:
        outpatient_model = joblib.load(f)
    return inpatient_model, outpatient_model

@st.cache_resource
def load_encoders():
    with gzip.open("le_drg.pkl.gz", "rb") as f:
        inpatient_le_drg = joblib.load(f)
    with gzip.open("le_state.pkl.gz", "rb") as f:
        inpatient_le_state = joblib.load(f)
    with gzip.open("le_name.pkl.gz", "rb") as f:
        inpatient_name = joblib.load(f)
    with gzip.open("le_drg_outpatient.pkl.gz", "rb") as f:
        outpatient_le_apc = joblib.load(f)
    with gzip.open("le_state_outpatient.pkl.gz", "rb") as f:
        outpatient_le_state = joblib.load(f)
    with gzip.open("le_name.pkl.gz", "rb") as f:
        outpatient_name = joblib.load(f)
    return inpatient_le_drg, inpatient_le_state, inpatient_name, outpatient_le_apc, outpatient_le_state, outpatient_name

@st.cache_resource
def load_hospital_data():
    with open('hospital_data_in.pkl', 'rb') as f:
        hospital_data_in = pickle.load(f)
    with open('hospital_data_out.pkl', 'rb') as f:
        hospital_data_out = pickle.load(f)
    return hospital_data_in, hospital_data_out

# ---- Load Everything ----
inpatient_model, outpatient_model = load_models()
inpatient_le_drg, inpatient_le_state, inpatient_name, outpatient_le_apc, outpatient_le_state, outpatient_name = load_encoders()
hospital_data_in, hospital_data_out = load_hospital_data()

# ---- User Credentials ----
Usercredentials = {"Sakthi": "Password", "Janani": "passwordjan"}

# ---- Login Page ----
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in Usercredentials and Usercredentials[username] == password:
            st.session_state["logged_in"] = True
            st.success(f"Welcome, {username}! 🎉")
        else:
            st.error("Invalid Username or Password")

if "logged_in" not in st.session_state:
    login()
    st.stop()

# ---- Main Interface ----
st.title("🏥 Medicare - Best Provider Finder")
st.write("Find the best hospital for your procedure based on cost and ranking.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("🔍 Search Criteria")
    service_type = st.selectbox("Select Service Type", ["Inpatient", "Outpatient"])
    Procedure = st.selectbox("Select Procedure", sorted(inpatient_le_drg.classes_ if service_type == "Inpatient" else outpatient_le_apc.classes_))
    State = st.selectbox("Select State", sorted(inpatient_le_state.classes_ if service_type == "Inpatient" else outpatient_le_state.classes_))

    if st.button("Find Best Hospitals"):
        if service_type == "Inpatient":
            encoded_procedure = inpatient_le_drg.transform([Procedure])[0]
            encoded_state = inpatient_le_state.transform([State])[0]
            model = inpatient_model
            hospital_data = hospital_data_in
        else:
            encoded_procedure = outpatient_le_apc.transform([Procedure])[0]
            encoded_state = outpatient_le_state.transform([State])[0]
            model = outpatient_model
            hospital_data = hospital_data_out

        default_values = [0, 0, 0]

        # Make prediction
        prediction_input = np.array([[encoded_procedure, encoded_state] + default_values[:3]])
        predicted_cost = model.predict(prediction_input)[0]

        # Filter hospital data
        filter_hospital_data = hospital_data[(hospital_data['Procedure'] == Procedure) & (hospital_data['Provider_State'] == State)]
        filter_hospital_data = filter_hospital_data.sort_values(by="Average_Total_Payments")

        # Rank Hospitals
        filter_hospital_data["Rank"] = range(1, len(filter_hospital_data) + 1)

        # Display results in the second column
        with col2:
            st.header("🏆 Best Hospitals")
            st.dataframe(filter_hospital_data[["Procedure", "Rank", "Provider_Name", "Average_Total_Payments"]])

