# streamlit_app.py
# Modern UI for Mobile Price Range Prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="📱 Mobile Price Prediction App",
    layout="wide",
    page_icon="📱"
)

# ==========================================================
# CUSTOM STYLING
# ==========================================================
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(145deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        }

        h1, h2, h3, h4, h5, p, div {
            color: #f1f1f1 !important;
            font-family: 'Segoe UI', sans-serif;
        }

        .card {
            background: rgba(255,255,255,0.08);
            padding: 20px;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(12px);
        }

        .prediction-box {
            background: rgba(0,255,150,0.2);
            border-left: 6px solid #00ff9d;
            padding: 15px;
            border-radius: 12px;
            font-size: 1.3rem;
        }

        .stButton>button {
            background-color: #00BFFF;
            color: white;
            border-radius: 10px;
            padding: 0.7rem 1.3rem;
            border: none;
        }

        .stButton>button:hover {
            background-color: #009acd;
            transform: scale(1.03);
        }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# HEADER
# ==========================================================
st.markdown(
    """
    <h1 style='text-align:center; font-size: 3rem; margin-bottom: 0;'>📱 Mobile Price Prediction</h1>
    <p style='text-align:center; font-size: 1.2rem;'>Predict mobile price range using ML model</p>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# LOAD MODEL
# ==========================================================
MODEL_PATH = "model_trained.pkl"



@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ==========================================================
# INPUT FEATURE SETUP
# ==========================================================
FEATURES = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc','four_g',
    'int_memory','m_dep','mobile_wt','n_cores','pc','px_height',
    'px_width','ram','sc_h','sc_w','talk_time','three_g','touch_screen','wifi'
]

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📥 Enter Mobile Specifications")

input_data = {}
cols = st.columns(2)

for i, feature in enumerate(FEATURES):
    with cols[i % 2]:
        if feature in ['blue','dual_sim','four_g','three_g','touch_screen','wifi']:
            input_data[feature] = st.selectbox(f"{feature}", [0,1])
        else:
            input_data[feature] = st.number_input(
                f"{feature}",
                min_value=0.0,
                value=1.0,
                step=0.1
            )

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# PREDICT BUTTON
# ==========================================================
if st.button("🚀 Predict Price Range"):
    df_input = pd.DataFrame([input_data])
    prediction = model.predict(df_input)[0]

    label_map = {0:'Low', 1:'Medium', 2:'High', 3:'Very High'}

    st.balloons()
    st.markdown(
        f"""
        <div class='prediction-box'>
        <strong>Predicted Price Range:</strong> {label_map.get(prediction)} ({prediction})
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================================
# FOOTER
# ==========================================================
st.markdown(
    """
    <hr>
    <p style='text-align:center;'>Made with ❤️ using Streamlit</p>
    """,
    unsafe_allow_html=True
)
