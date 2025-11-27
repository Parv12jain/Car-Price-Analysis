# used_car_app.py
import streamlit as st
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main {
            background-color: #1e1e1e;
        }
        .stButton>button {
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load Model
# -------------------------
MODEL_PATH = "model.pkl"

st.sidebar.title("⚙️ Model Controls")

if not os.path.exists(MODEL_PATH):
    st.sidebar.error("❌ model.pkl not found! Place it next to app.py.")
    model = None
else:
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Model failed to load: {e}")
        model = None

# Expected input columns
cat_cols = ["Manufacturer", "Model", "Fuel type"]
num_cols = ["Engine size", "Year of manufacture", "Mileage"]
expected = cat_cols + num_cols

st.title("🚗💰 Used Car Price Prediction App")
st.write("### Predict prices using your trained Machine Learning model (Random Forest Pipeline)")

st.markdown("---")

# -------------------------
# SINGLE PREDICTION
# -------------------------
st.header("🔧 Single Car Price Prediction")

col1, col2 = st.columns([2, 2])

with col1:
    manu = st.selectbox("Manufacturer", ["BMW", "Ford", "Toyota", "Honda", "Hyundai", "Mercedes", "Audi", "Other"])
    model_name = st.text_input("Model Name", value="Civic")
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"])

with col2:
    engine = st.number_input("Engine Size (Litres)", 0.5, 10.0, 1.5, step=0.1)
    year = st.number_input("Year of Manufacture", 1990, 2025, 2015)
    mileage = st.number_input("Mileage (km)", 0, 500000, 50000)

single_df = pd.DataFrame([{
    "Manufacturer": manu,
    "Model": model_name,
    "Fuel type": fuel,
    "Engine size": engine,
    "Year of manufacture": year,
    "Mileage": mileage,
}])

if st.button("🚀 Predict Single Car Price", use_container_width=True):
    if model is None:
        st.error("❌ Model not loaded!")
    else:
        with st.spinner("Predicting..."):
            try:
                pred = model.predict(single_df)
                st.success(f"💰 Predicted Car Price: **₹ {pred[0]:,.2f}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("If this error is about missing columns, your model.pkl must be the pipeline (preprocessor + model).")

st.markdown("---")

# -------------------------
# BATCH PREDICTION
# -------------------------
st.header("📁 Batch Prediction (Upload CSV)")
st.write("Upload a CSV containing these columns:")

st.code(expected, language="python")

file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if file:
    try:
        df = pd.read_csv(file)
        st.success("CSV loaded successfully!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df = None
    
    if df is not None:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
        else:
            if model is None:
                st.error("❌ Model not loaded!")
            else:
                if st.button("📊 Predict Prices for CSV", use_container_width=True):
                    with st.spinner("Running batch prediction..."):
                        try:
                            preds = model.predict(df[expected])
                            out = df.copy()
                            out["Predicted Price"] = preds

                            st.success("Batch prediction completed!")
                            st.dataframe(out.head())

                            csv = out.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="⬇️ Download Predictions CSV",
                                data=csv,
                                file_name="car_price_predictions.csv",
                                mime="text/csv",
                            )
                        except Exception as e:
                            st.error(f"Batch prediction failed: {e}")
                            st.info("Check if your model.pkl includes the preprocessing pipeline.")

st.markdown("---")

st.markdown("""
### ✔ Tips
- Ensure `model.pkl` is saved as a **Pipeline** (preprocessor + model).
- CSV must NOT include the target column (Price).
- Streamlit Cloud deployment works with this UI.
""")

st.markdown("Made with ❤️ using Streamlit + Machine Learning")
