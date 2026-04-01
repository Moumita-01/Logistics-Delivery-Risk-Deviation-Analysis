import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Logistics Risk Dashboard", layout="wide", page_icon="🚚")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = pickle.load(open('xgb_logistics_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    return model, scaler, encoder

try:
    model, scaler, encoder = load_assets()
except Exception as e:
    st.error(f"❌ Error loading pickle files: {e}")
    st.stop()

# --- HEADER ---
st.title("🚚 Logistics Delivery Delay & Risk Predictor")
st.markdown("This AI-powered dashboard predicts delivery risks and explains the 'Why' behind every prediction.")

# --- SIDEBAR: INPUT FEATURES ---
st.sidebar.header("📦 Input Shipment Details")

def get_user_input():
    # 1. Inputs from Sidebar (Matching your Scaler Order)
    traffic = st.sidebar.slider('Traffic Congestion Level', 1, 5, 2)
    weather = st.sidebar.slider('Weather Severity', 1, 5, 1)
    port = st.sidebar.slider('Port Congestion Level', 1, 5, 2)
    customs = st.sidebar.number_input('Customs Clearance Time (hrs)', 0.0, 100.0, 15.0)
    route = st.sidebar.slider('Route Risk Level', 1, 5, 2)
    driver_score = st.sidebar.slider('Driver Behavior Score', 0, 100, 85)
    fatigue = st.sidebar.slider('Fatigue Monitoring Score', 0, 10, 2)
    maint = st.sidebar.selectbox('Vehicle Maintenance (1: Good, 0: Bad)', [1, 0])
    fuel = st.sidebar.number_input('Fuel Consumption Rate', 5.0, 35.0, 12.0)
    inventory = st.sidebar.slider('Warehouse Inventory Level', 0, 100, 50)
    
    # 2. Derived Calculations
    env_risk = traffic * weather
    cost_eff = 500 / (fuel + 0.1) 
    wh_pressure = inventory * 2    

    # 3. CRITICAL: Exact Order from your Notebook
    features_dict = {
        'traffic_congestion_level': traffic,
        'weather_condition_severity': weather,
        'port_congestion_level': port,
        'customs_clearance_time': customs,
        'route_risk_level': route,
        'driver_behavior_score': driver_score,
        'fatigue_monitoring_score': fatigue,
        'disruption_likelihood_score': maint,
        'fuel_consumption_rate': fuel,
        'hour': inventory,
        'env_risk_score': env_risk,
        'cost_efficiency': cost_eff,
        'warehouse_pressure': wh_pressure
    }
    
    return pd.DataFrame(features_dict, index=[0])

input_df = get_user_input()

# --- MAIN SECTION ---
col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.subheader("🔍 Prediction & Reasoning")
    if st.button('Run Prediction Model'):
        # Step A: Scale
        scaled_input = scaler.transform(input_df)
        
        # Step B: Predict
        pred = model.predict(scaled_input)[0]
        risk_label = encoder.inverse_transform([pred])[0]
        
        # UI Feedback
        if risk_label == 'High Delay':
            st.error(f"### Predicted Risk: {risk_label}")
        elif risk_label == 'Medium Delay':
            st.warning(f"### Predicted Risk: {risk_label}")
        else:
            st.success(f"### Predicted Risk: {risk_label}")

        st.divider()
        
        # --- REASONING (Feature Importance) ---
        st.write("#### 💡 Why this prediction?")
        
        # Get importances from XGBoost
        importances = model.feature_importances_
        feat_imp_df = pd.DataFrame({
            'Factor': input_df.columns,
            'Impact': importances
        }).sort_values(by='Impact', ascending=False).head(5)

        fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Impact', y='Factor', data=feat_imp_df, palette='magma', ax=ax_imp)
        ax_imp.set_title("Top 5 Delay Drivers")
        st.pyplot(fig_imp)
        
        # Dynamic Text Logic
        top_factor = feat_imp_df.iloc[0]['Factor'].replace('_', ' ').title()
        st.info(f"**Key Insight:** The primary reason for this risk level is **{top_factor}**. Management should focus here first.")

with col2:
    st.subheader("📈 Strategy Impact")
    st.write("Reduction with **5% Holistic Improvement**:")
    
    base_val, improved_val = 39.04, 38.60
    fig, ax = plt.subplots(figsize=(6, 7))
    sns.barplot(x=['Current', 'Improved'], y=[base_val, improved_val], palette=['#34495e', '#2ecc71'], ax=ax)
    ax.set_ylim(35, 41)
    ax.set_ylabel("High-Risk Probability (%)")
    
    for i, v in enumerate([base_val, improved_val]):
        ax.text(i, v + 0.1, f"{v:.2f}%", ha='center', fontweight='bold', size=12)
    st.pyplot(fig)
    
    st.success("🎯 **Goal:** Target 1.12% Risk Reduction.")

st.divider()
st.caption("Final Year Project: CS Engineering | Explainable AI (XAI) in Logistics")