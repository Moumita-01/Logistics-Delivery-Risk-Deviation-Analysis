# 🚚 Logistics Delivery Risk & Deviation Analysis

An AI-powered web application that predicts delivery delay risks and provides explainable insights for logistics management. This project uses **XGBoost** and **Streamlit** to help supply chain managers mitigate risks before they happen.

## 🌟 Key Features
- **Real-time Risk Prediction:** Predicts if a shipment is at `High`, `Medium`, or `Low` risk of delay.
- **Explainable AI (XAI):** Visualizes the top 5 factors (e.g., Traffic, Weather, Driver Fatigue) influencing each specific prediction.
- **Prescriptive Analytics:** Simulates a "5% Holistic Improvement" strategy, showing a potential **1.12% reduction** in high-risk delays.
- **Interactive Dashboard:** User-friendly sidebar to adjust operational parameters and see instant results.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost
- **Web Framework:** Streamlit
- **Visualizations:** Matplotlib, Seaborn
- **Model Storage:** Pickle (PKL)

## 📊 Data Features (13 total)
The model processes a mix of operational and engineered features:
1. `traffic_congestion_level`
2. `weather_condition_severity`
3. `port_congestion_level`
4. `customs_clearance_time`
5. `route_risk_level`
6. `driver_behavior_score`
7. `fatigue_monitoring_score`
8. `disruption_likelihood_score` (Vehicle Maintenance)
9. `fuel_consumption_rate`
10. `hour` (Warehouse Inventory)
11. `env_risk_score` (Derived: Traffic × Weather)
12. `cost_efficiency` (Derived: Cost / Fuel)
13. `warehouse_pressure` (Derived: Inventory × 2)

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Moumita-01/logistics-risk-analysis.git](https://github.com/Moumita-01/logistics-risk-analysis.git)
   cd logistics-risk-analysis
