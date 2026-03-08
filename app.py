import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium

from src.data_processing import load_and_process_data
from src.model_training import create_train_test, train_dtr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

st.set_page_config(page_title="Brooklyn Crash Dashboard", layout="wide")
st.title("Brooklyn Daily Crash Injuries Dashboard")

# Load & process data
daily_df = load_and_process_data("data/brooklyn_crashes.csv")

# Train/test split & model
X_train, X_test, y_train, y_test = create_train_test(daily_df)
dtr = train_dtr(X_train, y_train)
y_pred = dtr.predict(X_test)

# ----------------------
# Sidebar filters
# ----------------------
st.sidebar.header("Filters")
date_filter = st.sidebar.date_input("Select Date Range",
                                   [daily_df['CRASH DATE'].min(), daily_df['CRASH DATE'].max()])
filtered_df = daily_df[(daily_df['CRASH DATE'] >= pd.to_datetime(date_filter[0])) &
                       (daily_df['CRASH DATE'] <= pd.to_datetime(date_filter[1]))]

# ----------------------
# Actual vs Predicted Plot
# ----------------------
st.subheader("Actual vs Predicted Daily Injuries")
pred_df = pd.DataFrame({
    'Date': daily_df['CRASH DATE'].iloc[len(X_train):],
    'Actual': y_test.values,
    'Predicted': y_pred
})
fig = px.line(pred_df, x='Date', y=['Actual','Predicted'], labels={'value':'Daily Injuries','variable':'Legend'})
st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Smoothed 7-day rolling
# ----------------------
pred_df['Actual_Smooth'] = pred_df['Actual'].rolling(7).mean()
pred_df['Predicted_Smooth'] = pred_df['Predicted'].rolling(7).mean()
st.subheader("Smoothed Actual vs Predicted (7-day rolling average)")
fig2 = px.line(pred_df, x='Date', y=['Actual_Smooth','Predicted_Smooth'], labels={'value':'Daily Injuries','variable':'Legend'})
st.plotly_chart(fig2, use_container_width=True)

# ----------------------
# Interactive Map with 4 Hospitals
# ----------------------
st.subheader("Brooklyn Map with Nearby Hospitals")
m = folium.Map(location=[40.6782, -73.9442], zoom_start=12)

hospitals = [
    {"name": "NYU Langone Hospital - Brooklyn", "lat": 40.6500, "lon": -73.9440},
    {"name": "Kings County Hospital Center", "lat": 40.6546, "lon": -73.9447},
    {"name": "Maimonides Medical Center", "lat": 40.6356, "lon": -73.9949},
    {"name": "Brooklyn Hospital Center", "lat": 40.6912, "lon": -73.9735}
]

for hosp in hospitals:
    folium.Marker(
        [hosp["lat"], hosp["lon"]],
        popup=hosp["name"],
        icon=folium.Icon(color='red', icon='plus')
    ).add_to(m)

st_folium(m, width=700, height=500)

# ----------------------
# Model Evaluation Metrics
# ----------------------
st.subheader("Model Evaluation Metrics")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred)/y_test))*100

st.write(f"RMSE: {rmse:.2f}")
st.write(f"MAE: {mae:.2f}")
st.write(f"MAPE: {mape:.2f}%")
st.write(f"R²: {r2:.2f}")

# ----------------------
# Explore Contributing Factors
# ----------------------
st.subheader("Daily Crashes by Contributing Factor")
factor_cols = [col for col in daily_df.columns if col not in ['CRASH DATE','DAILY_INJURIES','DAILY_TEMP','FEDERAL_HOLIDAY','day_of_week','month','is_weekend','dow_sin','dow_cos','month_sin','month_cos','lag_1','lag_2','lag_3','lag_7','lag_14','roll_mean_7','roll_mean_14']]
selected_factor = st.selectbox("Select Contributing Factor", factor_cols)
fig3 = px.bar(filtered_df, x='CRASH DATE', y=selected_factor)
st.plotly_chart(fig3, use_container_width=True)