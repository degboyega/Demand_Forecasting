import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# 🔹 Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("demand_forecasting_model.pkl")

# 🔹 Preprocessing function (same as before)
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract date components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['Category', 'Region', 'Weather Condition', 'Store ID'], drop_first=True)

    # Cyclic Encoding for Seasonality
    season_mapping = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}
    df['Seasonality_Num'] = df['Seasonality'].map(season_mapping)
    df['Seasonality_Sin'] = np.sin(2 * np.pi * df['Seasonality_Num'] / 4)
    df['Seasonality_Cos'] = np.cos(2 * np.pi * df['Seasonality_Num'] / 4)

    # Drop unnecessary columns
    df.drop(columns=['Date', 'Product ID', 'Seasonality', 'Seasonality_Num'], inplace=True, errors='ignore')

    return df

# 🔹 Predict demand function
def predict_demand(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        original_df = df.copy()  # Keep a copy for output

        # Preprocess new data
        df = preprocess_data(df)

        # Load model & make predictions
        model = load_model()
        predictions = model.predict(df)

        # Add predictions to original DataFrame
        original_df["Predicted Demand"] = predictions
        return original_df

# 🔹 Streamlit UI
st.title("📈 Demand Forecasting App")
st.write("Upload a CSV file with future data to predict demand.")

# 🔹 File Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    # Run predictions
    predictions_df = predict_demand(uploaded_file)

    if predictions_df is not None:
        st.write("### 📊 Demand Forecast Results")
        st.dataframe(predictions_df.head(10))

        # Provide Download Option
        csv = predictions_df.to_csv(index=False)
        b = BytesIO()
        b.write(csv.encode())
        b.seek(0)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=b,
            file_name="demand_predictions.csv",
            mime="text/csv",
        )
