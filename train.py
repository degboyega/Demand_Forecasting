import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Step 1: Data Preprocessing Function
def preprocess_data(df, is_training=True):
    """Preprocess dataset for model training or future predictions."""
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract date components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # One-Hot Encoding for categorical variables
    df = pd.get_dummies(df, columns=['Category', 'Region', 'Weather Condition', 'Store ID'], drop_first=True)
    
    # Encode Seasonality (Cyclic Encoding)
    season_mapping = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}
    df['Seasonality_Num'] = df['Seasonality'].map(season_mapping)
    df['Seasonality_Sin'] = np.sin(2 * np.pi * df['Seasonality_Num'] / 4)
    df['Seasonality_Cos'] = np.cos(2 * np.pi * df['Seasonality_Num'] / 4)

    # Drop unnecessary columns
    df.drop(columns=['Date', 'Product ID', 'Seasonality', 'Seasonality_Num'], inplace=True)

    # Drop target column for prediction data
    if not is_training:
        df.drop(columns=['Units Sold'], errors='ignore', inplace=True)
    
    return df

# Step 2: Model Training Function
def train_model(df):
    """Train RandomForest model and return trained model & evaluation metrics."""
    X = df.drop(columns=['Units Sold'])
    y = df['Units Sold']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize RandomForest
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 Model Performance:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-Squared Score (R²): {r2}")

    # Save Model
    joblib.dump(model, "demand_forecasting_model.pkl")
    print("✅ Model saved as demand_forecasting_model.pkl")

    return model

#Step 3: Future Demand Prediction Function
def predict_future_demand(future_data):
    """Loads trained model and predicts demand for new data."""
    # Load saved model
    model = joblib.load("demand_forecasting_model.pkl")
    print("✅ Model loaded successfully.")

    # Preprocess new data
    processed_future_data = preprocess_data(future_data, is_training=False)

    # Predict demand
    predictions = model.predict(processed_future_data)

    # Add predictions to DataFrame
    future_data['Predicted Demand'] = predictions
    return future_data[['Date', 'Store ID', 'Predicted Demand']]

#Step 4: Running the Pipeline
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("historical_demand_data.csv")  # Replace with actual dataset
    df = preprocess_data(df, is_training=True)

    # Train model
    trained_model = train_model(df)

    # Load new data (future data to predict demand)
    future_df = pd.read_csv("future_data.csv")  # Replace with actual future data
    predictions = predict_future_demand(future_df)

    print(predictions.head())
