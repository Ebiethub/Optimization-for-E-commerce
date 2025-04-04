import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
import pickle
import gdown


# Load the dataset
@st.cache_data
def load_data():
    # Define the Google Drive file URL
    file_id = '1t0AQbrHvizP19RVaA6Zb7NX7TpFA-tdy'  # Replace with your file ID
    file_url = f'https://drive.google.com/uc?export=download&id={file_id}'

  # Download the file
    gdown.download(file_url, 'downloaded_file.csv', quiet=False)

  # Read the file into pandas dataframe (assuming CSV file)
    data = pd.read_csv('downloaded_file.csv', encoding='ISO-8859-1')
    # data = pd.read_csv(url, encoding='ISO-8859-1')
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    return data


# Preprocess the data
def preprocess_data(data, stock_code):
    product_data = data[data['StockCode'] == stock_code]
    product_data.set_index('InvoiceDate', inplace=True)
    product_data = product_data.resample('D').sum()
    product_data.fillna(0, inplace=True)
    return product_data


import pickle

# Load the ARIMA model
with open('arima_model.pkl', 'rb') as f:
    arima_model = pickle.load(f)

import joblib

# Load the ARIMA model
# arima_model = joblib.load('arima_model.pkl')


def load_model_choice(model_type):
    if model_type == 'ARIMA':
        # Load the ARIMA model using pickle
        with open('arima_model.pkl', 'rb') as f:
            return pickle.load(f)
    elif model_type == 'Prophet':
        # Load the Prophet model using pickle
        with open('prophet_model.pkl', 'rb') as f:
            return pickle.load(f)
    elif model_type == 'LSTM':
        # Load the LSTM model using Keras
        from tensorflow.keras.models import load_model

        # Provide the custom loss function during loading
        return load_model('lstm_model.keras')



# Forecast demand using the selected model
def forecast_demand(model_type, model, data, periods):
    if model_type == 'ARIMA':
        forecast = model.forecast(steps=periods)
    elif model_type == 'Prophet':
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)['yhat'].tail(periods).values
    elif model_type == 'LSTM':
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        look_back = 60
        X = []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i, 0])
        X = np.array(X).reshape(len(X), look_back, 1)
        forecast = model.predict(X[-periods:])
        forecast = scaler.inverse_transform(forecast)

    # Ensure forecast is flattened as a NumPy array
    return np.array(forecast).flatten()


# Optimize pricing dynamically
def optimize_price(demand, base_price, elasticity=-1.5):
    optimal_price = base_price * (1 + elasticity * (demand - np.mean(demand)) / np.mean(demand))
    return np.maximum(optimal_price, 0)


# Streamlit app
def main():
    st.title("Dynamic Pricing Optimization for E-commerce")

    # Load data
    data = load_data()

    # User inputs
    st.sidebar.header("Input Parameters")
    stock_code = st.sidebar.text_input("Product Stock Code", value="85123A")
    base_price = st.sidebar.number_input("Base Price ($)", min_value=1.0, value=50.0)
    periods = st.sidebar.slider("Forecast Periods (Days)", min_value=1, max_value=30, value=7)
    model_type = st.sidebar.selectbox("Select Forecasting Model", ["ARIMA", "Prophet", "LSTM"])

    # Preprocess data
    product_data = preprocess_data(data, stock_code)

    # Load the model
    model = load_model_choice(model_type)

    # Forecast demand
    forecast = forecast_demand(model_type, model, product_data['Quantity'], periods)

    # Optimize pricing
    optimal_prices = optimize_price(forecast, base_price)

    # Display results
    st.subheader("Forecasted Demand and Optimized Prices")
    results = pd.DataFrame({
        'Date': pd.date_range(product_data.index[-1], periods=periods + 1, freq='D')[1:],
        'Forecasted Demand': forecast,
        'Optimized Price ($)': optimal_prices
    })
    st.dataframe(results)

    # Plot results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(results['Date'], results['Forecasted Demand'], label='Forecasted Demand', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Demand', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(results['Date'], results['Optimized Price ($)'], label='Optimized Price', color='green', linestyle='--')
    ax2.set_ylabel('Price ($)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    plt.title("Forecasted Demand and Optimized Prices")
    fig.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
