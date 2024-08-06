import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Title and description
st.title('Stock Market Analysis Tool')
st.info('This app fetches and visualizes stock market data. Users can analyze historical data, track their favorite stocks, and predict future trends using machine learning models.')

# Sidebar for user inputs
st.sidebar.header('Input Options')
ticker = st.sidebar.text_input('Ticker Symbol', 'AAPL')

# Fetch real-time stock data
stock = yf.Ticker(ticker)
df = stock.history(period='1d', start='2010-01-01', end='2024-01-01')

# Display historical data
st.write(f'### {ticker} Historical Data')
st.write(df)

# Historical data visualization
st.write('### Historical Data Visualization')
fig, ax = plt.subplots()
ax.plot(df['Close'], label='Close Price')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.legend()
st.pyplot(fig)

# Technical indicators
st.write('### Technical Indicators')

# Moving Averages
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

fig, ax = plt.subplots()
ax.plot(df['Close'], label='Close Price')
ax.plot(df['MA50'], label='50-day MA')
ax.plot(df['MA200'], label='200-day MA')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Relative Strength Index (RSI)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
RS = gain / loss
df['RSI'] = 100 - (100 / (1 + RS))

fig, ax = plt.subplots()
ax.plot(df['RSI'], label='RSI')
ax.axhline(70, color='red', linestyle='--')
ax.axhline(30, color='green', linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('RSI')
ax.legend()
st.pyplot(fig)

# Stock price predictions using machine learning
st.write('### Stock Price Predictions')

# Prepare the data
df['Target'] = df['Close'].shift(-1)
df = df.dropna()
X = df[['Close', 'MA50', 'MA200', 'RSI']]
y = df['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
st.write(f'Mean Squared Error: {mse}')

# Plot predictions vs actual values
fig, ax = plt.subplots()
ax.plot(y_test.values, label='Actual Prices')
ax.plot(predictions, label='Predicted Prices')
ax.set_xlabel('Samples')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)

# Portfolio tracker
st.write('### Portfolio Tracker')

# Add stocks to the portfolio
portfolio = st.session_state.get('portfolio', [])
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

stock_to_add = st.sidebar.text_input('Add stock to portfolio (Ticker Symbol)', '')
if st.sidebar.button('Add to Portfolio'):
    if stock_to_add:
        stock_data = yf.Ticker(stock_to_add).history(period='1d', start='2010-01-01', end='2024-01-01')
        st.session_state.portfolio.append({'Ticker': stock_to_add, 'Data': stock_data})
        st.sidebar.write(f'{stock_to_add} added to portfolio.')

# Display portfolio
if st.session_state.portfolio:
    for stock in st.session_state.portfolio:
        st.write(f"### {stock['Ticker']} Historical Data")
        st.write(stock['Data'])
else:
    st.write('No stocks in the portfolio.')

