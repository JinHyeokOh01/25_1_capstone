import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense


# Load Airline Passenger dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = pd.read_csv(url, usecols=[1])
data = data.values.astype('float32')


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)


# Prepare data for GRU (sliding window approach)
def create_dataset(data, look_back=1):
   X, y = [], []
   for i in range(len(data)-look_back-1):
       a = data[i:(i+look_back), 0]
       X.append(a)
       y.append(data[i + look_back, 0])
   return np.array(X), np.array(y)


look_back = 10
X, y = create_dataset(data, look_back)


# Reshape input to be [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))


# Build GRU model
model_gru_ts = Sequential()
model_gru_ts.add(GRU(50, return_sequences=True, input_shape=(look_back, 1)))
model_gru_ts.add(GRU(50))
model_gru_ts.add(Dense(1))


# Compile model
model_gru_ts.compile(optimizer='adam', loss='mean_squared_error')


# Train the model
model_gru_ts.fit(X, y, epochs=100, batch_size=32, verbose=1)


# Make predictions
predictions = model_gru_ts.predict(X)


# Inverse transform the predictions
predictions = scaler.inverse_transform(predictions)
y_actual = scaler.inverse_transform([y])


# Plot original data vs predictions
plt.figure(figsize=(12,6))
plt.plot(y_actual[0], label='Original Data')
plt.plot(predictions, label='Predicted Data', color='r')
plt.xlabel('Time')
plt.ylabel('Number of Passengers')
plt.title('Time-Series Forecasting using GRU')
plt.legend()
plt.show()