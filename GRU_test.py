import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# 1. data load
df = pd.read_csv("gpu_30sec.csv")
features = ['gpu_milli', 'num_gpu']
target_col = 'gpu_milli'
look_back = 24
forecast_horizon = 24  # multi-step 예측 범위 (시간 수)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features].astype('float32'))

# 2. Prepare data for GRU
def create_multistep_dataset(data, look_back=24, forecast_horizon=1):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon):
        X.append(data[i:i+look_back])
        y.append([data[i + look_back + j][0] for j in range(forecast_horizon)])  # gpu_milli만 예측
    return np.array(X), np.array(y)

X, y = create_multistep_dataset(data_scaled, look_back, forecast_horizon)
X = X.reshape((X.shape[0], X.shape[1], len(features)))  # (samples, timesteps, features)

# 3. Build GRU model
model = Sequential([
    GRU(50, return_sequences=True, input_shape=(look_back, len(features))),
    GRU(50),
    Dense(forecast_horizon)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Train the model and measure time
start_time = time.time()
model.fit(X, y, epochs=100, batch_size=32, verbose=1)
elapsed_time = time.time() - start_time
print(f"\n🕒 Training Time: {elapsed_time:.2f}초")

# 5. Prediction and inverse transform
preds = model.predict(X)

# inverse transform of gpu_milli
def inverse_gpu(scaled_array, scaler, n_features):
    temp = np.zeros((len(scaled_array), n_features))
    temp[:, 0] = scaled_array
    return scaler.inverse_transform(temp)[:, 0]

preds_inv = np.array([inverse_gpu(row, scaler, len(features)) for row in preds])
y_inv = np.array([inverse_gpu(row, scaler, len(features)) for row in y])

# 6. model evaluation
rmse = np.sqrt(mean_squared_error(y_inv[:, -1], preds_inv[:, -1]))
r2 = r2_score(y_inv[:, -1], preds_inv[:, -1])
mape = mean_absolute_percentage_error(y_inv[:, -1], preds_inv[:, -1]) * 100

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# 7. visualization
plt.figure(figsize=(14, 6))
time_axis = df['time_sec'][look_back+forecast_horizon : look_back+forecast_horizon+len(y_inv)]
plt.plot(time_axis, y_inv[:, -1], label='Actual (last step)', linewidth=2)
plt.plot(time_axis, preds_inv[:, -1], label='Predicted (last step)', linestyle='--', color='r')
plt.title("GPU Usage Prediction (Within Training Range)")
plt.xlabel("Time (sec)")
plt.ylabel("GPU Usage (milli)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Future prediction
last_input = data_scaled[-look_back:]
last_input = last_input.reshape((1, look_back, len(features)))
future_pred_scaled = model.predict(last_input)[0]

# 역정규화
future_pred = inverse_gpu(future_pred_scaled, scaler, len(features))

# 시간 축
last_time = df['time_sec'].iloc[-1]
future_times = [last_time + (i + 1) * 3600 for i in range(forecast_horizon)]

# 시각화
plt.figure(figsize=(14, 6))
plt.plot(future_times, future_pred, marker='o', label='Forecasted GPU Usage (Future)', color='purple')
plt.title("Future GPU Usage Forecast (Next 24 Hours)")
plt.xlabel("Time (sec)")
plt.ylabel("GPU Usage (milli)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
