import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# 1. Data Load
df = pd.read_csv("dataset/gpu_1hour.csv")
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
X = torch.tensor(X, dtype=torch.float32).reshape((-1, look_back, len(features)))
y = torch.tensor(y, dtype=torch.float32)

# Train/Validation/Test Split (7:1:2 비율)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, shuffle=True, random_state=42)

# 3. Define PyTorch GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x[:, -1, :]  # 마지막 타임스텝의 출력을 사용
        x = self.fc(x)
        return x

# 모델 설정
input_size = len(features)
hidden_size = 50
output_size = forecast_horizon
model = GRUModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Train the model
start_time = time.time()
epochs = 50
batch_size = 32

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

elapsed_time = time.time() - start_time
print(f"\n🕒 Training Time: {elapsed_time:.2f}초")

# 5. Prediction and inverse transform
def inverse_gpu(scaled_array, scaler, n_features):
    temp = np.zeros((len(scaled_array), n_features))
    temp[:, 0] = scaled_array
    return scaler.inverse_transform(temp)[:, 0]

# 전체 데이터 예측 (원본 크기 유지)
preds_all = model(X).detach().numpy()
preds_all_inv = np.array([inverse_gpu(row, scaler, len(features)) for row in preds_all])
y_inv = np.array([inverse_gpu(row, scaler, len(features)) for row in y.numpy()])


# 시간 축 설정 (원본 데이터 크기 유지)
time_axis = df['time_sec'][look_back+forecast_horizon : look_back+forecast_horizon+len(preds_all_inv)]

# 시각화
plt.figure(figsize=(14, 6))
plt.plot(time_axis, y_inv[:, -1], label='Actual (last step)', linewidth=2)
plt.plot(time_axis, preds_all_inv[:, -1], label='Predicted (last step)', linestyle='--', color='r')
plt.title("GPU Usage Prediction (Full Range)")
plt.xlabel("Time (sec)")
plt.ylabel("GPU Usage (milli)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Model Evaluation
preds_test = model(X_test).detach().numpy()
y_test_inv = np.array([inverse_gpu(row, scaler, len(features)) for row in y_test.numpy()])
preds_test_inv = np.array([inverse_gpu(row, scaler, len(features)) for row in preds_test])

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test_inv[:, -1], preds_test_inv[:, -1]))

# R² Score
r2 = r2_score(y_test_inv[:, -1], preds_test_inv[:, -1])

# MAPE (Mean Absolute Percentage Error)
mape = mean_absolute_percentage_error(y_test_inv[:, -1], preds_test_inv[:, -1]) * 100

# Print Evaluation Metrics
print(f"\n🔹 Model Evaluation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# 6. Future prediction
last_input = torch.tensor(data_scaled[-look_back:], dtype=torch.float32).reshape((1, look_back, len(features)))
model.eval()
with torch.no_grad():
    future_pred_scaled = model(last_input).numpy()[0]  # shape: (24,)

# 역정규화 (24시간치 예측)
future_pred = np.array([inverse_gpu([v], scaler, len(features))[0] for v in future_pred_scaled])

# 시간 축
last_time = df['time_sec'].iloc[-1]
future_times = [last_time + (i + 1) * 3600 for i in range(forecast_horizon)]

# 시각화
plt.figure(figsize=(14, 6))
plt.plot(future_times, future_pred, marker='o', label='Forecasted GPU Usage (Future)', color='purple')
plt.title("Future GPU Usage Forecast (Next 24 Steps, PyTorch Style)")
plt.xlabel("Time (sec)")
plt.ylabel("GPU Usage (milli)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
