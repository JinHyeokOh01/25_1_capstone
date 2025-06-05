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

# 변화율
df['gpu_milli_diff'] = df['gpu_milli'].diff().fillna(0)
df['num_gpu_diff'] = df['num_gpu'].diff().fillna(0)

# Rolling Mean (이동 평균)
df['gpu_milli_roll3'] = df['gpu_milli'].rolling(window=3).mean().fillna(method='bfill')
df['num_gpu_roll3'] = df['num_gpu'].rolling(window=3).mean().fillna(method='bfill')

features = ['gpu_milli', 'num_gpu', 
            'gpu_milli_diff', 'num_gpu_diff', 
            'gpu_milli_roll3', 'num_gpu_roll3']
target_col = 'gpu_milli'
look_back = 24
forecast_horizon = 24  # multi-step 예측 범위

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features].astype('float32'))

# 2. Prepare data
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

# 3. GRU 모델 정의
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x[:, -1, :]  # 마지막 타임스텝
        return self.fc(x)

# 4. 모델 학습 설정
input_size = len(features)
hidden_size = 50
output_size = forecast_horizon
model = GRUModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Training loop (Train + Validation Loss 추적)
epochs = 500
train_losses = []
val_losses = []
start_time = time.time()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    train_loss = criterion(output, y_train)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)

    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

elapsed_time = time.time() - start_time
print(f"\n🕒 Total Training Time: {elapsed_time:.2f}초")

# 6. 손실 시각화
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6-1. 오버피팅/언더피팅 판단
def check_overfitting(train_losses, val_losses):
    final_train = train_losses[-1]
    final_val = val_losses[-1]
    gap = abs(final_val - final_train)

    print("\n🔍 Overfitting/Underfitting Check:")
    print(f"final_train_loss: {final_train:.4f}")
    print(f"final_val_loss: {final_val:.4f}")
    print(f"gap: {gap:.4f}")


# 호출
check_overfitting(train_losses, val_losses)


# 7. 예측 후 역정규화
def inverse_gpu(scaled_array, scaler, n_features):
    temp = np.zeros((len(scaled_array), n_features))
    temp[:, 0] = scaled_array
    return scaler.inverse_transform(temp)[:, 0]

preds_all = model(X).detach().numpy()
preds_all_inv = np.array([inverse_gpu(row, scaler, len(features)) for row in preds_all])
y_inv = np.array([inverse_gpu(row, scaler, len(features)) for row in y.numpy()])
time_axis = df['time_sec'][look_back+forecast_horizon : look_back+forecast_horizon+len(preds_all_inv)]

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

# 8. 모델 평가
preds_test = model(X_test).detach().numpy()
y_test_inv = np.array([inverse_gpu(row, scaler, len(features)) for row in y_test.numpy()])
preds_test_inv = np.array([inverse_gpu(row, scaler, len(features)) for row in preds_test])

rmse = np.sqrt(mean_squared_error(y_test_inv[:, -1], preds_test_inv[:, -1]))
r2 = r2_score(y_test_inv[:, -1], preds_test_inv[:, -1])
mape = mean_absolute_percentage_error(y_test_inv[:, -1], preds_test_inv[:, -1]) * 100

print(f"\n🔹 Model Evaluation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# 9. 미래 예측
last_input = torch.tensor(data_scaled[-look_back:], dtype=torch.float32).reshape((1, look_back, len(features)))
model.eval()
with torch.no_grad():
    future_pred_scaled = model(last_input).numpy()[0]

future_pred = np.array([inverse_gpu([v], scaler, len(features))[0] for v in future_pred_scaled])
last_time = df['time_sec'].iloc[-1]
future_times = [last_time + (i + 1) * 3600 for i in range(forecast_horizon)]

plt.figure(figsize=(14, 6))
plt.plot(future_times, future_pred, marker='o', label='Forecasted GPU Usage (Future)', color='purple')
plt.title("Future GPU Usage Forecast (Next 24 Steps)")
plt.xlabel("Time (sec)")
plt.ylabel("GPU Usage (milli)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
