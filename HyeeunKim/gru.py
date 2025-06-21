import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

file_path = 'gpu_30sec.csv'
epochs = 50
forecast_horizon = 24
lookback_time = 24
batch_size = 64

df = pd.read_csv(file_path)
scaler_num = MinMaxScaler()
scaler_milli = MinMaxScaler()

df['gpu_milli'] = scaler_milli.fit_transform(df[['gpu_milli']])
df['num_gpu'] = scaler_num.fit_transform(df[['num_gpu']])

data_np = df[['gpu_milli', 'num_gpu']].values
X, y = [], []
for i in range(len(data_np) - lookback_time - forecast_horizon + 1):
    X.append(data_np[i:i + lookback_time])
    y.append([data_np[i + lookback_time + j][0] for j in range(forecast_horizon)])
X = np.array(X)
y = np.array(y)
if forecast_horizon == 1:
    y = y.squeeze()

total_len = len(X)
train_len = int(total_len * 0.7)
val_len = int(total_len * 0.1)
test_len = total_len - train_len - val_len

X_train, y_train = X[:train_len], y[:train_len]
X_val, y_val = X[train_len:train_len+val_len], y[train_len:train_len+val_len]
X_test, y_test = X[train_len+val_len:], y[train_len+val_len:]

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32)

if forecast_horizon == 1:
    y_train = y_train.unsqueeze(1)
    y_val = y_val.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

y_train = y_train.to(device)
y_val = y_val.to(device)
y_test = y_test.to(device)

class GRUModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=1, forecast_horizon=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return self.fc(x)

model = GRUModel(forecast_horizon=forecast_horizon).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(X_train, y_train)
val_dataset   = TensorDataset(X_val, y_val)
test_dataset  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

start_time = time.time()
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss/len(train_loader):.6f}, Validation Loss: {val_loss/len(train_loader):.6f}")

end_time = time.time()

model.eval()
y_pred_list = []
with torch.no_grad():
    for x_batch, _ in test_loader:
        x_batch = x_batch.to(device)
        preds = model(x_batch).detach().cpu()
        y_pred_list.append(preds)

y_pred = torch.cat(y_pred_list, dim=0)

y_test_inv = scaler_milli.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).reshape(-1, forecast_horizon)
y_pred_inv = scaler_milli.inverse_transform(y_pred.cpu().numpy().reshape(-1, 1)).reshape(-1, forecast_horizon)

rmse = np.sqrt(mean_squared_error(y_test_inv[:, -1], y_pred_inv[:, -1]))
mape = mean_absolute_percentage_error(y_test_inv[:, -1], y_pred_inv[:, -1]) * 100
r2 = r2_score(y_test_inv[:, -1], y_pred_inv[:, -1])

print('\n=== 테스트 결과 ===')
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")
print(f"Total Training Time: {end_time - start_time:.2f}")

start_index = train_len + val_len + lookback_time + forecast_horizon - 1
time_axis = df['time_sec'][start_index : start_index + len(y_pred)]

plt.figure(figsize=(16, 6))
plt.plot(time_axis, y_test_inv[:, -1], label='Actual (last step)', color='b')
plt.plot(time_axis, y_pred_inv[:, -1], label='Predicted (last step)', linestyle='--', color='r')
plt.title('GRU Usage Prediction')
plt.xlabel('Time (sec)')
plt.ylabel('Requested GPU Usage (milli)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.close()