import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

file_path = 'gpu_1hour.csv'
epochs = 50
lookback_time = 24

df = pd.read_csv(file_path)
scaler_milli = MinMaxScaler()
scaler_num = MinMaxScaler()
scaler_fft = MinMaxScaler()

topn = 500
fft_signal = np.fft.fft(df['gpu_milli'].values)
fft_signal[topn:len(fft_signal)//2]=0
fft_signal[len(fft_signal)//2:-topn]=0
reconstructed_signal = np.fft.ifft(fft_signal)
df['gpu_milli_fft'] = reconstructed_signal.real

df['gpu_milli'] = scaler_milli.fit_transform(df[['gpu_milli']])
df['num_gpu'] = scaler_num.fit_transform(df[['num_gpu']])
df['gpu_milli_fft'] = scaler_fft.fit_transform(df[['gpu_milli_fft']])

features = ['gpu_milli', 'num_gpu', 'gpu_milli_fft']
data_np = df[features].values
X, y = [], []

for i in range(len(data_np) - lookback_time):
    X.append(data_np[i:i + lookback_time])
    y.append(data_np[i + lookback_time][0])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

total_len = len(X)
train_len = int(total_len * 0.7)
val_len = int(total_len * 0.1)

X_train, y_train = X[:train_len], y[:train_len]
X_val, y_val = X[train_len:train_len+val_len], y[train_len:train_len+val_len]
X_test, y_test = X[train_len+val_len:], y[train_len+val_len:]

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

class GRUModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=128):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return self.fc(x)

model = GRUModel(input_size=3, hidden_size=128).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

start = datetime.datetime.now()
for epoch in range(epochs):
    model.train()
    train_losses = []

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val).item()

    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {np.mean(train_losses):.6f}, Validation Loss: {val_loss:.6f}")

end = datetime.datetime.now()

model.eval()
with torch.no_grad():
    y_pred = model(X_test)

y_test_inv = scaler_milli.inverse_transform(y_test.cpu().numpy())
y_pred_inv = scaler_milli.inverse_transform(y_pred.cpu().numpy())

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
r2 = r2_score(y_test_inv, y_pred_inv)

print("\n=== 테스트 결과 ===")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAPE: {mape:.2f}%")
print(f"Test R² Score: {r2:.4f}")
print(f"Total Training Time: {end - start}")

plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Actual', color='red')
plt.plot(y_pred_inv, label='Predicted', linestyle='--', color='blue')
plt.title('GRU Prediction (FFT)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.close()