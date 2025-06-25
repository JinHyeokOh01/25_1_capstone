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

file_path = 'gpu_30sec.csv'
epochs = 50
lookback = 24
forecast_horizon = 24
batch_size = 64
hidden_size = 128
input_features = ['gpu_milli', 'num_gpu', 'gpu_milli_fft']
target_col = 'gpu_milli'
cutoff_ratio=0.03

def load_and_preprocess_data(file_path, cutoff_ratio):
    df = pd.read_csv(file_path)
    df = df[df['time_sec'] > 1e7].reset_index(drop=True)

    fft_signal = np.fft.rfft(df['gpu_milli'].values)
    cutoff_idx = int(len(fft_signal) * cutoff_ratio)
    fft_signal[cutoff_idx:] = 0
    df['gpu_milli_fft'] = np.fft.irfft(fft_signal, n=len(df))

    scalers = {
        'gpu_milli': MinMaxScaler(),
        'num_gpu': MinMaxScaler(),
        'gpu_milli_fft': MinMaxScaler()
    }
    for feature_name in scalers:
        df[feature_name] = scalers[feature_name].fit_transform(df[[feature_name]])

    return df, scalers['gpu_milli']

def create_multistep_dataset(df, features, lookback, forecast_horizon):
    data_np = df[features].values
    X, y = [], []
    for i in range(len(data_np) - lookback - forecast_horizon + 1):
        X.append(data_np[i:i + lookback])
        y.append([data_np[i + lookback + j][0] for j in range(forecast_horizon)])
    return np.array(X), np.array(y).reshape(-1, forecast_horizon)

def split_data(X, y, train_ratio=0.7, val_ratio=0.1):
    total_len = len(X)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len
    
    X_train, y_train = X[:train_len], y[:train_len]
    X_val, y_val = X[train_len:train_len+val_len], y[train_len:train_len+val_len]
    X_test, y_test = X[train_len+val_len:], y[train_len+val_len:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), train_len, val_len, test_len

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true_list, y_pred_list = [], []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch).cpu()
            y_true_list.append(y_batch)
            y_pred_list.append(outputs)

    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()
    
    return y_true, y_pred

def inverse_transform_y_true_and_y_pred(y_true, y_pred, scaler, forecast_horizon):
    if forecast_horizon == 1:
        y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1))
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    else:
        y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(-1, forecast_horizon)
        y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1, forecast_horizon)
        
    return y_true_inv, y_pred_inv

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df, scaler_milli = load_and_preprocess_data('gpu_30sec.csv', cutoff_ratio)

X, y = create_multistep_dataset(df, input_features, lookback, forecast_horizon)

(X_train, y_train), (X_val, y_val), (X_test, y_test), train_len, val_len, test_len = split_data(X, y)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

model = GRUModel(
    input_size=len(input_features),
    hidden_size=hidden_size,
    output_size=forecast_horizon
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

start_time = time.time()

train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)

end_time = time.time()

plt.figure(figsize=(12, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("train_vs_val_loss.png")
plt.close()

y_true, y_pred = evaluate_model(model, test_loader, device)

y_true_inv, y_pred_inv = inverse_transform_y_true_and_y_pred(y_true, y_pred, scaler_milli, forecast_horizon)

rmse = np.sqrt(mean_squared_error(y_true_inv[:, -1], y_pred_inv[:, -1]))
mape = mean_absolute_percentage_error(y_true_inv[:, -1], y_pred_inv[:, -1]) * 100
r2 = r2_score(y_true_inv[:, -1], y_pred_inv[:, -1])

print('\n=== 모델 평가 지표 ===')
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R² Score: {r2:.4f}")
print(f"Total Training Time: {end_time - start_time:.2f}")

test_start_idx = train_len + val_len

target_time_indices = [
    i + lookback + forecast_horizon - 1
    for i in range(test_start_idx, test_start_idx + len(y_pred_inv))
]
time_axis = df['time_sec'].iloc[target_time_indices].reset_index(drop=True)

if forecast_horizon == 1:
    y_true_plot = y_true_inv.squeeze()[:len(time_axis)]
    y_pred_plot = y_pred_inv.squeeze()[:len(time_axis)]
else:
    y_true_plot = y_true_inv[:len(time_axis), -1]
    y_pred_plot = y_pred_inv[:len(time_axis), -1]

plt.figure(figsize=(14, 6))
plt.plot(time_axis, y_true_plot, label='Actual (last step)', color='b')
plt.plot(time_axis, y_pred_plot, label='Predicted (last step)', linestyle='--', color='r')
plt.title('GPU Usage Prediction (FFT)')
plt.xlabel('Time (sec)')
plt.ylabel('Requested GPU Usage (milli)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gpu_usage_prediction_fft.png")
plt.close()