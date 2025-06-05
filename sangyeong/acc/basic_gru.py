import math
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 하이퍼파라미터
look_back = 24
forecast_horizon = 12
input_features = ['gpu_milli', 'num_gpu',
                  'gpu_milli_diff', 'num_gpu_diff',
                  'gpu_milli_roll3', 'num_gpu_roll3']
target_col = 'gpu_milli'
epochs = 100
batch_size = 64
hidden_size = 64
dropout_rate = 0.3
patience = 15

class ImprovedGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(ImprovedGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        final_features = gru_out[:, -1, :]
        final_features = self.dropout(final_features)
        return self.fc(final_features)

def create_dataset(data, look_back, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon):
        X.append(data[i:i+look_back])
        y.append([data[i + look_back + j][0] for j in range(forecast_horizon)])
    return np.array(X), np.array(y)

def load_data(look_back, forecast_horizon):
    df = pd.read_csv(f"/data/tkddud386/repos/dataset/gpu_5sec.csv")
    df['gpu_milli_diff'] = df['gpu_milli'].diff().fillna(0)
    df['num_gpu_diff'] = df['num_gpu'].diff().fillna(0)
    df['gpu_milli_roll3'] = df['gpu_milli'].rolling(window=3).mean().fillna(method='bfill')
    df['num_gpu_roll3'] = df['num_gpu'].rolling(window=3).mean().fillna(method='bfill')
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[input_features].astype('float32'))
    X, y = create_dataset(data_scaled, look_back, forecast_horizon)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    total_len = len(X)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.1)
    test_len = total_len - train_len - val_len
    dataset = TensorDataset(X, y)
    return random_split(dataset, [train_len, val_len, test_len]), scaler

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            y_true.append(y_batch.numpy())
            y_pred.append(outputs.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def train_improved_gru(look_back, forecast_horizon):
    print(f"\n🚀 Training Improved GRU with look_back={look_back}, forecast_horizon={forecast_horizon}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, scaler = load_data(look_back, forecast_horizon)
    train_set, val_set, test_set = splits
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = ImprovedGRUModel(
        input_size=len(input_features),
        hidden_size=hidden_size,
        output_size=forecast_horizon,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_val_rmse = float('inf')
    early_stop_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_rmse, val_mae, val_r2 = evaluate_model(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_loss:.6f} | Val RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")
        scheduler.step(val_rmse)

        # 수정된 EarlyStopping 조건
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    test_rmse, test_mae, test_r2 = evaluate_model(model, test_loader, device)
    print(f"\n✅ Final Test Results:")
    print(f"RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

if __name__ == "__main__":
    train_improved_gru(look_back, forecast_horizon)