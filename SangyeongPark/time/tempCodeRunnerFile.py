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
forecast_horizon = 24
input_features = ['gpu_milli', 'num_gpu',
                  'gpu_milli_diff', 'num_gpu_diff',
                  'gpu_milli_roll3', 'num_gpu_roll3']
target_col = 'gpu_milli'
epochs = 50
batch_size = 64
hidden_size = 256
dropout_rate = 0.3

class BasicGRUModel(nn.Module):
    """
    Basic GRU Model for GPU prediction
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(BasicGRUModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Basic GRU layers
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout_rate)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True, dropout=dropout_rate)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final prediction layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # First GRU layer
        gru1_output, _ = self.gru1(x)
        gru1_output = self.layer_norm1(gru1_output)
        
        # Second GRU layer
        gru2_output, final_hidden = self.gru2(gru1_output)
        gru2_output = self.layer_norm2(gru2_output)
        
        # Apply dropout
        gru2_output = self.dropout(gru2_output)
        
        # Use the last time step for prediction
        final_features = gru2_output[:, -1, :]
        
        # Final prediction
        output = self.fc(final_features)
        
        return output

def create_dataset(data, look_back=24, forecast_horizon=24):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon):
        X.append(data[i:i+look_back])
        y.append([data[i + look_back + j][0] for j in range(forecast_horizon)])
    return np.array(X), np.array(y)

def load_data(rank=0):
    df = pd.read_csv(f"/data/tkddud386/repos/dataset/gpu_5sec.csv")
    
    # 추가된 파생 변수 생성
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

def train_basic_gru():
    print("Starting Basic GRU training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    splits, scaler = load_data(rank=0)
    train_set, val_set, test_set = splits

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = BasicGRUModel(
        input_size=len(input_features), 
        hidden_size=hidden_size, 
        output_size=forecast_horizon,
        dropout_rate=dropout_rate
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    start_time = time.time()
    best_val_loss = float('inf')

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

        val_rmse, val_mae, val_r2 = evaluate_basic(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {avg_loss:.6f}")
        print(f"  Val RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")

        scheduler.step(val_rmse)

        # 베스트 모델 저장 조건
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            print(f"  New best model saved (Loss: {best_val_loss:.6f})")

    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    test_rmse, test_mae, test_r2 = evaluate_basic(model, test_loader, device)
    print(f"Final Test Results:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R²: {test_r2:.4f}")

def evaluate_basic(model, dataloader, device):
    """Basic GRU 모델을 위한 평가 함수"""
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

if __name__ == "__main__":
    train_basic_gru()