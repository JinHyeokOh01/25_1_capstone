import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 하이퍼파라미터
look_back = 24
forecast_horizon = 24
input_features = ['gpu_milli', 'num_gpu']
target_col = 'gpu_milli'
epochs = 300
batch_size = 64
hidden_size = 256
dropout_rate = 0.3

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

def create_dataset(data, look_back=24, forecast_horizon=24):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon):
        X.append(data[i:i+look_back])
        y.append([data[i + look_back + j][0] for j in range(forecast_horizon)])
    return np.array(X), np.array(y)

def load_data():
    df = pd.read_csv("/data/tkddud386/repos/ddp_jin/gpu_1min.csv")  
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

def evaluate(model, dataloader, device):
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

def train():
    print("Starting single-GPU training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits, scaler = load_data()
    train_set, val_set, test_set = splits

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = GRUModel(input_size=len(input_features), hidden_size=hidden_size, output_size=forecast_horizon).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

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
            optimizer.step()
            total_loss += loss.item()

        val_rmse, val_mae, val_r2 = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.6f} | Val RMSE: {val_rmse:.4f} MAE: {val_mae:.4f} R2: {val_r2:.4f}")
        scheduler.step(val_rmse)

    end_time = time.time()
    print(f"Finished Training in {end_time - start_time:.2f} seconds")
    test_rmse, test_mae, test_r2 = evaluate(model, test_loader, device)
    print(f"Test RMSE: {test_rmse:.4f}\nTest MAE : {test_mae:.4f}\nTest R\u00b2  : {test_r2:.4f}")

if __name__ == "__main__":
    train()