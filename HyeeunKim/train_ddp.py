import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from sklearn.preprocessing import MinMaxScaler
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import glob

class GRUModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=1, forecast_horizon=5):
        super(GRUModel, self).__init__()
        self.output_size = 1 if forecast_horizon == 1 else forecast_horizon
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_size)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

def create_multistep_dataset(data, lookback_time=20, forecast_horizon=1):
    X, y = [], []
    for i in range(len(data) - lookback_time - forecast_horizon + 1):
        x_window = data[i:i + lookback_time]
        y_window = [data[i + lookback_time + j][0] for j in range(forecast_horizon)]
        X.append(x_window)
        y.append(y_window)
    X = np.array(X)
    y = np.array(y)
    if forecast_horizon == 1:
        y = y.squeeze()
    return X, y

def setup(rank, world_size, local_rank):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(local_rank)

def cleanup():
    dist.destroy_process_group()

def train(local_rank, rank, world_size, epochs=50):
    print(f"Rank {rank}/{world_size} on device {local_rank} starting training")
    setup(rank, world_size, local_rank)

    device = torch.device(f"cuda:{local_rank}")

    forecast_horizon = 24
    lookback_time = 24

    data = pd.read_csv(f"cycle_{rank}.csv")

    scaler_num = MinMaxScaler()
    scaler_milli = MinMaxScaler()
    data['num_gpu'] = scaler_num.fit_transform(data[['num_gpu']])
    data['gpu_milli'] = scaler_milli.fit_transform(data[['gpu_milli']])

    data_np = data[['gpu_milli', 'num_gpu']].values
    X, y = create_multistep_dataset(data_np, lookback_time=lookback_time, forecast_horizon=forecast_horizon)

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
    
    train_dataset = TensorDataset(X_train, y_train)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

    model = GRUModel(forecast_horizon=forecast_horizon).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    start = datetime.datetime.now()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        ddp_model.train()
        train_losses = []

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = ddp_model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        ddp_model.eval()
        with torch.no_grad():
            val_output = ddp_model(X_val)
            val_loss = criterion(val_output, y_val).item()

        print(f"Rank {rank} Epoch [{epoch+1}/{epochs}] - Train Loss: {np.mean(train_losses):.6f}, Validation Loss: {val_loss:.6f}")

    end = datetime.datetime.now()

    ddp_model.eval()
    with torch.no_grad():
        y_pred = ddp_model(X_test).detach()

    def inverse_gpu(scaled_values, scaler, n_features=2):
        temp = torch.zeros((scaled_values.shape[0], n_features))
        if forecast_horizon == 1:
            temp[:, 0] = scaled_values.view(-1)
            return scaler.inverse_transform(temp.cpu().numpy())[:, 0]
        else:
            result = []
            for row in scaled_values:
                temp = torch.zeros((n_features,))
                temp[0] = row[-1]
                result.append(scaler.inverse_transform(temp.unsqueeze(0).cpu().numpy())[:, 0][0])
            return np.array(result)

    y_test_inv = inverse_gpu(y_test, scaler_milli)
    y_pred_inv = inverse_gpu(y_pred, scaler_milli)

    rmse_local = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mape_local = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
    r2_local = r2_score(y_test_inv, y_pred_inv)

    print(f'Rank {rank} 테스트 결과')
    print(f"RMSE: {rmse_local:.2f}")
    print(f"MAPE: {mape_local:.2f}%")
    print(f"R² Score: {r2_local:.4f}")
    print(f"Total Training Time: {end - start}")

    np.save(f'y_test_inv_rank{rank}.npy', y_test_inv)
    np.save(f'y_pred_inv_rank{rank}.npy', y_pred_inv)

    cleanup()

    if rank == 0:
        print("\n[Rank 0] 병합 및 최종 평가 수행 중...")
        pred_files = sorted(glob.glob("y_pred_inv_rank*.npy"))
        test_files = sorted(glob.glob("y_test_inv_rank*.npy"))

        y_preds = [np.load(f) for f in pred_files]
        y_tests = [np.load(f) for f in test_files]

        y_pred_all = np.concatenate(y_preds)
        y_test_all = np.concatenate(y_tests)

        rmse_all = np.sqrt(mean_squared_error(y_test_all, y_pred_all))
        mape_all = mean_absolute_percentage_error(y_test_all, y_pred_all) * 100
        r2_all = r2_score(y_test_all, y_pred_all)

        print("\n전체 테스트 병합 결과")
        print(f"Total RMSE: {rmse_all:.2f}")
        print(f"Total MAPE: {mape_all:.2f}%")
        print(f"Total R² Score: {r2_all:.4f}")

        plt.figure(figsize=(12, 6))
        plt.plot(y_test_all, label='Actual', color='red')
        plt.plot(y_pred_all, label='Predicted', linestyle='--', color='blue')
        plt.title('Merged GRU Prediction (All Ranks)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.close()

if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    train(local_rank, rank, world_size)