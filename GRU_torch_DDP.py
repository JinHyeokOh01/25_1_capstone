# ddp_gru_train.py
import os
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import time

# ---------- DDP Setup ----------
def setup(rank, world_size):
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ---------- GRU Model ----------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x[:, -1, :]
        return self.fc(x)

# ---------- Main DDP Entry ----------
def train_ddp(rank, world_size):
    print(f"Rank {rank} is starting training...")
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 1. Load and preprocess
    df = pd.read_csv("gpu_1hour.csv")
    features = ['gpu_milli', 'num_gpu']
    look_back = 24
    forecast_horizon = 24

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[features].astype('float32'))

    def create_multistep_dataset(data, look_back=24, forecast_horizon=1):
        X, y = [], []
        for i in range(len(data) - look_back - forecast_horizon):
            X.append(data[i:i+look_back])
            y.append([data[i + look_back + j][0] for j in range(forecast_horizon)])
        return np.array(X), np.array(y)

    X, y = create_multistep_dataset(data_scaled, look_back, forecast_horizon)
    X = torch.tensor(X, dtype=torch.float32).reshape((-1, look_back, len(features)))
    y = torch.tensor(y, dtype=torch.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, shuffle=True, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

    # 2. Build model
    model = GRUModel(input_size=len(features), hidden_size=50, output_size=forecast_horizon).to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001 * world_size)

    # 3. Train
    start_time = time.time()
    for epoch in range(100):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0 and (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100] Loss: {total_loss/len(train_loader):.4f}")

    if rank == 0:
        print(f"\n🕒 Training Time: {time.time() - start_time:.2f}초")

        # 예측 (선택적, 단일 GPU만 수행)
        def inverse_gpu(scaled_array, scaler, n_features):
            temp = np.zeros((len(scaled_array), n_features))
            temp[:, 0] = scaled_array
            return scaler.inverse_transform(temp)[:, 0]

        model.eval()
        with torch.no_grad():
            preds_all = model.module(X.to(device)).cpu().numpy()
            preds_all_inv = np.array([inverse_gpu(row, scaler, len(features)) for row in preds_all])
            y_inv = np.array([inverse_gpu(row, scaler, len(features)) for row in y.numpy()])
            time_axis = df['time_sec'][look_back+forecast_horizon : look_back+forecast_horizon+len(preds_all_inv)]

        plt.figure(figsize=(14, 6))
        plt.plot(time_axis, y_inv[:, -1], label='Actual', linewidth=2)
        plt.plot(time_axis, preds_all_inv[:, -1], label='Predicted', linestyle='--', color='r')
        plt.title("GPU Usage Prediction (DDP)")
        plt.xlabel("Time (sec)")
        plt.ylabel("GPU Usage (milli)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    cleanup()

# ---------- LINUX Entry ----------
if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train_ddp(rank, world_size)

# To run this script, use the following command:
# torchrun --nproc_per_node=4 GRU_torch_DDP.py

# # ---------- Windows Entry ----------
# if __name__ == "__main__": 
#     world_size = torch.cuda.device_count()  # ex. 2 or more
#     mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)