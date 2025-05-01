import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Dataset 정의 (정규화 포함)
class SequenceDataset(Dataset):
    def __init__(self, csv_path, input_len=24, target_col='gpu_milli'):
        df = pd.read_csv(csv_path)

        self.input_len = input_len
        self.target_col = target_col

        # 정규화기 선언
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # 입력/출력 분리 및 정규화
        X_raw = df[['gpu_milli', 'cpu_milli', 'memory_mib', 'num_gpu']]
        y_raw = df[[target_col]]

        self.X = self.scaler_X.fit_transform(X_raw).astype('float32')
        self.y = self.scaler_y.fit_transform(y_raw).astype('float32').flatten()

    def __len__(self):
        return len(self.X) - self.input_len

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.input_len]
        y = self.y[idx + self.input_len]
        return torch.tensor(x), torch.tensor(y)

    def get_scalers(self):
        return self.scaler_X, self.scaler_y

# 2. GRU + Cross Attention 모델 정의
class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out  # (B, T, H)

class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        Q = self.q_proj(q)
        K = self.k_proj(k)
        V = self.v_proj(v)
        attn_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5))
        return torch.matmul(attn_weights, V)

class MultiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoders = nn.ModuleList([GRUEncoder(input_size, hidden_size) for _ in range(4)])
        self.cross_attn = CrossAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, inputs):  # inputs: list of 4 tensors (B, T, F)
        encoded = [enc(x) for enc, x in zip(self.encoders, inputs)]
        q = encoded[0]
        kv = torch.cat(encoded[1:], dim=1)
        out = self.cross_attn(q, kv, kv)
        return self.fc(out[:, -1, :]).squeeze()

# 3. 데이터 불러오기 및 분할
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

paths = [
    'gpu_dataset_part1.csv',
    'gpu_dataset_part2.csv',
    'gpu_dataset_part3.csv',
    'gpu_dataset_part4.csv'
]
datasets = [SequenceDataset(p) for p in paths]

train_loaders = []
test_loaders = []
scalers_y = []

for ds in datasets:
    split = int(len(ds) * 0.8)
    train_ds = Subset(ds, range(0, split))
    test_ds  = Subset(ds, range(split, len(ds)))

    train_loaders.append(DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True))
    test_loaders.append(DataLoader(test_ds, batch_size=32, shuffle=False, drop_last=False))

    _, y_scaler = ds.get_scalers()
    scalers_y.append(y_scaler)

# 4. 모델 및 학습 설정
model = MultiGRUModel(input_size=4, hidden_size=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 10
start_time = time.time()

# 5. 학습
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch1, batch2, batch3, batch4 in zip(*train_loaders):
        xs = [batch[0].to(device) for batch in [batch1, batch2, batch3, batch4]]
        y = batch1[1].to(device)

        optimizer.zero_grad()
        preds = model(xs)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

end_time = time.time()
print(f"\n총 학습 시간: {end_time - start_time:.2f}초")

# 6. 테스트 예측 및 역정규화
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for batch1, batch2, batch3, batch4 in zip(*test_loaders):
        xs = [batch[0].to(device) for batch in [batch1, batch2, batch3, batch4]]
        y = batch1[1].to(device)
        preds = model(xs)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

# 7. 역정규화 및 성능 평가
scaler_y = scalers_y[0]  # 기준 y_scaler 사용
preds_inv = scaler_y.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
targets_inv = scaler_y.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(targets_inv, preds_inv))
r2 = r2_score(targets_inv, preds_inv)
print(f"\nRMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# 8. 시각화
plt.figure(figsize=(12, 5))
plt.plot(targets_inv, label='Actual (gpu_milli)', linewidth=2)
plt.plot(preds_inv, label='Predicted (gpu_milli)', linestyle='--', linewidth=2)
plt.xlabel("Sample")
plt.ylabel("GPU Usage (milli)")
plt.title("GRU + Cross Attention (Test Set, Inverse-Scaled)")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig("gru_prediction_plot_scaled.png") # 시각화 파일 저장
plt.show()