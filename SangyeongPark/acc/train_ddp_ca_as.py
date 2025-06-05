import math

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 하이퍼파라미터
look_back = 24
forecast_horizon = 24
input_features = ['gpu_milli', 'num_gpu']
target_col = 'gpu_milli'
epochs = 100
batch_size = 64
hidden_size = 256
dropout_rate = 0.3

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, current_score):
        if self.best_score is None or current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class ImprovedGRUCell(nn.Module):
    """
    Improved GRU with reset gate replaced by attention mechanism
    Based on MCI-GRU paper
    """
    def __init__(self, input_size, hidden_size):
        super(ImprovedGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Update gate
        self.W_z = nn.Linear(input_size, hidden_size, bias=False)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        
        # Attention mechanism (replacing reset gate)
        self.W_a = nn.Linear(input_size, hidden_size, bias=False)
        self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_a = nn.Parameter(torch.randn(hidden_size))
        
        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size, bias=False)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x, h_prev):
        # Update gate
        z_t = torch.sigmoid(self.W_z(x) + self.U_z(h_prev) + self.b_z)
        
        # Attention mechanism (replacing reset gate)
        attention_input = torch.tanh(self.W_a(x) + self.U_a(h_prev))
        attention_weights = torch.softmax(attention_input * self.v_a, dim=-1)
        attended_h = attention_weights * h_prev
        
        # Candidate hidden state
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(attended_h) + self.b_h)
        
        # Final hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t

class ImprovedGRU(nn.Module):
    """
    Improved GRU layer using ImprovedGRUCell
    """
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(ImprovedGRU, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gru_cell = ImprovedGRUCell(input_size, hidden_size)
        
    def forward(self, x):
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
        else:
            seq_len, batch_size, _ = x.size()
            x = x.transpose(0, 1)
            
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            h = self.gru_cell(x[:, t, :], h)
            outputs.append(h.unsqueeze(1))
            
        outputs = torch.cat(outputs, dim=1)
        return outputs, h

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention mechanism for capturing temporal and cross-sectional features
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.W_o(attention_output)
        output = self.output_dropout(output)
        
        return output, attention_weights

class MCI_GRU_Model(nn.Module):
    """
    MCI-GRU Model for GPU prediction
    Combines Improved GRU with Multi-Head Cross-Attention
    """
    def __init__(self, input_size, hidden_size, output_size, num_heads=4, dropout_rate=0.3):
        super(MCI_GRU_Model, self).__init__()
        self.hidden_size = hidden_size
        
        # Improved GRU layers
        self.improved_gru1 = ImprovedGRU(input_size, hidden_size)
        self.improved_gru2 = ImprovedGRU(hidden_size, hidden_size)
        
        # Multi-Head Cross-Attention
        self.cross_attention = MultiHeadCrossAttention(hidden_size, num_heads, dropout_rate)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feature interaction layers
        self.temporal_projection = nn.Linear(hidden_size, hidden_size)
        self.cross_sectional_projection = nn.Linear(hidden_size, hidden_size)
        
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
        
        # First Improved GRU layer
        gru1_output, _ = self.improved_gru1(x)
        gru1_output = self.layer_norm1(gru1_output)
        
        # Second Improved GRU layer
        gru2_output, final_hidden = self.improved_gru2(gru1_output)
        gru2_output = self.layer_norm2(gru2_output)
        
        # Prepare temporal and cross-sectional features
        # Temporal features: focus on time dimension
        temporal_features = self.temporal_projection(gru2_output)
        
        # Cross-sectional features: focus on feature interactions
        cross_sectional_features = self.cross_sectional_projection(gru2_output)
        
        # Multi-Head Cross-Attention
        # Query: temporal features, Key & Value: cross-sectional features
        attention_output, attention_weights = self.cross_attention(
            query=temporal_features,
            key=cross_sectional_features,
            value=cross_sectional_features
        )
        
        # Residual connection
        attended_features = attention_output + gru2_output
        attended_features = self.dropout(attended_features)
        
        # Use the last time step for prediction
        final_features = attended_features[:, -1, :]
        
        # Final prediction
        output = self.fc(final_features)
        
        return output, attention_weights
    
def create_dataset(data, look_back=24, forecast_horizon=24):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon):
        X.append(data[i:i+look_back])
        y.append([data[i + look_back + j][0] for j in range(forecast_horizon)])
    return np.array(X), np.array(y)

def load_data(rank):
    df = pd.read_csv(f"/data/tkddud386/repos/ddp_jin/cycle_{rank}.csv")
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

def setup(rank, world_size, local_rank):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

def cleanup():
    dist.destroy_process_group()

# DDP에서 사용할 수 있도록 수정된 학습 함수
def train_mci_gru(local_rank, rank, world_size):
    print(f"[Rank {rank}] Starting MCI-GRU training...")
    setup(rank, world_size, local_rank)
    device = torch.device(f"cuda:{local_rank}")

    splits, scaler = load_data(rank)
    train_set, val_set, test_set = splits

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = MCI_GRU_Model(
        input_size=len(input_features), 
        hidden_size=hidden_size, 
        output_size=forecast_horizon,
        num_heads=4,
        dropout_rate=dropout_rate
    ).to(device)

    ddp_model = DDP(model, device_ids=[local_rank])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    # ✅ EarlyStopping 객체 초기화
    if rank == 0:
        early_stopper = EarlyStopping(patience=20, min_delta=1e-4)

    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            outputs, attention_weights = ddp_model(batch_x)
            loss = criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            val_rmse, val_mae, val_r2 = evaluate_mci(ddp_model, val_loader, device)
            avg_loss = total_loss / len(train_loader)

            print(f"[Rank {rank}] Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {avg_loss:.6f}")
            print(f"  Val RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")

            # 스케줄러 및 EarlyStopping
            scheduler.step(val_rmse)
            early_stopper.step(val_rmse)

            # 베스트 모델 저장 조건
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                print(f"  New best model saved (Loss: {best_val_loss:.6f})")

            # 조기 종료
            if early_stopper.early_stop:
                print(f"[Rank {rank}] Early stopping triggered at epoch {epoch+1}")
                break

    if rank == 0:
        end_time = time.time()
        print(f"\n[Rank {rank}] Training completed in {end_time - start_time:.2f} seconds")
        test_rmse, test_mae, test_r2 = evaluate_mci(ddp_model, test_loader, device)
        print(f"[Rank {rank}] Final Test Results:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  R²: {test_r2:.4f}")

    cleanup()

def evaluate_mci(model, dataloader, device):
    """MCI-GRU 모델을 위한 평가 함수"""
    model.eval()
    y_true, y_pred = [], []
    total_attention_weights = []
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            outputs, attention_weights = model(x_batch)
            
            y_true.append(y_batch.numpy())
            y_pred.append(outputs.cpu().numpy())
            total_attention_weights.append(attention_weights.cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return rmse, mae, r2

# main 함수에서 사용
if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    train_mci_gru(local_rank, rank, world_size)