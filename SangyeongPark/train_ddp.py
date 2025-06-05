import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# 간단한 GRU 모델
class GRUModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

# 더미 데이터셋
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.x = torch.randn(num_samples, 5, 10)  # (batch, seq_len, input_size)
        self.y = torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(local_rank, rank, world_size):
    print(f"Rank {rank}/{world_size} on device {local_rank} starting training")
    setup(rank, world_size)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    model = GRUModel().to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    start_time = time.time()

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Rank {rank} Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    end_time = time.time()
    print(f"Rank {rank} finished training in {end_time - start_time:.2f} seconds")

    cleanup()

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train(local_rank, rank, world_size)