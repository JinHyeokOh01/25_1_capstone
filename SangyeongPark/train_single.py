import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# GRU 모델
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
        self.x = torch.randn(num_samples, 5, 10)
        self.y = torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def train(epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GRUModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start = time.time()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    end = time.time()
    print(f"Training complete in {end - start:.2f} seconds")

if __name__ == "__main__":
    train()