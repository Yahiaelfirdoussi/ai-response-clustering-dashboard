import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResponseAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 768, compressed_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, compressed_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_autoencoder(
    embeddings: np.ndarray,
    compressed_dim: int = 64,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> np.ndarray:
    """Train the autoencoder and return the compressed (N, compressed_dim) array."""
    X = torch.tensor(embeddings, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    ae = ResponseAutoencoder(input_dim=embeddings.shape[1], compressed_dim=compressed_dim)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ae.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(ae(batch), batch)
            loss.backward()
            optimizer.step()

    ae.eval()
    with torch.no_grad():
        compressed = ae.encode(X).numpy()
    return compressed
