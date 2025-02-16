import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

class SupervisedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super(SupervisedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.classifier = nn.Linear(encoding_dim, 1)  # Supervised classification

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        prediction = torch.sigmoid(self.classifier(encoded))
        return reconstructed, prediction

def train_supervised_autoencoder(X, y, input_dim, epochs=10, batch_size=128, lr=1e-3):
    """
    Train a supervised autoencoder with anomaly detection.
    """
    model = SupervisedAutoencoder(input_dim)
    criterion_reconstruction = nn.MSELoss()
    criterion_classification = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            reconstructed, predicted_y = model(batch_x)
            
            # Combined loss
            loss = criterion_reconstruction(reconstructed, batch_x) + criterion_classification(predicted_y.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model
