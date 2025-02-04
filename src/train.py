# import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from quantum_model import QuantumLayer
from utils import save_hyperparameters, log_metrics

# Load dataset
df_train = pd.read_csv("../data/train_data.csv")
df_val = pd.read_csv("../data/validation_data.csv")
X_train = torch.tensor(df_train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(df_train["y"].values.reshape(-1, 1), dtype=torch.float32)
X_val = torch.tensor(df_val[["x1", "x2"]].values, dtype=torch.float32)
y_val = torch.tensor(df_val["y"].values.reshape(-1, 1), dtype=torch.float32)


# Define Quantum-Classical Hybrid Model
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum = QuantumLayer()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        q_out = self.quantum(x)
        return self.fc(q_out)


model = HybridModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Save Hyperparameters
hyperparams = {
    "learning_rate": 0.01,
    "epochs": 100,
    "optimizer": "Adam",
    "loss_function": "MSELoss",
}
save_hyperparameters("../results/hyperparams.json", hyperparams)

weights = []
# Training Loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        log_metrics(
            "../results/logs/metrics.txt", f"Epoch {epoch}, Loss: {loss.item():.4f}"
        )
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# Save Model Checkpoint (Version 2)
torch.save(model.state_dict(), "../results/model_checkpoints/qnn_model_v2.pth")
print("🎯 Model Training Complete & Saved!")
