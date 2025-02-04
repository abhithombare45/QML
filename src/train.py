import torch
import torch.nn as nn
import torch.optim as optim
from quantum_model import QuantumLayer

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

X_train = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "../results/model_checkpoints/qnn_model.pth")
print("🎯 Model Training Complete & Saved!")
