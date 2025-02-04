import torch
import pandas as pd
from quantum_model import QuantumLayer

# Load test data
df_test = pd.read_csv("../data/test_data.csv")
X_test = torch.tensor(df_test[["x1", "x2"]].values, dtype=torch.float32)
y_test = torch.tensor(df_test["y"].values.reshape(-1, 1), dtype=torch.float32)


# Define Hybrid Model (Same as training)
class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum = QuantumLayer()
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x):
        q_out = self.quantum(x)
        return self.fc(q_out)


# Load Trained Model
model = HybridModel()
model.load_state_dict(torch.load("../results/model_checkpoints/qnn_model.pth"))
model.eval()

# Make Predictions
predictions = model(X_test).detach().numpy()
print("\n🔮 Quantum AI Predictions:")
for i, (inp, pred) in enumerate(zip(X_test.tolist(), predictions)):
    print(f"Input: {inp} → Predicted: {pred[0]:.3f} | Actual: {y_test[i].item()}")
