import torch
from quantum_model import QuantumLayer

class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quantum = QuantumLayer()
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x):
        q_out = self.quantum(x)
        return self.fc(q_out)

model = HybridModel()
model.load_state_dict(torch.load("../results/model_checkpoints/qnn_model.pth"))
model.eval()

test_input = torch.tensor([[0, 1]], dtype=torch.float32)
prediction = model(test_input).detach().numpy()
print(f"🔮 Quantum AI Prediction for {test_input.tolist()[0]}: {prediction[0][0]:.3f}")
