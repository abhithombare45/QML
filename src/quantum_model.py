import pennylane as qml
import torch
import torch.nn as nn

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=[0, 1])
    qml.BasicEntanglerLayers(weights, wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(2, requires_grad=True))

    def forward(self, x):
        return quantum_circuit(x, self.weights)


print("✅ Quantum Neural Network (QNN) Ready!")
