
import numpy as np
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import List, Tuple, Dict

from src.sas.types import Circuit, H, S, T, CX
from .interface import ISurrogate

GNN_SURROGATE = "gnn"


def circuit_to_pyg_data(circuit: Circuit) -> Data:

    GATE2ID = {
        H: 0,
        S: 1,
        T: 2,
        CX: 3
    }

    node_features = []
    edges_from = []
    edges_to = []

    last_active_gate = {}
    for qubit in range(circuit.qubit_num):
        last_active_gate[qubit] = None

    for gate_i, gate in enumerate(circuit.gates):

        node_features.append([
            GATE2ID[type(gate)],
            gate.qubits[0],
            gate.qubits[1] if len(gate.qubits) > 1 else -1
        ])

        for qubit in gate.qubits:

            if last_active_gate[qubit] is not None:
                pred_i = last_active_gate[qubit]

                edges_from.append(pred_i)
                edges_to.append(gate_i)

            last_active_gate[qubit] = gate_i

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([
        edges_from, edges_to
    ], dtype=torch.long)

    return Data(
        x=x, edge_index=edge_index
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = GCNConv(3, 32)
        self.layer2 = GCNConv(32, 32)
        self.layer3 = nn.Linear(32, 1)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        x = self.layer1(x, edge_index)
        x = nn.functional.relu(x)
        x = self.layer2(x, edge_index)
        x = nn.functional.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.layer3(x)
        return x


class GNNSurrogate(ISurrogate):
    type = GNN_SURROGATE

    model: Model
    max_epochs: int
    patience: int
    delta: float
    validation_split: float

    def __init__(self,
                 max_epochs: int = 200,
                 patience: int = 5,
                 delta: float = 1e-5,
                 validation_split: float = 0.2):
        self.model = Model()

        self.max_epochs = max_epochs
        self.patience = patience
        self.delta = delta
        self.validation_split = validation_split

    def train(self, circuits: List[Circuit]) -> None:
        X = [
            circuit_to_pyg_data(circuit) for circuit in circuits
        ]
        y = torch.Tensor([
            [circuit.fitness] for circuit in circuits
        ])

        X_train = X[:int(len(X) * self.validation_split)]
        # Needed grouping of data as batch since cannot create
        # tensor of graph data like we did with normal ANN.
        X_train = Batch.from_data_list(X_train)

        X_val = X[int(len(X) * self.validation_split):]
        X_val = Batch.from_data_list(X_val)

        y_train = y[:int(len(X) * self.validation_split)]
        y_val = y[int(len(X) * self.validation_split):]

        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())

        last_val_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):

            optimizer.zero_grad()

            y_pred = self.model(X_train)

            loss = criterion(y_pred, y_train)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y_val_pred = self.model(X_val)
                val_loss = criterion(y_val_pred, y_val)

            if last_val_loss - val_loss <= self.delta:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0

            last_val_loss = val_loss

            if self.patience is not None and epochs_without_improvement >= self.patience:
                break

    def predict(self, circuits: List[Circuit]) -> List[float]:
        with torch.no_grad():
            X = [
                circuit_to_pyg_data(circuit) for circuit in circuits
            ]
            X = Batch.from_data_list(X)
            predictions = self.model(X)
            predictions = predictions.tolist()
            predictions = [
                pred[0] for pred in predictions
            ]
            return predictions

    @property
    def params(self) -> Dict:
        return {
            "type": self.type,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "delta": self.delta,
            "validation_split": self.validation_split,
        }
