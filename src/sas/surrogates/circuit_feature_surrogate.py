
import numpy as np
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict

from src.sas.types import Circuit, H, S, T, CX
from .interface import ISurrogate

CIRCUIT_FEATURE_SURROGATE = "circuit_feature"


class Model(nn.Module):
    def __init__(self, qubit_num: int, neuron_counts: List[int], dropout: float):
        super().__init__()

        feature_count = 4 + 4 * qubit_num

        layers = [
            nn.Linear(feature_count, neuron_counts[0])
        ]

        for i, neuron_count in enumerate(neuron_counts[1:]):
            layers.append(
                nn.Linear(neuron_counts[i - 1], neuron_count)
            )
            layers.append(
                nn.ReLU()
            )
            layers.append(
                nn.Dropout(p=dropout)
            )

        layers.append(
            nn.Linear(neuron_counts[-1], 1)
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def extract_gate_frequencies(circuit: Circuit) -> List:
    gate_frequencies = np.zeros(
        shape=(4 + 4 * circuit.qubit_num, ), dtype=float).tolist()

    for gate in circuit.gates:
        if type(gate) == H:
            gate_frequencies[0] += 1 / len(circuit.gates)
            gate_frequencies[4 + 4 * gate.target + 0] += 1 / len(circuit.gates)
        elif type(gate) == S:
            gate_frequencies[1] += 1 / len(circuit.gates)
            gate_frequencies[4 + 4 * gate.target + 1] += 1 / len(circuit.gates)
        elif type(gate) == T:
            gate_frequencies[2] += 1 / len(circuit.gates)
            gate_frequencies[4 + 4 * gate.target + 2] += 1 / len(circuit.gates)
        elif type(gate) == CX:
            gate_frequencies[3] += 1 / len(circuit.gates)
            gate_frequencies[4 + 4 * gate.control +
                             3] += 0.5 / len(circuit.gates)
            gate_frequencies[4 + 4 * gate.target +
                             3] += 0.5 / len(circuit.gates)
        else:
            raise NotImplementedError(
                f"No mapping found for gate type '{type(gate)}'")

    return gate_frequencies


class CircuitFeatureSurrogate(ISurrogate):
    type = CIRCUIT_FEATURE_SURROGATE

    model: Model
    max_epochs: int
    patience: int
    delta: float
    validation_split: float
    dropout: float
    neuron_counts: List[int]

    def __init__(self, qubit_num: int,
                 neuron_counts: List[int],
                 max_epochs: int = 200,
                 patience: int = 5,
                 delta: float = 1e-5,
                 validation_split: float = 0.2,
                 dropout: float = 0.0):
        self.model = Model(qubit_num=qubit_num,
                           neuron_counts=neuron_counts, dropout=dropout)

        self.neuron_counts = neuron_counts
        self.max_epochs = max_epochs
        self.patience = patience
        self.delta = delta
        self.validation_split = validation_split
        self.dropout = dropout

    def train(self, circuits: List[Circuit]) -> None:
        X = torch.Tensor([
            extract_gate_frequencies(circuit) for circuit in circuits
        ])
        y = torch.Tensor([
            [circuit.true_fitness] for circuit in circuits
        ])

        X_train = X[:int(len(X) * (1 - self.validation_split))]
        X_val = X[int(len(X) * (1 - self.validation_split)):]
        y_train = y[:int(len(X) * (1 - self.validation_split))]
        y_val = y[int(len(X) * (1 - self.validation_split)):]

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

        # Reset predicted fitness to allow for a more objective surrogate evaluation.
        for circuit in circuits:
            circuit.surrogate_fitness = None

    def predict(self, circuits: List[Circuit]) -> List[float]:
        with torch.no_grad():
            X = torch.Tensor([
                extract_gate_frequencies(circuit) for circuit in circuits
            ])
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
            "neuron_counts": self.neuron_counts,
            "dropout": self.dropout
        }
