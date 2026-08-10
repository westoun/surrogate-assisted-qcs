
import numpy as np
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict

from src.sas.types import Circuit, H, S, T, CX
from .interface import ISurrogate

GATE_FREQUENCY_SURROGATE = "gate_frequency"


class Model(nn.Module):
    def __init__(self, qubit_num: int):
        super().__init__()

        feature_count = 4 * qubit_num

        # TODO: Play around with layer and neuron count.
        self.linear1 = nn.Linear(in_features=feature_count, out_features=128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        x = nn.functional.relu(x)
        x = self.linear3(x)
        x = nn.functional.relu(x)
        x = self.linear4(x)
        return x


def extract_gate_frequencies(circuit: Circuit) -> List:
    gate_frequencies = np.zeros(
        shape=(4 * circuit.qubit_num, ), dtype=float).tolist()

    for gate in circuit.gates:
        if type(gate) == H:
            gate_frequencies[4 * gate.target + 0] += 1 / len(circuit.gates)
        elif type(gate) == S:
            gate_frequencies[4 * gate.target + 1] += 1 / len(circuit.gates)
        elif type(gate) == T:
            gate_frequencies[4 * gate.target + 2] += 1 / len(circuit.gates)
        elif type(gate) == CX:
            gate_frequencies[4 * gate.control + 3] += 0.5 / len(circuit.gates)
            gate_frequencies[4 * gate.target + 3] += 0.5 / len(circuit.gates)
        else:
            raise NotImplementedError(
                f"No mapping found for gate type '{type(gate)}'")

    return gate_frequencies


class GateFrequencySurrogate(ISurrogate):
    type = GATE_FREQUENCY_SURROGATE

    model: Model
    max_epochs: int
    patience: int
    delta: float

    def __init__(self, qubit_num: int, max_epochs: int = 200, patience: int = 5, delta: float = 1e-5):
        self.model = Model(qubit_num=qubit_num)
        self.max_epochs = max_epochs
        self.patience = patience
        self.delta = delta

    def train(self, circuits: List[Circuit]) -> None:
        X = torch.Tensor([
            extract_gate_frequencies(circuit) for circuit in circuits
        ])
        y_true = torch.Tensor([
            [circuit.fitness] for circuit in circuits
        ])

        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())

        last_loss = np.inf
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):

            optimizer.zero_grad()

            y_pred = self.model(X)

            loss = criterion(y_pred, y_true)

            loss.backward()
            optimizer.step()

            if last_loss - loss <= self.delta:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0

            last_loss = loss

            if self.patience is not None and epochs_without_improvement >= self.patience:
                break

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
            "delta": self.delta
        }
