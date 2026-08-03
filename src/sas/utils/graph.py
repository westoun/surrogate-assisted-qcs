
import numpy as np
import networkx as nx

from .circuit import Circuit
from .gates import Gate


def circuit_to_dag(circuit: Circuit) -> nx.DiGraph:
    adjacency_matrix = np.zeros(
        shape=(len(circuit.gates), len(circuit.gates)), dtype=int
    )

    last_active_gate = {}
    for qubit in range(circuit.qubit_num):
        last_active_gate[qubit] = None

    for gate_i, gate in enumerate(circuit.gates):

        for qubit in gate.qubits:

            if last_active_gate[qubit] is not None:
                pred_i = last_active_gate[qubit]
                adjacency_matrix[pred_i][gate_i] = 1

            last_active_gate[qubit] = gate_i

    dag = nx.from_numpy_array(
        adjacency_matrix, create_using=nx.DiGraph, nodelist=[str(g) for g in circuit.gates])

    node_attrs = {}
    for node, gate in zip(dag.nodes, circuit.gates):
        node_attrs[node] = gate.__repr__().replace(
            "arget", "").replace("ontrol", "")

    nx.set_node_attributes(dag, node_attrs, "gate_config")
    return dag


def graph_to_hash(graph: nx.DiGraph) -> str:
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="gate_config")
