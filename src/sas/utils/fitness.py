import numpy as np


def unitary_distance(unitary1: np.ndarray, unitary2: np.ndarray) -> float:
    rows1, cols1 = unitary1.shape
    rows2, cols2 = unitary2.shape

    assert rows1 == rows2 and cols1 == cols2, "Unitary matrices must have same shape."

    distance = 0.0
    for row in range(rows1):
        for col in range(cols1):

            distance += abs(unitary1[row][col] - unitary2[row][col])

    normed_distance = distance / (rows1 * cols1)
    return normed_distance


def vector_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    assert len(vector1) == len(vector2)

    distance = 0.0
    for val1, val2 in zip(vector1, vector2):
        distance += abs(val1 - val2)

    normed_distance = distance / len(vector1)
    return normed_distance
