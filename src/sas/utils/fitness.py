import numpy as np


def compute_distance(unitary1: np.ndarray, unitary2: np.ndarray) -> float:
    rows1, cols1 = unitary1.shape
    rows2, cols2 = unitary2.shape

    assert rows1 == rows2 and cols1 and cols2, "Unitary matrices must have same shape."

    distance = 0.0
    for row in range(rows1):
        for col in range(cols1):

            distance += abs(unitary1[row][col] - unitary2[row][col])

    normed_distance = distance / (rows1 * cols1)
    return normed_distance
