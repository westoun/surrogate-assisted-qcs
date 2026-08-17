import tracemalloc
from typing import Dict


class MemoryRecorder():
    """Context manager that tracks peak memory usage within 
    its context. The current peak is tracked until its value
    is retrieved.
    """

    _peak: float

    def __init__(self):
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        self._peak = None

    def __enter__(self):
        self._peak = 0.0
        tracemalloc.reset_peak()

    def __exit__(self, exc_type, exc, tb):
        current, peak = tracemalloc.get_traced_memory()

        if self._peak is None:
            self._paek = peak 
        else:
            self._peak = max(peak, self._peak)

    @property
    def peak(self) -> float:
        tmp = self._peak
        self._peak = None
        return tmp


class MultiMemoryRecorder():
    """Wrapper that groups multipe memory recorders.
    If a key is used that has not been used before, a
    new recorder is created."""

    _recorders: Dict

    def __init__(self):
        self._recorders = {}

    def __getitem__(self, key: str) -> MemoryRecorder:
        if key not in self._recorders:
            self._recorders[key] = MemoryRecorder()

        return self._recorders[key]
