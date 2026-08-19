import tracemalloc
from typing import Dict


class MemoryRecorder():
    """Context manager that tracks peak memory usage within 
    its context. The current peak is tracked until its value
    is retrieved.
    """

    _peak: float
    _baseline: float

    def __init__(self):
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        self._peak = None
        self._baseline = None

    def __enter__(self):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()

        self._baseline = current
        self._peak = None


    def __exit__(self, exc_type, exc, tb):
        current, peak = tracemalloc.get_traced_memory()

        self._peak = peak 

    @property
    def peak(self) -> float:
        if self._peak is None:
            return None

        tmp = self._peak - self._baseline
        self._peak = None
        self._baseline = 0.0
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
