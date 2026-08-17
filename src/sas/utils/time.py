from datetime import datetime
from typing import Dict, List


def get_timestamp() -> str:
    return str(datetime.now())


class TimeRecorder():
    """Context manager that tracks the amount of time spent within its
    context. Said time is accumulated until the .duration value is 
    retrieved.
    """

    _duration: float

    def __init__(self):
        self._duration = 0.0

    def __enter__(self):
        self._start = datetime.now()

    def __exit__(self, exc_type, exc, tb):
        self._duration += (datetime.now() - self._start).total_seconds()

    @property
    def duration(self) -> float:
        tmp = self._duration
        self._duration = 0.0
        return tmp


class MultiTimeRecorder():
    """Wrapper that groups multipe time recorders.
    If a key is used that has not been used before, a
    new recorder is created."""

    _recorders: Dict

    def __init__(self):
        self._recorders = {}

    def __getitem__(self, key: str) -> TimeRecorder:
        if key not in self._recorders:
            self._recorders[key] = TimeRecorder()

        return self._recorders[key]
