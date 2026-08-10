from datetime import datetime


def get_timestamp() -> str:
    return str(datetime.now())


class TimeRecorder():
    _duration: float

    def __init__(self):
        self._duration = 0.0

    def __enter__(self):
        self._start = datetime.now()

    def __exit__(self, exc_type, exc, tb):
        self._duration += (datetime.now() - self._start).total_seconds()
        self._start = None

    @property
    def duration(self) -> float:
        tmp = self._duration
        self._duration = 0.0
        return tmp
