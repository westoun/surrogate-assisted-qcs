from datetime import datetime


def get_timestamp() -> str:
    return str(datetime.now())


class TimeRecorder():
    aggregate: bool
    duration: float

    def __init__(self, aggregate: bool = False):
        self.aggregate = aggregate
        self.duration = 0.0

    def __enter__(self):
        if not self.aggregate:
            self.duration = 0.0

        self.start = datetime.now()

    def __exit__(self, exc_type, exc, tb):
        self.duration = (datetime.now() - self.start).total_seconds()
        self.start = None

    def reset(self) -> None:
        self.start = None
        self.duration = 0.0
