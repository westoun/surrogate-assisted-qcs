from datetime import datetime


def get_timestamp() -> str:
    return str(datetime.now())


class TimeRecorder():

    def __enter__(self):
        self.start = datetime.now()

    def __exit__(self, exc_type, exc, tb):
        self.end = datetime.now()

    @property
    def duration(self) -> float:
        return (self.end - self.start).total_seconds()
