from datetime import datetime


def duration_to_seconds(duration: str) -> float:
    hours = int(duration.split(":")[0])
    minutes = int(duration.split(":")[1])
    seconds = float(duration.split(":")[2])

    return seconds + minutes * 60 + hours * 60 * 60


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
