import numpy as np


def events(raster: np.ndarray) -> np.ndarray:
    steps, neurons = np.nonzero(raster)
    return np.stack([neurons, steps], axis=1).astype(np.int32)


def to_bytes(events: np.ndarray) -> bytes:
    return events.astype("<i4").tobytes()


def from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i4").reshape(-1, 2)
