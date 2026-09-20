import numpy as np
import pandas as pd

from paths import DATA_DIR

SPAN = 32000
MESH = DATA_DIR / "flywire_brain.ply"


def frame(points: pd.DataFrame) -> tuple[np.ndarray, float]:
    # one scale for everything, so a fly that holds part of the brain still draws its
    # neurons where they belong inside it
    xyz = points.to_numpy(dtype=np.float64)
    middle = np.nanmedian(xyz, axis=0)
    return middle, float(np.nanmax(np.abs(xyz - middle)))


def scaled(xyz: np.ndarray, middle: np.ndarray, spread: float) -> np.ndarray:
    return np.nan_to_num((xyz - middle) / spread, nan=0.0) * SPAN


def atlas(root_id: np.ndarray, points: pd.DataFrame, middle: np.ndarray, spread: float) -> bytes:
    # where each of a fly's neurons sits, in the order its simulator uses, so a spike
    # can be drawn where it happened
    xyz = points.reindex(root_id).to_numpy(dtype=np.float64)
    return np.uint32(len(root_id)).tobytes() + scaled(xyz, middle, spread).astype("<i2").tobytes()


def surface(middle: np.ndarray, spread: float) -> bytes:
    # the brain's own outline, the tissue mesh FlyWire publishes, in the same frame
    raw = MESH.read_bytes()
    head = raw[: raw.index(b"end_header\n") + 11]
    lines = head.split(b"\n")
    vertices = int([line for line in lines if line.startswith(b"element vertex")][0].split()[-1])
    faces = int([line for line in lines if line.startswith(b"element face")][0].split()[-1])
    body = raw[len(head) :]
    xyz = np.frombuffer(body, dtype="<f4", count=vertices * 3).reshape(-1, 3)
    rest = np.frombuffer(body, dtype=np.uint8, offset=vertices * 12)
    corners = rest.reshape(faces, 13)[:, 1:].copy().view("<u4")
    return np.uint32([vertices, faces]).tobytes() + scaled(xyz, middle, spread).astype("<i2").tobytes() + corners.astype("<u4").tobytes()
