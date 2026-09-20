import numpy as np
import pandas as pd

SPAN = 32000


def atlas(root_id: np.ndarray, points: pd.DataFrame) -> bytes:
    # where each of a fly's neurons sits in the brain, in the order its simulator uses,
    # so a spike can be drawn where it happened: the count, then xyz as 16-bit numbers
    xyz = points.reindex(root_id).to_numpy(dtype=np.float64)
    middle = np.nanmedian(xyz, axis=0)
    spread = np.nanmax(np.abs(xyz - middle))
    scaled = np.nan_to_num((xyz - middle) / spread, nan=0.0) * SPAN
    return np.uint32(len(root_id)).tobytes() + scaled.astype("<i2").tobytes()
