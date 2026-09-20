import numpy as np

from paths import DATA_DIR

BODY = DATA_DIR / "flybody.npz"
STEP = 3200  # sixteen-bit steps to one brain half-width, so the wings still fit


def outline(spread: float, heart: np.ndarray) -> bytes:
    # The fly the brain belongs to, drawn in the same frame as its neurons: the whole
    # body model published with flybody (Vaxenburg et al.), traced from a micro-CT scan,
    # placed at its own true size with the middle of its head on the middle of the brain.
    with np.load(BODY) as kept:
        points, faces = kept["points"], kept["faces"]
    put = np.clip(points / spread + heart, -10, 10) * STEP
    return np.uint32([len(points), len(faces)]).tobytes() + put.astype("<i2").tobytes() + faces.astype("<u4").tobytes()
