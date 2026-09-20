import numpy as np
import torch

LIMIT = 400_000


def packed(frames: torch.Tensor, limit: int = LIMIT) -> bytes:
    # the spikes of one board, from the bits saying which 5 ms frames each neuron fired
    # in: sorted, each number the gap from the one before, neuron * 64 + frame
    bits = frames.numpy().astype(np.int64)
    neurons, which = np.nonzero((bits[:, None] >> np.arange(32, dtype=np.int64)) & 1)
    out = np.sort(neurons.astype(np.int64) * 64 + which)[:limit]
    return np.diff(out, prepend=0).astype("<u4").tobytes()


def summary(blob: bytes) -> dict:
    spikes = np.cumsum(np.frombuffer(blob, dtype="<u4"), dtype="<u8")
    return {"spikes": len(spikes), "neurons": int(len(np.unique(spikes // 64))), "frames": int(spikes.max() % 64) + 1 if len(spikes) else 0}
