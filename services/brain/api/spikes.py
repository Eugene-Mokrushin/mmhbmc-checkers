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


def varints(values: np.ndarray) -> bytes:
    # each number seven bits at a time, low bits first, the top bit saying "more to come"
    v = np.asarray(values, dtype=np.uint64)
    lengths = np.ones(len(v), dtype=np.int64)
    for shift in (7, 14, 21, 28):
        lengths += v >= np.uint64(1 << shift)
    starts = np.cumsum(lengths) - lengths
    out = np.zeros(int(lengths.sum()), dtype=np.uint8)
    for k in range(5):
        take = lengths > k
        if not take.any():
            break
        more = np.where(lengths[take] > k + 1, np.uint64(0x80), np.uint64(0))
        out[starts[take] + k] = ((v[take] >> np.uint64(7 * k)) & np.uint64(0x7F)) | more
    return out.tobytes()


def gapped(ids: np.ndarray) -> bytes:
    # a frame of firing: the neuron numbers in order, each as its gap from the one before
    return varints(np.diff(np.asarray(ids, dtype=np.int64), prepend=0))


def summary(blob: bytes) -> dict:
    spikes = np.cumsum(np.frombuffer(blob, dtype="<u4"), dtype="<u8")
    return {"spikes": len(spikes), "neurons": int(len(np.unique(spikes // 64))), "frames": int(spikes.max() % 64) + 1 if len(spikes) else 0}
