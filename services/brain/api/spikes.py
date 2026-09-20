import numpy as np
import torch

FRAME_MS = 5
LIMIT = 400_000


def packed(fly, position, limit: int = LIMIT) -> bytes:
    # every spike of the board the fly settled on. A spike is neuron * 64 + the 5 ms
    # frame it fell in; they go out sorted, each one as the gap from the one before,
    # which costs a quarter of the bytes once compressed and adds up again in one pass.
    per_frame = max(1, round(FRAME_MS / 1000 / fly.sim.p.dt))
    inputs = fly.inputs([position])
    events, kept = [], 0
    for t, (_, fired) in enumerate(fly.sim.spikes(inputs, getattr(fly, "rest", None))):
        source = fired[1]
        if len(source) and kept < limit:
            events.append(source[: limit - kept].to(torch.int64) * 64 + t // per_frame)
            kept += len(events[-1])
    if not events:
        return b""
    order = torch.cat(events).sort().values.cpu().numpy().astype("<u8")
    return np.diff(order, prepend=0).astype("<u4").tobytes()


def summary(blob: bytes) -> dict:
    spikes = np.cumsum(np.frombuffer(blob, dtype="<u4"), dtype="<u8")
    return {"spikes": len(spikes), "neurons": int(len(np.unique(spikes // 64))), "frames": int(spikes.max() % 64) + 1 if len(spikes) else 0}
