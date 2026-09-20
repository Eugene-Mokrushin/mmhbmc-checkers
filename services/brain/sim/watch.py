from collections.abc import Iterator

import scipy.sparse as sp
import torch

from sim.fast import FastLIF, State
from sim.kernels import step

FRAME_MS = 5
FRAMES = 32  # a 32-bit mask holds this many, which is 160 ms of window


class Watcher:
    # The serving path. The event-driven simulator asks the card what fired at every
    # step, and each question costs a wait; a thousand waits is seconds of nothing.
    # Here the spikes ride through the same weights as a matrix product, and the card is
    # asked once per 5 ms frame, which is also how often the website wants a picture.
    def __init__(self, sim: FastLIF):
        self.sim = sim
        weights = sp.csr_array((sim.rows.data.cpu().numpy(), sim.rows.indices.cpu().numpy(), sim.rows.indptr.cpu().numpy()), shape=sim.rows.shape)
        lines = sp.csr_array((sim.lines.data.cpu().numpy(), sim.lines.indices.cpu().numpy(), sim.lines.indptr.cpu().numpy()), shape=sim.lines.shape)
        self.weights = self.transposed(weights, sim.device, sim.dtype)
        self.lines = self.transposed(lines, sim.device, sim.dtype)

    @staticmethod
    def transposed(matrix: sp.csr_array, device: str, dtype) -> torch.Tensor:
        # held as columns, since every step multiplies by it from the left
        other = sp.csr_array(matrix.T)
        return torch.sparse_csr_tensor(
            torch.from_numpy(other.indptr.astype("int64")).to(device),
            torch.from_numpy(other.indices.astype("int64")).to(device),
            torch.from_numpy(other.data).to(dtype).to(device),
            other.shape,
        )

    def stream(self, inputs, state: State | None = None) -> Iterator[tuple[str, object]]:
        # ("frame", the neurons that fired in those 5 ms, over every board being judged),
        # and at the end ("end", spike counts, the frames each neuron fired in)
        sim, p = self.sim, self.sim.p
        steps, batch = inputs.shape[0], inputs.shape[1]
        lines = inputs.columns().to(sim.device) if hasattr(inputs, "columns") else torch.stack([inputs[t] for t in range(steps)]).to(sim.device).permute(0, 2, 1).contiguous()
        v, x, last = (part.T.contiguous() for part in sim.start(batch, state))
        counts = torch.zeros((sim.n, batch), dtype=torch.int32, device=sim.device)
        frames = torch.zeros((sim.n, batch), dtype=torch.int64, device=sim.device)
        lately = torch.zeros(sim.n, dtype=torch.bool, device=sim.device)
        rest = (p.v_rest + sim.bias).reshape(-1, 1)
        per_frame = max(1, round(FRAME_MS / 1000 / p.dt))
        clock = torch.zeros((), dtype=torch.int64, device=sim.device)
        for t in range(steps):
            v, x, last, spike = step(v, x, last, clock.fill_(t), rest, sim.decay_v, sim.decay_x, sim.x_to_v, p.v_reset, p.v_th, p.ref_steps)
            fired = spike.to(sim.dtype)
            counts += spike
            frames |= fired.to(torch.int64) << min(t // per_frame, FRAMES - 1)
            lately |= spike.any(dim=1)
            x = x + torch.sparse.mm(self.weights, fired) + torch.sparse.mm(self.lines, lines[t].to(sim.dtype))
            if (t + 1) % per_frame == 0:
                yield "frame", torch.nonzero(lately)[:, 0].to(torch.int32).cpu().numpy()
                lately.zero_()
        yield "end", (counts.T.cpu(), frames.T.cpu())

    def run(self, inputs, state: State | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        for kind, payload in self.stream(inputs, state):
            if kind == "end":
                return payload
        raise RuntimeError("the simulation ended without a result")
