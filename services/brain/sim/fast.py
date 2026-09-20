import math
from collections.abc import Generator

import numpy as np
import scipy.sparse as sp
import torch

from sim.kernels import Rows, find, step
from sim.params import LIF

State = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class FastLIF:
    # Weights live only as sparse rows, so the same code runs the mushroom body
    # (0.5M connections) and the whole brain (up to 20M), on the CPU or a GPU.
    def __init__(self, weights, projection, p: LIF = LIF(), dtype=torch.float32, bias=None, device="cpu"):
        self.p, self.dtype, self.device, self.n = p, dtype, device, weights.shape[0]
        self.n_lines = projection.shape[0]
        self.bias = torch.from_numpy(np.zeros(self.n) if bias is None else np.asarray(bias)).to(dtype).to(device)
        self.rows = Rows(sp.csr_array(weights) * p.w_syn, dtype, device)
        self.lines = Rows(sp.csr_array(projection) * p.w_syn * p.input_gain, dtype, device)
        # exact solution of dv/dt = (x + bias - (v - v_rest)) / tau_m, dx/dt = -x / tau_s over one step
        self.decay_v = math.exp(-p.dt / p.tau_m)
        self.decay_x = math.exp(-p.dt / p.tau_s)
        self.x_to_v = p.tau_s / (p.tau_s - p.tau_m) * (self.decay_x - self.decay_v)

    def start(self, batch: int, state: State | None = None) -> State:
        if state is not None:
            return tuple(s.expand(batch, -1).clone() for s in state)
        v = torch.full((batch, self.n), self.p.v_rest, dtype=self.dtype, device=self.device)
        x = torch.zeros((batch, self.n), dtype=self.dtype, device=self.device)
        last = torch.full((batch, self.n), -(10**9), dtype=torch.int64, device=self.device)
        return v, x, last

    def spikes(self, inputs: torch.Tensor, state: State | None = None, scale=None) -> Generator[tuple, None, State]:
        steps, batch, _ = inputs.shape
        p, inputs = self.p, inputs.to(self.device)
        v, x, last = self.start(batch, state)
        if scale is not None:
            scale = torch.as_tensor(scale, dtype=self.dtype).to(self.device)
        rest = p.v_rest + self.bias
        clock = torch.zeros((), dtype=torch.int64, device=self.device)
        for t in range(steps):
            v, x, last, spike = step(v, x, last, clock.fill_(t), rest, self.decay_v, self.decay_x, self.x_to_v, p.v_reset, p.v_th, p.ref_steps)
            fired = find(spike)
            if len(fired[0]):
                self.rows.send(x, *fired)
            fed = find(inputs[t])
            if len(fed[0]):
                self.lines.send(x, *fed, scale)
            yield spike, fired
        return v, x, last - steps

    def settle(self, steps: int) -> State:
        run = self.spikes(torch.zeros((steps, 1, self.n_lines), dtype=torch.bool))
        while True:
            try:
                next(run)
            except StopIteration as done:
                return tuple(s[0] for s in done.value)

    def counts(self, inputs: torch.Tensor, state: State | None = None, scale=None) -> torch.Tensor:
        total = torch.zeros((inputs.shape[1], self.n), dtype=torch.int32, device=self.device)
        flat = total.view(-1)
        for _, (row, source) in self.spikes(inputs, state, scale):
            flat.index_add_(0, row * self.n + source, torch.ones_like(row, dtype=torch.int32))
        return total.cpu()

    def raster(self, inputs: torch.Tensor, state: State | None = None, scale=None) -> torch.Tensor:
        return torch.stack([spike for spike, _ in self.spikes(inputs, state, scale)]).cpu()
