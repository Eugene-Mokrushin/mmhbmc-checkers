import math
from collections.abc import Iterator

import scipy.sparse as sp
import torch

from sim.params import LIF


class FastLIF:
    def __init__(self, weights: sp.csr_array, projection: sp.csr_array, p: LIF = LIF(), dtype=torch.float32):
        self.p, self.dtype, self.n = p, dtype, weights.shape[0]
        self.w = torch.from_numpy(weights.toarray() * p.w_syn).to(dtype)
        self.w_in = torch.from_numpy(projection.toarray() * p.w_syn * p.input_gain).to(dtype)
        # exact solution of dv/dt = (x - (v - v_rest)) / tau_m, dx/dt = -x / tau_s over one step
        self.decay_v = math.exp(-p.dt / p.tau_m)
        self.decay_x = math.exp(-p.dt / p.tau_s)
        self.x_to_v = p.tau_s / (p.tau_s - p.tau_m) * (self.decay_x - self.decay_v)

    def spikes(self, inputs: torch.Tensor) -> Iterator[torch.Tensor]:
        steps, batch, _ = inputs.shape
        p = self.p
        v = torch.full((batch, self.n), p.v_rest, dtype=self.dtype)
        x = torch.zeros((batch, self.n), dtype=self.dtype)
        last = torch.full((batch, self.n), -(10**9), dtype=torch.int64)
        for t in range(steps):
            free = t - last >= p.ref_steps
            v = torch.where(free, p.v_rest + (v - p.v_rest) * self.decay_v + x * self.x_to_v, v)
            x = x * self.decay_x
            spike = free & (v > p.v_th)
            last = torch.where(spike, t, last)
            # only the few neurons and input lines that spiked this step send anything
            for fired, weights in ((spike, self.w), (inputs[t], self.w_in)):
                row, source = fired.nonzero(as_tuple=True)
                if len(row):
                    x.index_add_(0, row, weights[source])
            v = torch.where(spike, p.v_reset, v)
            yield spike

    def counts(self, inputs: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((inputs.shape[1], self.n), dtype=torch.int32)
        for spike in self.spikes(inputs):
            total += spike
        return total

    def raster(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack(list(self.spikes(inputs)))
