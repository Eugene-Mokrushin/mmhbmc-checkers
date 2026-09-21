import math

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.checkpoint import checkpoint

from sim.params import LIF
from sim.surrogate import Pattern, Propagate, Spike

SEGMENT = 25


class GradLIF(torch.nn.Module):
    # The same neurons and equations as FastLIF, made differentiable: every connection
    # keeps its sign and synapse count, times a learned factor exp(gain) that starts at 1.
    # Input lines each drive one neuron, as the photoreceptors do. The output is the
    # spike count of each readout neuron over the window.
    def __init__(self, weights: sp.csr_array, projection: sp.csr_array, readout: np.ndarray, p: LIF, device="cpu"):
        super().__init__()
        w = sp.csr_array(weights)
        w.sort_indices()
        lines = sp.csr_array(projection)
        assert (np.diff(lines.indptr) == 1).all(), "each input line drives exactly one neuron"
        self.p, self.n, self.device = p, w.shape[0], device
        self.pattern = Pattern(w.indptr, w.indices, self.n, device)
        self.register_buffer("base", torch.as_tensor(w.data * p.w_syn, dtype=torch.float32))
        self.register_buffer("target", torch.as_tensor(lines.indices, dtype=torch.int64))
        self.register_buffer("kick", torch.as_tensor(lines.data * p.w_syn * p.input_gain, dtype=torch.float32))
        self.register_buffer("readout", torch.as_tensor(readout, dtype=torch.int64))
        self.gain = torch.nn.Parameter(torch.zeros(len(w.data)))
        self.to(device)
        self.decay_v = math.exp(-p.dt / p.tau_m)
        self.decay_x = math.exp(-p.dt / p.tau_s)
        self.x_to_v = p.tau_s / (p.tau_s - p.tau_m) * (self.decay_x - self.decay_v)

    def values(self) -> torch.Tensor:
        return self.base * torch.exp(self.gain)

    def weights(self) -> sp.csr_array:
        # the learned connectome in synapse units, for the fast simulator
        data = (self.values() / self.p.w_syn).detach().cpu().double().numpy()
        return sp.csr_array((data, self.pattern.indices.cpu().numpy(), self.pattern.indptr.cpu().numpy()), shape=(self.n, self.n))

    def segment(self, v, x, last, values, inputs, t0: int, t1: int):
        p, n = self.p, self.n
        counts = torch.zeros((v.shape[0], len(self.readout)), device=v.device)
        for t in range(t0, t1):
            refractory = last > t - p.ref_steps
            v = torch.where(refractory, p.v_reset, (v - p.v_rest) * self.decay_v + p.v_rest + x * self.x_to_v)
            s = Spike.apply((v - p.v_th) / (p.v_th - p.v_rest)) * ~refractory
            fired = s.detach() > 0
            v = torch.where(fired, p.v_reset, v)
            last = torch.where(fired, t, last)
            x = Propagate.apply(x * self.decay_x, s, values, self.pattern)
            row, line = inputs[t].nonzero(as_tuple=True)
            x = x.reshape(-1).index_add(0, row * n + self.target[line], self.kick[line]).view(x.shape)
            counts = counts + s.index_select(1, self.readout)
        return v, x, last, counts

    def rest(self, batch: int):
        v = torch.full((batch, self.n), self.p.v_rest, device=self.device)
        x = torch.zeros((batch, self.n), device=self.device)
        last = torch.full((batch, self.n), -(10**9), dtype=torch.int64, device=self.device)
        return v, x, last

    def forward(self, inputs, state=None) -> torch.Tensor:
        return self.run(inputs, state)[0]

    def run(self, inputs, state=None):
        # inputs[t] -> (batch, lines) bool: which input lines fire at step t. A brain that
        # is carrying on from an earlier board starts from the state that one left, and
        # the state this one ends in comes back, clocked from its own first step.
        steps, batch = inputs.shape[0], inputs.shape[1]
        inputs = inputs.to(self.device)
        v, x, last = self.rest(batch) if state is None else tuple(part.to(self.device) for part in state)
        values, total = self.values(), 0
        for t0 in range(0, steps, SEGMENT):
            args = (v, x, last, values, inputs, t0, min(t0 + SEGMENT, steps))
            if torch.is_grad_enabled():
                v, x, last, counts = checkpoint(self.segment, *args, use_reentrant=False, determinism_check="none")
            else:
                v, x, last, counts = self.segment(*args)
            total = total + counts
        return total, (v.detach(), x.detach(), last.detach() - steps)
