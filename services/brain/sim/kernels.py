import os
import subprocess
import sys
import warnings

import numpy as np
import scipy.sparse as sp
import torch

if sys.platform == "darwin" and "SDKROOT" not in os.environ:
    # torch.compile's C++ compiler can't find the standard headers on macOS without it
    sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True)
    if sdk.returncode == 0:
        os.environ["SDKROOT"] = sdk.stdout.strip()


def find(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # numpy scans a boolean array several times faster than torch.nonzero
    flat = torch.from_numpy(np.flatnonzero(mask.numpy()))
    return flat // mask.shape[1], flat % mask.shape[1]


class Rows:
    # a sparse matrix whose rows are sent whole: the targets of one spiking neuron or line
    def __init__(self, m: np.ndarray, dtype):
        m = sp.csr_array(m)
        self.indptr = torch.from_numpy(m.indptr.astype(np.int64))
        self.indices = torch.from_numpy(m.indices.astype(np.int64))
        self.data = torch.from_numpy(m.data).to(dtype)

    def send(self, x: torch.Tensor, row: torch.Tensor, source: torch.Tensor, scale=None) -> None:
        start, length = self.indptr[source], self.indptr[source + 1] - self.indptr[source]
        first = torch.cumsum(length, 0) - length
        pos = torch.repeat_interleave(start - first, length) + torch.arange(int(length.sum()))
        values = self.data[pos] if scale is None else self.data[pos] * torch.repeat_interleave(scale[row], length)
        x.view(-1).index_add_(0, torch.repeat_interleave(row, length) * x.shape[1] + self.indices[pos], values)


def integrate(v, x, last, t, rest, decay_v: float, decay_x: float, x_to_v: float, v_reset: float, v_th: float, ref_steps: int):
    # one step for every neuron at once; a refractory neuron sits at v_reset
    refractory = last > t - ref_steps
    v = torch.where(refractory, v_reset, (v - rest) * decay_v + rest + x * x_to_v)
    spike = (v > v_th) & ~refractory
    return torch.where(spike, v_reset, v), x * decay_x, torch.where(spike, t, last), spike


class Fused:
    # the compiled step fuses a dozen passes over the whole state into one;
    # without a working compiler it falls back to the same function uncompiled
    def __init__(self, fn):
        self.fn, self.compiled = fn, torch.compile(fn, dynamic=True)

    def __call__(self, *args):
        if self.compiled is not None:
            try:
                return self.compiled(*args)
            except Exception as error:
                warnings.warn(f"torch.compile unavailable, running uncompiled: {type(error).__name__}")
                self.compiled = None
        return self.fn(*args)


step = Fused(integrate)
