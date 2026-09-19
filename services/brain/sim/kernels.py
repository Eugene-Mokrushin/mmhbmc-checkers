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


def best_device() -> str:
    # Apple's GPU isn't faster for this, so outside CUDA the CPU it is
    return "cuda" if torch.cuda.is_available() else "cpu"


def find(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if mask.device.type == "cpu":
        # numpy scans a boolean array several times faster than torch.nonzero
        flat = torch.from_numpy(np.flatnonzero(mask.numpy()))
        return flat // mask.shape[1], flat % mask.shape[1]
    return mask.nonzero(as_tuple=True)


class Rows:
    # a sparse matrix whose rows are sent whole: the targets of one spiking neuron or line
    def __init__(self, m, dtype, device="cpu"):
        m = sp.csr_array(m)
        self.shape = m.shape
        self.indptr = torch.from_numpy(m.indptr.astype(np.int64)).to(device)
        self.indices = torch.from_numpy(m.indices.astype(np.int64)).to(device)
        self.data = torch.from_numpy(m.data).to(dtype).to(device)

    def send(self, x: torch.Tensor, row: torch.Tensor, source: torch.Tensor, scale=None) -> None:
        start, length = self.indptr[source], self.indptr[source + 1] - self.indptr[source]
        first = torch.cumsum(length, 0) - length
        pos = torch.repeat_interleave(start - first, length) + torch.arange(int(length.sum()), device=x.device)
        values = self.data[pos] if scale is None else self.data[pos] * torch.repeat_interleave(scale[row], length)
        x.view(-1).index_add_(0, torch.repeat_interleave(row, length) * x.shape[1] + self.indices[pos], values)

    def locate(self, pre: np.ndarray, post: np.ndarray) -> torch.Tensor:
        # where each pre -> post weight sits in the data array, -1 if they aren't connected
        indptr, indices = self.indptr.cpu().numpy(), self.indices.cpu().numpy()
        column = np.full(self.shape[1], -1)
        column[post] = np.arange(len(post))
        where = np.full((len(pre), len(post)), -1)
        for i, row in enumerate(pre):
            span = np.arange(indptr[row], indptr[row + 1])
            j = column[indices[span]]
            where[i, j[j >= 0]] = span[j >= 0]
        return torch.from_numpy(where).to(self.data.device)

    def read(self, where: torch.Tensor) -> torch.Tensor:
        return torch.where(where >= 0, self.data[where.clamp(min=0)], 0.0)

    def write(self, where: torch.Tensor, values: torch.Tensor) -> None:
        self.data[where[where >= 0]] = values[where >= 0].to(self.data.dtype)


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
