import warnings

import torch

warnings.filterwarnings("ignore", "Sparse CSR tensor support is in beta")
SLOPE = 10.0


class Spike(torch.autograd.Function):
    # a spike wherever the scaled potential is above threshold; on the way back the
    # step is replaced by a smooth bump around threshold (Zenke & Ganguli 2018)
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, grad):
        (u,) = ctx.saved_tensors
        return grad / (1 + SLOPE * u.abs()) ** 2


class Pattern:
    # the fixed sparse pattern of connections, row = presynaptic neuron
    def __init__(self, indptr, indices, n: int, device):
        self.n = n
        self.indptr = torch.as_tensor(indptr, dtype=torch.int64, device=device)
        self.indices = torch.as_tensor(indices, dtype=torch.int64, device=device)
        self.lengths = self.indptr[1:] - self.indptr[:-1]

    def entries(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # every connection leaving each listed neuron, and which listed neuron it belongs to
        length = self.lengths[source]
        owner = torch.repeat_interleave(torch.arange(len(source), device=source.device), length)
        start = torch.repeat_interleave(self.indptr[source] - torch.cumsum(length, 0) + length, length)
        return start + torch.arange(len(owner), device=source.device), owner


class Propagate(torch.autograd.Function):
    # x[b, post] += spikes[b, pre] * values[pre -> post], added in the same order as the
    # fast simulator; only neurons that fired are sent forward, but the gradient reaches
    # every neuron and every weight
    @staticmethod
    def forward(ctx, x, spikes, values, pattern: Pattern):
        batch, source = spikes.nonzero(as_tuple=True)
        where, owner = pattern.entries(source)
        row, pre = batch[owner], source[owner]
        out = x.clone()
        contribution = values[where] * spikes[row, pre]
        out.view(-1).index_add_(0, row * pattern.n + pattern.indices[where], contribution)
        ctx.save_for_backward(values, where, row, spikes[row, pre])
        ctx.pattern = pattern
        return out

    @staticmethod
    def backward(ctx, grad):
        values, where, row, strength = ctx.saved_tensors
        p = ctx.pattern
        weights = torch.sparse_csr_tensor(p.indptr, p.indices, values, (p.n, p.n), check_invariants=False)
        grad_spikes = (weights @ grad.T.contiguous()).T
        grad_values = torch.zeros_like(values)
        grad_values.index_add_(0, where, grad[row, p.indices[where]] * strength)
        return grad, grad_spikes, grad_values, None
