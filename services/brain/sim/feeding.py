import argparse

import numpy as np
import scipy.sparse as sp
import torch

from connectome.graph import Connectome
from connectome.load import load
from sim.fast import FastLIF
from sim.inputs import poisson
from sim.kernels import best_device
from sim.params import LIF

# Shiu et al. 2024: taste neurons driven at 150 Hz, strongly enough that each input
# spike fires them. Sugar should make MN9, the motor neuron behind feeding
# (FlyWire 720575940660219265, type CB0701), fire; bitter alongside should hold it back.
RATE = 150.0
KICK = 250
MN9 = "CB0701"
CONDITIONS = {"nothing": (), "sugar": ("sugar",), "bitter": ("bitter",), "sugar + bitter": ("sugar", "bitter")}


def taste(c: Connectome, device: str = "cpu") -> tuple[FastLIF, dict[str, np.ndarray]]:
    gustatory = c.cell_class == "gustatory"
    groups = {name: np.flatnonzero(gustatory & (c.sub_class == sub)) for name, sub in (("sugar", "sugar/water"), ("bitter", "bitter"))}
    lines = np.concatenate(list(groups.values()))
    projection = sp.csr_array((np.full(len(lines), KICK), (np.arange(len(lines)), lines)), shape=(len(lines), c.n))
    first, spans = 0, {}
    for name, members in groups.items():
        spans[name] = np.arange(first, first + len(members))
        first += len(members)
    return FastLIF(c.weights(), projection, LIF(input_gain=1.0), device=device), spans


def run(c: Connectome, seconds: float, device: str) -> dict[str, np.ndarray]:
    sim, spans = taste(c, device)
    active = np.zeros((len(CONDITIONS), sim.n_lines), dtype=bool)
    for row, groups in enumerate(CONDITIONS.values()):
        for group in groups:
            active[row, spans[group]] = True
    steps = round(seconds / sim.p.dt)
    inputs = poisson(active, steps, RATE, sim.p.dt, np.random.default_rng(0))
    counts = sim.counts(torch.from_numpy(inputs)).numpy() / seconds
    return dict(zip(CONDITIONS, counts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-syn", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    c = load(min_syn=args.min_syn)
    rates = run(c, args.seconds, best_device())
    mn9 = c.cell_type == MN9
    print(f"whole brain, {c.counts.nnz:,} connections (min_syn {args.min_syn}), {args.seconds} s per condition")
    for name, r in rates.items():
        print(f"{name:>15}: MN9 left/right {' / '.join(f'{v:.0f}' for v in r[mn9])} Hz | brain: "
              f"{(r > 0).mean():6.2%} of neurons fired, mean {r.mean():.2f} Hz")
    sugar, both, rest = (rates[k][mn9].mean() for k in ("sugar", "sugar + bitter", "nothing"))
    print(f"gate: sugar drives MN9 {'PASS' if sugar > 10 and rest == 0 else 'FAIL'}, "
          f"bitter holds it back {'PASS' if both < 0.5 * sugar else 'FAIL'}")


if __name__ == "__main__":
    main()
