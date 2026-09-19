import argparse

import numpy as np
import scipy.sparse as sp
import torch

import progress
from connectome.coordinates import one_per_neuron, read_markers
from connectome.extract import extract
from connectome.load import load
from progress import Progress
from sim.eye import KICK, field, rates, rhythm
from sim.fast import FastLIF
from sim.kernels import best_device
from sim.params import LIF
from train.skills import auc, exam

CHUNK = 16


def stages(c, mb) -> dict[str, np.ndarray]:
    kc = mb.neurons[mb.members("KC")]
    visual_kc = kc[np.asarray(c.counts[np.flatnonzero(c.super_class == "visual_projection")][:, kc].sum(axis=0)).ravel() > 0]
    return {
        "photoreceptors": np.flatnonzero(np.isin(c.cell_type, ("R1-6", "R7", "R8"))),
        "optic lobe": np.flatnonzero(c.super_class == "optic"),
        "visual projection": np.flatnonzero(c.super_class == "visual_projection"),
        "visual Kenyon cells": visual_kc,
        "all Kenyon cells (right)": kc,
        "MBONs (right)": mb.neurons[mb.members("MBON")],
        "descending": np.flatnonzero(c.super_class == "descending"),
    }


def readout(x: np.ndarray, y: np.ndarray, train: np.ndarray) -> float:
    # the best linear readout of this stage (ridge, solved board by board since there
    # are more neurons than boards), scored on held-out boards
    x = (x - x[train].mean(axis=0)) / (x[train].std(axis=0) + 1e-9)
    kernel = x @ x[train].T
    alpha = np.linalg.solve(kernel[train] + x.shape[1] * np.eye(train.sum()), y[train] - y[train].mean())
    keep = ~train & (y != 0)
    return auc((kernel @ alpha)[keep], y[keep])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", type=int, default=1200)
    parser.add_argument("--ms", type=int, default=100)
    args = parser.parse_args()

    progress.use("seeing")
    c = load(min_syn=5)
    receptors, where = field(c, one_per_neuron(read_markers()))
    projection = sp.csr_array((np.full(len(receptors), KICK), (np.arange(len(receptors)), receptors)), shape=(len(receptors), c.n))
    sim = FastLIF(c.weights(), projection, LIF(input_gain=1.0), device=best_device())
    groups = stages(c, extract(load()))
    boards, safe, ahead = exam(args.boards, seed=5)
    phase = np.random.default_rng(0).random(len(receptors))
    steps = round(args.ms / 1000 / sim.p.dt)
    seen = {name: [] for name in groups}
    with Progress(len(boards), "seeing") as bar:
        for i in range(0, len(boards), CHUNK):
            chunk = boards[i : i + CHUNK]
            spikes = rhythm(rates(chunk, where, c.side[receptors]), steps, sim.p.dt, phase)
            counts = sim.counts(torch.from_numpy(spikes)).numpy()
            for name, idx in groups.items():
                seen[name].append(counts[:, idx])
            bar.update(len(chunk))
    train = np.random.default_rng(1).random(len(boards)) < 0.7
    print(f"{len(boards)} boards seen through the eyes for {args.ms} ms; best linear readout on held-out boards (AUC)")
    print(f"{'':26}{'neurons':>8}{'active':>8}{'safety':>8}{'material':>10}")
    for name, parts in seen.items():
        x = np.concatenate(parts).astype(float)
        print(f"{name:26}{x.shape[1]:>8}{(x > 0).mean():>8.1%}{readout(x, safe, train):>8.2f}{readout(x, ahead, train):>10.2f}")


if __name__ == "__main__":
    main()
