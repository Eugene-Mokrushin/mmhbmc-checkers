import numpy as np
import scipy.sparse as sp
import torch

from connectome.extract import MushroomBody, extract
from connectome.load import load
from progress import Progress
from sim.fast import FastLIF
from sim.inputs import poisson, projection, random_lines
from sim.params import LIF

STEPS = 1000
BATCH = 32
ACTIVE_LINES = (4, 8, 12, 16, 24)


def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a & b).sum(-1) / np.maximum((a | b).sum(-1), 1)


def without_apl(mb: MushroomBody) -> sp.csr_array:
    keep = (mb.population != "APL").astype(np.float32)
    return (sp.diags_array(keep) @ mb.graph.weights()).tocsr()


def kc_codes(sim: FastLIF, mb: MushroomBody, lines: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = sim.p
    counts = sim.counts(torch.from_numpy(poisson(lines, STEPS, p.input_rate, p.dt, rng))).numpy()
    return counts[:, mb.members("KC")] > 0


def measure(sim: FastLIF, mb: MushroomBody, active: int, rng: np.random.Generator) -> dict:
    lines = random_lines(BATCH, active, rng)
    first, again = kc_codes(sim, mb, lines, rng), kc_codes(sim, mb, lines, rng)
    return {
        "active": first.mean(),
        "overlap": jaccard(first, np.roll(first, 1, axis=0)).mean(),
        "repeat": jaccard(first, again).mean(),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    c = load()
    mb = extract(c)
    proj = projection(c, mb, rng)
    sims = {"APL": FastLIF(mb.graph.weights(), proj, LIF()), "no APL": FastLIF(without_apl(mb), proj, LIF())}
    rows = []
    with Progress(len(sims) * len(ACTIVE_LINES), "sparsity gate") as bar:
        for name, sim in sims.items():
            for active in ACTIVE_LINES:
                rows.append((name, active, measure(sim, mb, active, rng)))
                bar.update(active_kc=rows[-1][2]["active"])
    print(f"{'':8}{'lines':>6}{'KCs active':>12}{'overlap':>10}{'repeat':>9}")
    for name, active, r in rows:
        print(f"{name:8}{active:>6}{r['active']:>12.1%}{r['overlap']:>10.2f}{r['repeat']:>9.2f}")


if __name__ == "__main__":
    main()
