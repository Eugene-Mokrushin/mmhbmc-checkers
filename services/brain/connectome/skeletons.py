import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import progress
from connectome.load import load
from paths import DATA_DIR
from progress import Progress

WHERE = "https://storage.googleapis.com/flywire-data/codex/skeletons/fafb/{kind}/{root}.swc"
KINDS = ("l2", "lod1")  # the coarse skeleton first; the full one only where there is none
STORE = DATA_DIR / "skeletons.npz"
KEEP = 64  # points kept per neuron, spread evenly along the skeleton FlyWire publishes
HANDS = 16


def thinned(text: str, keep: int = KEEP) -> np.ndarray:
    # an SWC row is: number, kind, x, y, z, radius, parent — the places are what we want
    rows = [line.split() for line in text.splitlines() if line[:1].isdigit()]
    xyz = np.array([row[2:5] for row in rows], dtype=np.float32) * 1000.0
    return xyz if len(xyz) <= keep else xyz[np.linspace(0, len(xyz) - 1, keep).astype(int)]


def ask(root: int, kind: str, tries: int = 3) -> np.ndarray | None:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(WHERE.format(kind=kind, root=root), timeout=60) as reply:
                return thinned(reply.read().decode())
        except urllib.error.HTTPError as refused:
            if refused.code == 404:
                return None
        except (urllib.error.URLError, OSError, ValueError):
            pass
    return None


def fetch(root: int) -> np.ndarray:
    for kind in KINDS:
        found = ask(root, kind)
        if found is not None and len(found):
            return found
    return np.zeros((0, 3), dtype=np.float32)


def keep(root_id: np.ndarray, drawn: list[np.ndarray]) -> None:
    counts = np.array([len(one) for one in drawn], dtype=np.int32)
    xyz = np.concatenate(drawn) if len(drawn) else np.zeros((0, 3), dtype=np.float32)
    np.savez_compressed(STORE, root_id=root_id[: len(drawn)], counts=counts, xyz=xyz.astype(np.float32))


def held(root_id: np.ndarray) -> list[np.ndarray]:
    # what an earlier run already fetched, so a second run only fills the gaps
    if not STORE.exists():
        return [np.zeros((0, 3), dtype=np.float32) for _ in root_id]
    with np.load(STORE) as kept:
        ends = np.cumsum(kept["counts"])
        have = {int(root): kept["xyz"][end - n : end] for root, n, end in zip(kept["root_id"], kept["counts"], ends)}
    return [have.get(int(root), np.zeros((0, 3), dtype=np.float32)) for root in root_id]


def main() -> None:
    # FlyWire draws each cell as a skeleton, not a dot. This keeps a thinned copy of
    # theirs, so the website can draw a neuron where it actually runs.
    progress.use("skeletons")
    c = load()
    drawn = held(c.root_id)
    wanted = [i for i, one in enumerate(drawn) if not len(one)]
    print(f"{len(c.root_id) - len(wanted)} already here, {len(wanted)} to fetch")
    with ThreadPoolExecutor(HANDS) as hands, Progress(len(wanted), "skeletons") as bar:
        for done, one in enumerate(hands.map(fetch, [int(c.root_id[i]) for i in wanted]), start=1):
            drawn[wanted[done - 1]] = one
            bar.update(1, points=len(one))
            if done % 5000 == 0:
                keep(c.root_id, drawn)
    keep(c.root_id, drawn)
    missing = sum(1 for one in drawn if not len(one))
    print(f"{len(drawn) - missing} skeletons, {sum(len(one) for one in drawn)} points, {missing} without one")


if __name__ == "__main__":
    main()
