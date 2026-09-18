import gzip

import numpy as np
import pytest
from tiny_brain import tiny_brain

from connectome import raw
from connectome.build import build
from connectome.graph import ANNOTATIONS

NEURONS = {
    30: ("left", "MBON01", ""),
    10: ("left", "KCg-m", "ACH"),
    20: ("right", "APL", "GABA"),
    40: ("right", "", ""),
}
EDGES = [
    (10, 20, "MB_CA_L", 3, "ACH"),
    (10, 20, "MB_ML_L", 3, "ACH"),
    (20, 10, "MB_CA_L", 4, "GABA"),
    (30, 10, "SMP_L", 7, "GLUT"),
    (10, 30, "SMP_L", 1, "ACH"),
]


def write_csv(path, header, rows):
    with gzip.open(path, "wt") as f:
        f.write(",".join(header) + "\n")
        f.writelines(",".join(map(str, row)) + "\n" for row in rows)


@pytest.fixture
def make_raw(tmp_path):
    def make(edges=EDGES):
        write_csv(
            tmp_path / raw.CLASSIFICATION,
            ["root_id", "flow", "super_class", "class", "sub_class", "hemilineage", "side", "nerve"],
            [(rid, "intrinsic", "central", "", "", "", side, "") for rid, (side, _, _) in NEURONS.items()],
        )
        write_csv(
            tmp_path / raw.CELL_TYPES,
            ["root_id", "primary_type", "additional_type(s)"],
            [(rid, t, "") for rid, (_, t, _) in NEURONS.items() if t],
        )
        write_csv(
            tmp_path / raw.TRANSMITTERS,
            ["root_id", "group", "nt_type", "nt_type_score"],
            [(rid, "MB", nt, 0.9 if nt else 0.0) for rid, (_, _, nt) in NEURONS.items()],
        )
        write_csv(tmp_path / raw.EDGES, ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"], edges)
        return tmp_path

    return make


@pytest.fixture
def tiny(make_raw):
    return build(make_raw(), verify=False)


@pytest.fixture
def brain():
    return tiny_brain()


def assert_same(a, b):
    assert a.meta == b.meta
    assert (a.counts != b.counts).nnz == 0
    for name in ("root_id", "sign", "nt_confident", *ANNOTATIONS):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
