import numpy as np
import pytest

from connectome.load import NPZ_PATH, load
from sim.feeding import MN9, run

pytestmark = [
    pytest.mark.data,
    pytest.mark.slow,
    pytest.mark.skipif(not NPZ_PATH.exists(), reason="run python -m connectome.build first"),
]


def test_sugar_drives_mn9_and_bitter_holds_it_back():
    c = load(min_syn=5)
    rates = run(c, seconds=0.5, device="cpu")
    mn9 = c.cell_type == MN9
    assert mn9.sum() == 2
    assert (rates["nothing"][mn9] == 0).all()
    assert (rates["sugar"][mn9] > 100).all()
    assert (rates["bitter"][mn9] == 0).all()
    assert rates["sugar + bitter"][mn9].mean() < 0.5 * rates["sugar"][mn9].mean()
    assert (rates["sugar"] > 0).mean() < 0.1
