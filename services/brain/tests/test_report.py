import json

import numpy as np
import pytest

from train.evaluate import COLUMNS
from train.report import compare, load_runs


def fake_run(root, name, control, seed, start, end):
    d = root / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"control": control, "seed": seed}))
    lines = [",".join(COLUMNS), f"1,0,{start},{start},0,,,,", f"9,2048,{end},{end},0,0.5,0.07,20,0.01"]
    (d / "log.csv").write_text("\n".join(lines) + "\n")


@pytest.fixture
def runs(tmp_path):
    for seed in range(4):
        fake_run(tmp_path, f"c-real-{seed}", "real", seed, 0.45, 0.75 + 0.01 * seed)
        fake_run(tmp_path, f"c-degree-{seed}", "degree", seed, 0.45, 0.55 + 0.01 * seed)
    fake_run(tmp_path, "other", "real", 0, 0.1, 0.1)
    return tmp_path


def test_load_runs_filters_by_pattern(runs):
    assert len(load_runs("c-*", runs)) == 8
    assert len(load_runs("*", runs)) == 9


def test_compare_reports_gain_and_a_test_against_real(runs):
    rows = {r["control"]: r for r in compare(load_runs("c-*", runs), "random")}
    assert set(rows) == {"real", "degree"}
    assert np.allclose(rows["real"]["gain"], [0.30, 0.31, 0.32, 0.33])
    assert np.isnan(rows["real"]["p"])
    assert rows["degree"]["p"] < 0.001
