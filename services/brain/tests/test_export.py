import pandas as pd
import pytest
from conftest import write_csv

from connectome.coordinates import COORDINATES, one_per_neuron, read_markers
from connectome.export import neurons_json
from connectome.extract import extract


def test_read_markers_parses_positions(tmp_path):
    write_csv(
        tmp_path / COORDINATES,
        ["root_id", "position", "supervoxel_id"],
        [(7, "[631424 216180  55880]", 1)],
    )
    assert read_markers(tmp_path)[["root_id", "x", "y", "z"]].values.tolist() == [[7, 631424, 216180, 55880]]


def test_one_point_per_neuron_nearest_the_median():
    markers = pd.DataFrame(
        [(1, 10, 0, 0, 0), (1, 11, 10, 0, 0), (1, 12, 1000, 0, 0), (2, 20, 5, 5, 5)],
        columns=["root_id", "supervoxel_id", "x", "y", "z"],
    )
    points = one_per_neuron(markers)
    assert points.loc[1].tolist() == [10, 0, 0]
    assert points.loc[2].tolist() == [5, 5, 5]


def test_neurons_json(brain):
    mb = extract(brain)
    points = pd.DataFrame({"x": range(9), "y": 0, "z": 0}, index=pd.Index(range(100, 109), name="root_id"))
    doc = neurons_json(mb, points)
    assert doc["hemisphere"] == "right"
    assert [n["root_id"] for n in doc["neurons"]] == ["100", "101", "104", "105", "107"]
    assert doc["neurons"][2] == {
        "root_id": "104",
        "population": "MBON",
        "cell_type": "MBON07",
        "side": "left",
        "position": [4, 0, 0],
    }


def test_neurons_json_needs_every_position(brain):
    points = pd.DataFrame({"x": [0], "y": [0], "z": [0]}, index=pd.Index([100], name="root_id"))
    with pytest.raises(ValueError, match="no coordinates"):
        neurons_json(extract(brain), points)
