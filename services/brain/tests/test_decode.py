import numpy as np

from connectome.extract import extract
from game.decode import score, valence


def test_mbon_in_a_reward_compartment_signals_avoid(brain):
    assert valence(extract(brain)).tolist() == [-1]


def test_score_is_approach_minus_avoid(brain):
    mb = extract(brain)
    counts = np.zeros((2, mb.graph.n), dtype=np.int32)
    counts[:, mb.members("MBON")] = [[3], [5]]
    assert score(counts, mb, np.array([-1])).tolist() == [-3, -5]
    assert score(counts, mb, np.array([1])).tolist() == [3, 5]
